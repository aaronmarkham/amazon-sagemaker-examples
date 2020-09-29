Hyperparameter Tuning Your Own R Algorithm with Your Own Container in Amazon SageMaker
======================================================================================

**Using Amazon SageMaker’s Hyperparameter Tuning with a customer Docker
container and R algorithm**

--------------

--------------

Contents
--------

1.  `Background <#Background>`__
2.  `Setup <#Setup>`__
3.  `Permissions <#Permissions>`__
4.  `Code <#Code>`__
5.  `Publish <#Publish>`__
6.  `Data <#Data>`__
7.  `Tune <#Tune>`__
8.  `HPO Analysis <#HPO-Analysis>`__
9.  `Host <#Host>`__
10. `Predict <#Predict>`__
11. `(Optional) Clean-up <#(Optional)-Clean-up>`__
12. `Wrap-up <#Wrap-up>`__

Setup
-----

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the notebook instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the
   `documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/using-identity-based-policies.html>`__
   for more details on creating these. Note, if a role not associated
   with the current notebook instance, or more than one role is required
   for training and/or hosting, please replace
   ``sagemaker.get_execution_role()`` with a the appropriate full IAM
   role arn string(s).

.. code:: ipython3

    import sagemaker
    
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-hpo-r-byo'
    
    role = sagemaker.get_execution_role()

Now we’ll import the libraries we’ll need for the remainder of the
notebook.

.. code:: ipython3

    import os
    import boto3
    import sagemaker
    import pandas as pd
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

Permissions
~~~~~~~~~~~

Running this notebook requires permissions in addition to the normal
``SageMakerFullAccess`` permissions. This is because we’ll be creating a
new repository in Amazon ECR. The easiest way to add these permissions
is simply to add the managed policy
``AmazonEC2ContainerRegistryFullAccess`` to the role associated with
your notebook instance. There’s no need to restart your notebook
instance when you do this, the new permissions will be available
immediately.

--------------

Code
----

For this example, we’ll need 3 supporting code files. We’ll provide just
a brief overview of what each one does. See the full R bring your own
notebook for more details.

-  **Fit**: ``mars.R`` creates functions to train and serve our model.
-  **Serve**: ``plumber.R`` uses the
   `plumber <https://www.rplumber.io/>`__ package to create a
   lightweight HTTP server for processing requests in hosting. Note the
   specific syntax, and see the plumber help docs for additional detail
   on more specialized use cases.
-  **Dockerfile**: This specifies the configuration for our docker
   container. Smaller containers are preferred for Amazon SageMaker as
   they lead to faster spin up times in training and endpoint creation,
   so this container is kept minimal. It simply starts with Ubuntu,
   installs R, mda, and plumber libraries, then adds ``mars.R`` and
   ``plumber.R``, and finally sets ``mars.R`` to run as the entrypoint
   when launched.

Publish
~~~~~~~

Now, to publish this container to ECR, we’ll run the comands below.

This command will take several minutes to run the first time.

.. code:: ipython3

    algorithm_name = 'rmars'

.. code:: sh

    %%sh
    
    # The name of our algorithm
    algorithm_name=rmars
    
    #set -e # stop if anything fails
    account=$(aws sts get-caller-identity --query Account --output text)
    
    # Get the region defined in the current configuration (default to us-west-2 if none defined)
    region=$(aws configure get region)
    region=${region:-us-east-1}
    
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
    
    # If the repository doesn't exist in ECR, create it.
    aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
    
    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
    fi
    
    # Get the login command from ECR and execute it directly
    $(aws ecr get-login --region ${region} --no-include-email)
    
    # Build the docker image locally with the image name and then push it to ECR
    # with the full name.
    docker build  -t ${algorithm_name} .
    docker tag ${algorithm_name} ${fullname}
    
    docker push ${fullname}

--------------

Data
----

For this illustrative example, we’ll simply use ``iris``. This a
classic, but small, dataset used to test supervised learning algorithms.
Typically the goal is to predict one of three flower species based on
various measurements of the flowers’ attributes. Further detail can be
found `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`__.

Let’s split the data to train and test datasets (70% / 30%) and then
copy the data to S3 so that SageMaker training can access it.

.. code:: ipython3

    data = pd.read_csv('iris.csv')

.. code:: ipython3

    # Train/test split, 70%-30%
    train_data = data.sample(frac=0.7, random_state=42)
    test_data = data.drop(train_data.index)
    test_data.head()

.. code:: ipython3

    # Write to csv
    train_data.to_csv('iris_train.csv', index=False)
    test_data.to_csv('iris_test.csv', index=False)

.. code:: ipython3

    # write to S3
    train_file = 'iris_train.csv'
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_file(train_file)


*Note: Although we could do preliminary data transformations in the
notebook, we’ll avoid doing so, instead choosing to do those
transformations inside the container. This is not typically the best
practice for model efficiency, but provides some benefits in terms of
flexibility.*

--------------

Tune
----

Now, let’s setup the information needed to train a Multivariate Adaptive
Regression Splines model on ``iris`` data. In this case, we’ll predict
``Sepal.Length`` rather than the more typical classification of
``Species`` in order to show how factors might be included in a model
and to limit the use case to regression.

First, we’ll get our region and account information so that we can point
to the ECR container we just created.

.. code:: ipython3

    region = boto3.Session().region_name
    account = boto3.client('sts').get_caller_identity().get('Account')

Now we’ll create an estimator using the `SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__. This allows us to
specify: - The training container image in ECR - The IAM role that
controls permissions for accessing the S3 data and executing SageMaker
functions - Number and type of training instances - S3 path for model
artifacts to be output to - Any hyperparameters that we want to have the
same value across all training jobs during tuning

.. code:: ipython3

    estimator = sagemaker.estimator.Estimator(
        image_name='{}.dkr.ecr.{}.amazonaws.com/rmars:latest'.format(account, region),
        role=role,
        train_instance_count=1,
        train_instance_type='ml.m4.xlarge',
        output_path='s3://{}/{}/output'.format(bucket, prefix),
        sagemaker_session=sagemaker.Session(),
        hyperparameters={'degree': 2})      # Setting constant hyperparameter
    
    # target is by defauld "Sepal.Length". See mars.R where this is set.

Once we’ve defined our estimator we can specify the hyperparameters that
we’d like to tune and their possible values. We have three different
types of hyperparameters. - Categorical parameters need to take one
value from a discrete set. We define this by passing the list of
possible values to ``CategoricalParameter(list)`` - Continuous
parameters can take any real number value between the minimum and
maximum value, defined by ``ContinuousParameter(min, max)`` - Integer
parameters can take any integer value between the minimum and maximum
value, defined by ``IntegerParameter(min, max)``

*Note, if possible, it’s almost always best to specify a value as the
least restrictive type. For example, tuning ``thresh`` as a continuous
value between 0.01 and 0.2 is likely to yield a better result than
tuning as a categorical parameter with possible values of 0.01, 0.1,
0.15, or 0.2.*

.. code:: ipython3

    # to set the degree as a varying HP to tune, use: 'degree': IntegerParameter(1, 3) and remove it from the Estimator
    
    hyperparameter_ranges = {'thresh': ContinuousParameter(0.001, 0.01),
                             'prune': CategoricalParameter(['TRUE', 'FALSE'])}

Next we’ll specify the objective metric that we’d like to tune and its
definition. This metric is output by a ``print`` statement in our
``mars.R`` file. Its critical that the format aligns with the regular
expression (Regex) we then specify to extract that metric from the
CloudWatch logs of our training job.

.. code:: ipython3

    objective_metric_name = 'mse'
    metric_definitions = [{'Name': 'mse',
                           'Regex': 'mse: ([0-9\\.]+)'}]

Now, we’ll create a ``HyperparameterTuner`` object, which we pass: - The
MXNet estimator we created above - Our hyperparameter ranges - Objective
metric name and definition - Whether we should maximize or minimize our
objective metric (defaults to ‘Maximize’) - Number of training jobs to
run in total and how many training jobs should be run simultaneously.
More parallel jobs will finish tuning sooner, but may sacrifice
accuracy. We recommend you set the parallel jobs value to less than 10%
of the total number of training jobs (we’ll set it higher just for this
example to keep it short).

.. code:: ipython3

    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                objective_type='Minimize',
                                max_jobs=9,
                                max_parallel_jobs=3)

And finally, we can start our hyperparameter tuning job by calling
``.fit()`` and passing in the S3 paths to our train and test datasets.

*Note, typically for hyperparameter tuning, we’d want to specify both a
training and validation (or test) dataset and optimize the objective
metric from the validation dataset. However, because ``iris`` is a very
small dataset we’ll skip the step of splitting into training and
validation. In practice, doing this could lead to a model that overfits
to our training data and does not generalize well.*

.. code:: ipython3

    tuner.fit({'train': 's3://{}/{}/train'.format(bucket, prefix)})

Let’s just run a quick check of the hyperparameter tuning jobs status to
make sure it started successfully and is ``InProgress``.

.. code:: ipython3

    import time
    
    status = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
    
    while status != "Completed":
        status = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
        
        completed = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['TrainingJobStatusCounters']['Completed']
        
        prog = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['TrainingJobStatusCounters']['InProgress']
        
        print(f'{status}, Completed Jobs: {completed}, In Progress Jobs: {prog}')
        
        time.sleep(30)

Wait until the HPO job is complete, and then run the following cell:

.. code:: ipython3

    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['BestTrainingJob']

--------------

HPO Analysis
------------

Now that we’ve started our hyperparameter tuning job, it will run in the
background and we can close this notebook. Once finished, we can use the
`HPO Analysis
notebook <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb>`__
to determine which set of hyperparameters worked best.

For more detail on Amazon SageMaker’s Hyperparameter Tuning, please
refer to the AWS documentation.

--------------

Host
----

Hosting the model we just tuned takes three steps in Amazon SageMaker.
First, we define the model we want to host, pointing the service to the
model artifact our training job just wrote to S3.

We will use the results of the HPO for this purpose, but using
``hyper_parameter_tuning_job`` method.

.. code:: ipython3

    best_training = boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['BestTrainingJob']

.. code:: ipython3

    # Get the best trainig job and S3 location for the model file
    best_model_s3 = boto3.client('sagemaker').describe_training_job(
        TrainingJobName=best_training['TrainingJobName'])['ModelArtifacts']['S3ModelArtifacts']
    best_model_s3

.. code:: ipython3

    import time
    r_job = 'DEMO-r-byo-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

.. code:: ipython3

    r_hosting_container = {
        'Image': '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name),
        'ModelDataUrl': best_model_s3
    }
    
    create_model_response = boto3.client('sagemaker').create_model(
        ModelName=r_job,
        ExecutionRoleArn=role,
        PrimaryContainer=r_hosting_container)
    
    print(create_model_response['ModelArn'])

Next, let’s create an endpoing configuration, passing in the model we
just registered. In this case, we’ll only use a few c4.xlarges.

.. code:: ipython3

    r_endpoint_config = 'DEMO-r-byo-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print(r_endpoint_config)
    
    create_endpoint_config_response = boto3.client('sagemaker').create_endpoint_config(
        EndpointConfigName=r_endpoint_config,
        ProductionVariants=[{
            'InstanceType': 'ml.t2.medium',
            'InitialInstanceCount': 1,
            'ModelName': r_job,
            'VariantName': 'AllTraffic'}])
    
    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

Finally, we’ll create the endpoints using our endpoint configuration
from the last step.

.. code:: ipython3

    %%time
    
    r_endpoint = 'DEMO-r-endpoint-' + time.strftime("%Y%m%d%H%M", time.gmtime())
    print(r_endpoint)
    create_endpoint_response = boto3.client('sagemaker').create_endpoint(
        EndpointName=r_endpoint,
        EndpointConfigName=r_endpoint_config)
    print(create_endpoint_response['EndpointArn'])
    
    resp = boto3.client('sagemaker').describe_endpoint(EndpointName=r_endpoint)
    status = resp['EndpointStatus']
    print("Status: " + status)
    
    try:
        boto3.client('sagemaker').get_waiter('endpoint_in_service').wait(EndpointName=r_endpoint)
    finally:
        resp = boto3.client('sagemaker').describe_endpoint(EndpointName=r_endpoint)
        status = resp['EndpointStatus']
        print("Arn: " + resp['EndpointArn'])
        print("Status: " + status)
    
        if status != 'InService':
            raise Exception('Endpoint creation did not succeed')

--------------

Predict
-------

To confirm our endpoints are working properly, let’s try to invoke the
endpoint.

*Note: The payload we’re passing in the request is a CSV string with a
header record, followed by multiple new lines. It also contains text
columns, which the serving code converts to the set of indicator
variables needed for our model predictions. Again, this is not a best
practice for highly optimized code, however, it showcases the
flexibility of bringing your own algorithm.*

.. code:: ipython3

    import pandas as pd
    import json
    
    iris_test = pd.read_csv('iris_test.csv')
    
    runtime = boto3.Session().client('runtime.sagemaker')

.. code:: ipython3

    %%time 
    
    # there is a limit of max 500 samples at a time for invoking endpoints
    payload = iris_test.drop(['Sepal.Length'], axis=1).to_csv(index=False)
    
    response = runtime.invoke_endpoint(EndpointName=r_endpoint,
                                       ContentType='text/csv',
                                       Body=payload)
    
    result = json.loads(response['Body'].read().decode())
    display(result)

We can see the result is a CSV of predictions for our target variable.
Let’s compare them to the actuals to see how our model did.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    
    plt.scatter(iris_test['Sepal.Length'], np.fromstring(result[0], sep=','), alpha=0.4, s=50)
    plt.xlabel('Sepal Length(Actual)')
    plt.ylabel('Sepal Length(Prediction)')
    x = np.linspace(*plt.xlim())
    plt.plot(x, x, linestyle='--', color='g', linewidth=1)
    plt.xlim(4,8)
    plt.ylim(4,8)
    
    plt.show()

(Optional) Clean-up
~~~~~~~~~~~~~~~~~~~

If you’re ready to be done with this notebook, please run the cell
below. This will remove the hosted endpoint you created and avoid any
charges from a stray instance being left on.

.. code:: ipython3

    boto3.client('sagemaker').delete_endpoint(EndpointName=r_endpoint)

