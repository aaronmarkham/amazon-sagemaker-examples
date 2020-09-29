Hyperparameter Tuning with Your Own Container in Amazon SageMaker
=================================================================

**Using Amazon SageMaker’s Hyperparameter Tuning with a customer Docker
container and R algorithm**

--------------

--------------

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Permissions <#Permissions>`__
4. `Code <#Code>`__
5. `Publish <#Publish>`__
6. `Data <#Data>`__
7. `Tune <#Tune>`__
8. `Wrap-up <#Wrap-up>`__

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

.. code:: sh

    %%sh
    
    # The name of our algorithm
    algorithm_name=rmars
    
    #set -e # stop if anything fails
    account=$(aws sts get-caller-identity --query Account --output text)
    
    # Get the region defined in the current configuration (default to us-west-2 if none defined)
    region=$(aws configure get region)
    region=${region:-us-west-2}
    
    if [ "$region" = "cn-north-1" ] || [ "$region" = "cn-northwest-1" ]; then domain="amazonaws.com.cn"; 
    else domain="amazonaws.com"; fi
    
    fullname="${account}.dkr.ecr.${region}.${domain}/${algorithm_name}:latest"
    
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

Let’s copy the data to S3 so that SageMaker training can access it.

.. code:: ipython3

    train_file = 'iris.csv'
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

    domain = "amazonaws.com.cn" if (region == "cn-north-1" or region == "cn-northwest-1") else "amazonaws.com"
    
    estimator = sagemaker.estimator.Estimator(
        image_name='{}.dkr.ecr.{}.{}/rmars:latest'.format(account, region, domain),
        role=role,
        train_instance_count=1,
        train_instance_type='ml.m4.xlarge',
        output_path='s3://{}/{}/output'.format(bucket, prefix),
        sagemaker_session=sagemaker.Session(),
        hyperparameters={'target': 'Sepal.Length'})

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

    hyperparameter_ranges = {'degree': IntegerParameter(1, 3),
                             'thresh': ContinuousParameter(0.001, 0.01),
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

    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

--------------

Wrap-up
-------

Now that we’ve started our hyperparameter tuning job, it will run in the
background and we can close this notebook. Once finished, we can use the
`HPO Analysis
notebook <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb>`__
to determine which set of hyperparameters worked best.

For more detail on Amazon SageMaker’s Hyperparameter Tuning, please
refer to the AWS documentation.
