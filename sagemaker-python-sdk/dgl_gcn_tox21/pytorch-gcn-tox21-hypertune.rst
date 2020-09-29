Hyperparameter tuning with Amazon SageMaker for molecular property prediction
=============================================================================

Contents
--------

1. `Background <##Background>`__
2. `Setup <##Setup>`__
3. `Code <##Code>`__
4. `Tune <##Tune>`__
5. `Wrap-up <##Wrap-up>`__

Background
----------

This example notebook demonstrates a graph-based molecular property
prediction model with automatic hyperparameter tuning. The
implementation is based on DGL and PyTorch. To find the best
hyperparameters, it leverages SageMaker to kick off multiple training
jobs with different hyperparameter combinations. In this example, you
use the `Amazon SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__ to create a
hyperparameter tuning job.

Setup
-----

This notebook was created and tested on an ml.p3.2xlarge notebook
instance.

Prerequisites \* Before you start this tutorial, review the
``pytorch-gcn-tox21.ipynb`` example and ensure you have an account under
your Amazon Elastic Container Registry (Amazon ECR) specified by
{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-dgl-pytorch-gcn-tox21:latest.
\* An S3 bucket and prefix exists that you want to use for training and
model data. This should be within the same Region as the notebook
instance, training, and hosting. \* An IAM role ARN exists that you are
going to use to give training and hosting access to your data. See the
documentation for more details on creating these. Note that if a role is
not associated with the current notebook instance, or more than one role
is required for training or hosting, you should replace
sagemaker.get_execution_role() with the appropriate full IAM role ARN
strings.

.. code:: ipython3

    import sagemaker
    
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # Setup session
    sess = sagemaker.Session()
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = sess.default_bucket()
    
    # Location to put your custom code.
    custom_code_upload_location = 'customcode'
    
    # IAM execution role that gives Amazon SageMaker access to resources in your AWS account.
    # Use the Amazon SageMaker Python SDK to get the role from the notebook environment. 
    role = get_execution_role()

Code
----

To run Docker containers with Amazon SageMaker, provide a Python script
for the container to run. In this example, ``main.py`` provides all the
code you need to train an Amazon SageMaker model.

.. code:: ipython3

    !cat main.py

Tune
----

Similar to training a single training job in Amazon SageMaker, Define
your training estimator passing in the code scripts, IAM role, (per job)
hardware configuration, and any hyperparameters that you are not tuning.

You must have a Docker image in your Amazon Elastic Container Registry
(Amazon ECR) following steps in pytorch-gcn-tox21.ipynb.

.. code:: ipython3

    # Set target dgl-docker name
    docker_name='sagemaker-dgl-pytorch-gcn-tox21'
    
    CODE_PATH = 'main.py'
    code_location = sess.upload_data(CODE_PATH, bucket=bucket, key_prefix=custom_code_upload_location)
    
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, docker_name)
    
    estimator = sagemaker.estimator.Estimator(image,
                                              role, 
                                              train_instance_count=1, 
                                              train_instance_type='ml.p3.2xlarge',
                                              hyperparameters={'entrypoint': CODE_PATH},
                                              sagemaker_session=sess)

After you define your estimator, specify the hyperparameters that you
want to tune and their possible values. Depending on the type of
possible values, the hyperparameters can be divided into three classes:

-  **Categorical**: Its possible values form a discrete set and is
   represented by ``CategoricalParameter(list)``.
-  **Continuous**: It can take any real number within an interval
   ``[min, max]`` and is represented by
   ``ContinuousParameter(min, max)``.
-  **Integer**: It can take any integer value within an interval
   ``[min, max]`` and is represented by ``IntegerParameter(min, max)``.

Note that it’s almost always better to specify a value as the least
restrictive type. For example, ``ContinuousParameter(0.01, 0.2)`` is
less restrictive than ``CategoricalParameter([0.01, 0.1, 0.15, 0.2])``.

.. code:: ipython3

    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter
    
    hyper_ranges = {'lr': ContinuousParameter(1e-4, 1e-2),
                    'patience': IntegerParameter(5, 30),
                    'n_hidden': CategoricalParameter([32, 64, 128])}

Next, specify the objective metric to tune and its definition. This
includes the regular expression (regex) needed to extract that metric
from the Amazon CloudWatch logs of the training job.

.. code:: ipython3

    objective_name = 'Validation_roc_auc'
    metric_definitions = [{'Name': objective_name,
                           'Regex': 'Best validation score ([0-9\\.]+)'}]

Now, create a ``HyperparameterTuner`` object, which you pass:

-  The training estimator you created above
-  The hyperparameter ranges
-  Objective metric name and definition
-  Number of training jobs to run in total and how many training jobs
   should be run simultaneously. More parallel jobs will finish tuning
   sooner, but may sacrifice accuracy. We recommend you set the parallel
   jobs value to less than 10 percent of the total number of training
   jobs. It is set higher just for this example to keep it short.
-  Whether you should maximize or minimize the objective metric. You
   haven’t specified here since it defaults to ‘Maximize’, which is what
   you want for validation roc-auc)

You can also add a task_tag with value ‘DGL’ to help tracking the
hyperparameter tuning task.

.. code:: ipython3

    from sagemaker.tuner import HyperparameterTuner
    
    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    tuner = HyperparameterTuner(estimator,
                                objective_name,
                                hyper_ranges,
                                metric_definitions,
                                tags=task_tags,
                                max_jobs=6,
                                max_parallel_jobs=2)

Finally, start the tuning job by calling ``.fit()``.

.. code:: ipython3

    tuner.fit(inputs={'training-code': code_location})

Check the hyperparameter tuning jobs status to make sure it started
successfully and is InProgress.

.. code:: ipython3

    import boto3
    
    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

Wrap-up
-------

After the hyperparameter tuning job is started, it runs in the
background and you can close this notebook. When it’s finished, you can
go to console to analyze the result.

For more information about Amazon SageMaker’s Hyperparameter Tuning, see
the AWS documentation.
