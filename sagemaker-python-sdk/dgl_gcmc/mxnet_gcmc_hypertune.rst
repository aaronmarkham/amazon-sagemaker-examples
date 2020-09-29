Graph convolutional matrix completion hyperparameter tuning with Amazon SageMaker and Deep Graph Library with MXNet backend
===========================================================================================================================

**Creating a hyperparameter tuning job for a DGL network**

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Code <#Code>`__
4. `Tune <#Train>`__
5. `Wrap-up <#Wrap-up>`__

Background
----------

This example notebook focuses on how to create a graph neural network
(GNN) model to train `Graph Convolutional Matrix Completion
(GCMC) <https://arxiv.org/abs/1706.02263>`__ network using DGL with
mxnet backend with the `MovieLens
dataset <https://grouplens.org/datasets/movielens/>`__. It leverages
SageMaker’s hyperparameter tuning to kick off multiple training jobs
with different hyperparameter combinations, to find the set with best
model performance. This is an important step in the machine learning
process as hyperparameter settings can have a large impact on model
accuracy. In this example, you use the `SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__ to create a
hyperparameter tuning job for an Amazomn SageMaker estimator.

Setup
-----

This notebook is tested on an ml.p3.2xlarge notebook instance.

Prerequisites \* You should be able to successfully run the GCMC
example. You have your
{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-dgl-gcmc:latest under
your Amazon Elastic Container Registry (Amazon ECR) with a specific
account and Region. \* You have an S3 bucket and prefix that you want to
use for training and model data. This exists within the same Region as
the notebook instance, training, and hosting. \* You have established
the IAM role Amazon Resource Name (ARN) used to give training and
hosting access to your data. See the documentation for more details on
creating these. Note, if a role not associated with the current notebook
instance, or more than one role is required for training and hosting,
please replace sagemaker.get_execution_role() with a the appropriate
full IAM role ARN string.

.. code:: ipython3

    import sagemaker
    
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # Setup session
    sess = sagemaker.Session()
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here.
    bucket = sess.default_bucket()
    
    # Location to put your custom code.
    custom_code_upload_location = 'customcode'
    
    # Location where results of model training are saved.
    model_artifacts_location = 's3://{}/artifacts'.format(bucket)
    
    # IAM role that gives Amazon SageMaker access to resources in your AWS account.
    # You can use the Amazon SageMaker Python SDK to get the role from a notebook environment. 
    role = sagemaker.get_execution_role()

Now import the Python libraries.

.. code:: ipython3

    import boto3
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

Code
----

To use Amazon SageMaker to run Docker containers, we need to provide an
python script for the container to run. In this example, mxnet_gcn.py
provides all the code we need for training a SageMaker model.

.. code:: ipython3

    !cat train.py

After you specify and tested the training script to ensure it works,
start the tuning job. Testing can be done in either local mode or using
SageMaker training.

Tune
----

Similar to training a single training job in SageMaker, define your
training estimator passing in the code scripts, IAM role, (per job)
hardware configuration, and any hyperparameters we’re not tuning.

We assume you have already got your own GCMC Docker image in your ECR
following the steps in mxnet_gcmc.ipynb.

.. code:: ipython3

    from sagemaker.mxnet.estimator import MXNet
    
    # Set target dgl-docker name
    docker_name='sagemaker-dgl-gcmc'
    
    CODE_PATH = '../dgl_gcmc'
    CODE_ENTRY = 'train.py'
    #code_location = sess.upload_data(CODE_PATH, bucket=bucket, key_prefix=custom_code_upload_location)
    
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, docker_name)
    print(image)
    
    params = {}
    params['data_name'] = 'ml-1m'
    # set output to Amazon SageMaker ML output
    params['save_dir'] = '/opt/ml/model/'
    estimator = MXNet(entry_point=CODE_ENTRY,
                      source_dir=CODE_PATH,
                      role=role,
                      train_instance_count=1,
                      train_instance_type='ml.p3.2xlarge',
                      image_name=image,
                      hyperparameters=params,
                      sagemaker_session=sess)

After you define your estimator, specify the hyperparameters you want to
tune and their possible values. You have three different types of
hyperparameters. \* Categorical parameters need to take one value from a
discrete set. You define this by passing the list of possible values to
CategoricalParameter(list) \* Continuous parameters can take any real
number value between the minimum and maximum value, defined by
ContinuousParameter(min, max) \* Integer parameters can take any integer
value between the minimum and maximum value, defined by
IntegerParameter(min, max)

If possible, it’s almost always best to specify a value as the least
restrictive type. For example, tuning thresh as a continuous value
between 0.01 and 0.2 is likely to yield a better result than tuning as a
categorical parameter with possible values of 0.01, 0.1, 0.15, or 0.2.

.. code:: ipython3

    hyperparameter_ranges = {'gcn_agg_accum': CategoricalParameter(['sum', 'stack']),
                             'train_lr': ContinuousParameter(0.001, 0.1),
                             'gen_r_num_basis_func': IntegerParameter(1, 3)}

Next, specify the objective metric to tune and its definition. This
includes the regular expression (Regex) needed to extract that metric
from the CloudWatch logs of our training job.

.. code:: ipython3

    objective_metric_name = 'Validation-accuracy'
    metric_definitions = [{'Name': 'Validation-accuracy',
                           'Regex': 'Best Iter Idx=[0-9\\.]+, Best Valid RMSE=[0-9\\.]+, Best Test RMSE=([0-9\\.]+)'}]


Now, create a HyperparameterTuner object, which you pass:

-  The training estimator created above
-  Your hyperparameter ranges
-  Objective metric name and definition
-  Number of training jobs to run in total and how many training jobs
   should be run simultaneously. More parallel jobs will finish tuning
   sooner, but may sacrifice accuracy. We recommend you set the parallel
   jobs value to less than 10% of the total number of training jobs
   (we’ll set it higher just for this example to keep it short).
-  Whether you should maximize or minimize the objective metric. You
   haven’t specified here since it defaults to ‘Maximize’, which is what
   you want for validation accuracy.

You can also add a task_tag with value ‘DGL’ to help tracking the
hyperparameter tuning task.

.. code:: ipython3

    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                objective_type='Minimize',
                                tags=task_tags,
                                max_jobs=10,
                                max_parallel_jobs=2)

And finally, you can start the tuning job by calling .fit().

.. code:: ipython3

    tuner.fit()

Let’s just run a quick check of the hyperparameter tuning jobs status to
make sure it started successfully and is InProgress.

.. code:: ipython3

    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

Wrap-up
-------

Now that you started the hyperparameter tuning job, it runs in the
background and you can close this notebook. Once finished, you can go to
console to analyze the result.

For more detail on Amazon SageMaker’s Hyperparameter Tuning, please
refer to the AWS documentation.
