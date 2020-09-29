Hyperparameter tuning with Amazon SageMaker and Deep Graph Library with PyTorch backend
=======================================================================================

**Creating a Hyperparameter Tuning Job for an Deep Graph Library (DGL)
Network**

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
model to train the [Cora dataset] using DGL with PyTorch backend. It
leverages SageMaker’s hyperparameter tuning to kick off multiple
training jobs with different hyperparameter combinations, to find the
set with best model performance. This is an important step in the
machine learning process as hyperparameter settings can have a large
impact on model accuracy. In this example, you use the `Amazon SageMaker
Python SDK <https://github.com/aws/sagemaker-python-sdk>`__ to create a
hyperparameter tuning job for an Amazon SageMaker estimator.

Setup
-----

This notebook was created and tested on an ml.p3.2xlarge notebook
instance.

Prerequisites \* You can successfully run the pytorch_gcn example (see
pytorch_gcn.ipynb). \* An S3 bucket and prefix that you want to use
exists for training and model data. This should be within the same
Region as the notebook instance, training, and hosting. \* You have the
IAM role ARN used to give training and hosting access to your data. See
the documentation for more details on creating these. If a role not
associated with the current notebook instance, or more than one role is
required for training and/or hosting, replace
sagemaker.get_execution_role() with a the appropriate full IAM role arn
string(s).

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
    # You can use the Amazon SageMaker Python SDK to get the role from the notebook environment. 
    role = sagemaker.get_execution_role()

Now import the Python libraries.

.. code:: ipython3

    import boto3
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

Code
----

To use Amazon SageMaker to run Docker containers, provide an Python
script for the container to run. In this example, pytorch_gcn.py
provides all the code for training an Amazon SageMaker model.

.. code:: ipython3

    !cat pytorch_gcn.py

After you specify and test the training script to ensure it works, you
can start the tuning job. Testing can be done in either local mode or
using Amazon SageMaker training.

Tune
----

Similar to training a single training job in Amazon SageMaker, define
the training estimator passing in the code scripts, IAM role, (per job)
hardware configuration, and any hyperparameters you are not tuning.

.. code:: ipython3

    from sagemaker.pytorch import PyTorch
    
    CODE_PATH = 'pytorch_gcn.py'
    
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    
    params = {}
    params['dataset'] = 'cora'
    estimator = PyTorch(entry_point=CODE_PATH,
                        role=role,
                        train_instance_count=1,
                        train_instance_type='ml.p3.2xlarge',
                        framework_version="1.3.1",
                        py_version='py3',
                        debugger_hook_config=False,
                        hyperparameters=params,
                        sagemaker_session=sess)

After you define the estimator, specify the hyperparameters you want to
tune and their possible values. You have three different types of
hyperparameters. \* Categorical parameters need to take one value from a
discrete set. Define this by passing the list of possible values to
CategoricalParameter(list) \* Continuous parameters can take any real
number value between the minimum and maximum value, defined by
ContinuousParameter(min, max) \* Integer parameters can take any integer
value between the minimum and maximum value, defined by
IntegerParameter(min, max)

Note, if possible, it’s almost always best to specify a value as the
least restrictive type. For example, tuning threshold as a continuous
value between 0.01 and 0.2 is likely to yield a better result than
tuning as a categorical parameter with possible values of 0.01, 0.1,
0.15, or 0.2.

.. code:: ipython3

    hyperparameter_ranges = {'lr': ContinuousParameter(0.001, 0.01),
                             'n-epochs': IntegerParameter(100, 200)}

Next, specify the objective metric that you want to tune and its
definition. This includes the regular expression (regex) needed to
extract that metric from the Amazon CloudWatch logs of the training job

.. code:: ipython3

    objective_metric_name = 'Validation-accuracy'
    metric_definitions = [{'Name': 'Validation-accuracy',
                           'Regex': 'Test Accuracy ([0-9\\.]+)%'}]

Now, create a HyperparameterTuner object, which you pass:

-  The training estimator you created above
-  The hyperparameter ranges
-  Objective metric name and definition
-  Number of training jobs to run in total and how many training jobs
   should be run simultaneously. More parallel jobs will finish tuning
   sooner, but may sacrifice accuracy. We recommend you set the parallel
   jobs value to less than 10% of the total number of training jobs
   (we’ll set it higher just for this example to keep it short).
-  Whether you should maximize or minimize the objective metric. You
   haven’t specified here since it defaults to ‘Maximize’, which is what
   you want for validation accuracy

You can also add a task_tag with value ‘DGL’ to help tracking the
hyperparameter tuning task.

.. code:: ipython3

    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                tags=task_tags,
                                max_jobs=6,
                                max_parallel_jobs=2)

And finally, start the tuning job by calling .fit().

.. code:: ipython3

    tuner.fit()

Let’s just run a quick check of the hyperparameter tuning jobs status to
make sure it started successfully and is InProgress.

.. code:: ipython3

    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

Wrap-up
-------

After you start the hyperparameter tuning job, it will run in the
background. You can close this notebook. After it finishes, you can go
to console to analyze the result.

For more information about Amazon SageMaker’s Hyperparameter tuning, see
the AWS documentation.
