Hyperparameter tuning with Amazon SageMaker and Deep Graph Library with MXNet backend
=====================================================================================

**Creating a Hyperparameter tuning job for a DGL network**

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Tune <#Train>`__
4. `Wrap-up <#Wrap-up>`__

Background
----------

This example notebook shows how to generate knowledge graph embedding
using the DMLC DGL API and FB15k dataset. It uses the Amazon SageMaker
hyperparameter tuning to start multiple training jobs with different
hyperparameter combinations. This helps you find the set with best model
performance. This is an important step in the machine learning process
as hyperparameter settings can have a large effect on model accuracy. In
this example, you use the `Amazon SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__ to create a
hyperparameter tuning job for an Amazon SageMaker estimator.

Setup
-----

This notebook was created and tested on an ml.p3.2xlarge notebook
instance.

Prerequisites \* You can successfully run the kge_mxnet example (see
kge_mxnet.ipynb). \* You have an S3 bucket and prefix that you want to
use for training and model data. This should be within the same Region
as the notebook instance, training, and hosting. \* You have the IAM
role ARN used to give training and hosting access to your data. See the
documentation for more details on creating these. If a role not
associated with the current notebook instance, or more than one role, is
required for training or hosting, replace sagemaker.get_execution_role()
with the appropriate full IAM role ARN strings.

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

Now we’ll import the Python libraries we’ll need.

.. code:: ipython3

    import boto3
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

Tune
----

Similar to training a single training job in Amazon SageMaker, you
define the training estimator passing in the code scripts, IAM role,
(per job) hardware configuration, and any hyperparameters you’re not
tuning.

.. code:: ipython3

    from sagemaker.mxnet.estimator import MXNet
    
    ENTRY_POINT = 'train.py'
    CODE_PATH = './'
    
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    
    params = {}
    params['dataset'] = 'FB15k'
    params['model'] = 'DistMult'
    params['batch_size'] = 1024
    params['neg_sample_size'] = 256
    params['hidden_dim'] = 2000
    params['max_step'] = 100000
    params['batch_size_eval'] = 16
    params['valid'] = True
    params['test'] = True
    params['neg_adversarial_sampling'] = True
    
    estimator = MXNet(entry_point=ENTRY_POINT,
                      source_dir=CODE_PATH,
                      role=role,
                      train_instance_count=1,
                      train_instance_type='ml.p3.2xlarge',
                      framework_version="1.6.0",
                      py_version='py3',
                      hyperparameters=params,
                      sagemaker_session=sess)

After you define your estimator, specify the hyperparameters you want to
tune and their possible values. You have three different types of
hyperparameters. \* Categorical parameters need to take one value from a
discrete set. Define this by passing the list of possible values to
CategoricalParameter(list) \* Continuous parameters can take any real
number value between the minimum and maximum value, defined by
ContinuousParameter(min, max) \* Integer parameters can take any integer
value between the minimum and maximum value, defined by
IntegerParameter(min, max)

If possible, it’s almost always best to specify a value as the least
restrictive type. For example, tuning threshold as a continuous value
between 0.01 and 0.2 is likely to yield a better result than tuning as a
categorical parameter with possible values of 0.01, 0.1, 0.15, or 0.2.

.. code:: ipython3

    hyperparameter_ranges = {'lr': ContinuousParameter(0.01, 0.1),
                             'gamma': ContinuousParameter(400, 600)}

Next, specify the objective metric that you want to tune and its
definition. This includes the regular expression needed to extract that
metric from the Amazon CloudWatch logs of the training job.

You can capture evalution results such as MR, MRR and Hit10.

.. code:: ipython3

    metric = []
    mr_metric = {'Name': 'final_MR', 'Regex':"Test average MR at \[\S*\]: (\S*)"}
    mrr_metric = {'Name': 'final_MRR', 'Regex':"Test average MRR at \[\S*\]: (\S*)"}
    hit10_metric = {'Name': 'final_Hit10', 'Regex':"Test average HITS@10 at \[\S*\]: (\S*)"}
    metric.append(mr_metric)
    metric.append(mrr_metric)
    metric.append(hit10_metric)

Now, create a HyperparameterTuner object, which you pass.

-  The training estimator you created above
-  The hyperparameter ranges
-  Objective metric name and definition
-  Number of training jobs to run in-total and how many training jobs
   should be run simultaneously. More parallel jobs will finish tuning
   sooner, but may sacrifice accuracy. We recommend that you set the
   parallel jobs value to less than 10 percent of the total number of
   training jobs It’s set it higher in this example to keep it short.
-  Whether you should maximize or minimize the objective metric. You
   choose ‘Minimize’ in this example, which is what you want for the MR
   result.

You can also add a task_tag with value ‘DGL’ to help tracking the
hyperparameter tuning task.

.. code:: ipython3

    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    tuner = HyperparameterTuner(estimator,
                                objective_metric_name='final_MR',
                                objective_type='Minimize',
                                hyperparameter_ranges=hyperparameter_ranges,
                                metric_definitions=metric,
                                tags=task_tags,
                                max_jobs=6,
                                max_parallel_jobs=2)

And finally, you can start the tuning job by calling .fit().

.. code:: ipython3

    tuner.fit()

Run a quick check of the hyperparameter tuning jobs status to make sure
it started successfully and is InProgress.

.. code:: ipython3

    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

Wrap-up
-------

Now that we’ve started the hyperparameter tuning job, it will run in the
background. You can close this notebook. When it’s finished, you can go
to console to analyze the result.

For more information about Amazon SageMaker’s Hyperparameter Tuning, see
the AWS documentation.
