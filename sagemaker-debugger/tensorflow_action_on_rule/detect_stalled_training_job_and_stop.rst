Detect stalled training and stop training job using debugger rule
=================================================================

In this notebook, we’ll show you how you can use StalledTrainingRule
rule which can take action like stopping your training job when it finds
that there has been no update in training job for certain threshold
duration.

How does StalledTrainingRule works?
-----------------------------------

Amazon Sagemaker debugger automatically captures tensors from training
job which use AWS DLC(tensorflow, pytorch, mxnet, xgboost)\ `refer doc
for supported
versions <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md#zero-script-change>`__.
StalledTrainingRule keeps watching on emission of tensors like loss. The
execution happens outside of training containers. It is evident that if
training job is running good and is not stalled it is expected to emit
loss and metrics tensors at frequent intervals. If Rule doesn’t find new
tensors being emitted from training job for threshold period of time, it
takes automatic action to issue StopTrainingJob.

With no changes to your training script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use one of the SageMaker provided `Deep Learning
Containers <https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html>`__.
`Refer doc for supported framework
versions <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md#zero-script-change>`__,
then you don’t need to make any changes to your training script for
activating this rule. Loss tensors will automatically be captured and
monitored by the rule.

You can also emit tensors periodically by using `save scalar api of
hook <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#common-hook-api>`__
.

Also look at example how to use save_scalar api
`here <https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf_keras_fit_non_eager.py#L42>`__

.. code:: ipython3

    ! pip install -q sagemaker

.. code:: ipython3

    import boto3
    import os
    import sagemaker
    from sagemaker.tensorflow import TensorFlow
    print(sagemaker.__version__)

.. code:: ipython3

    from sagemaker.debugger import Rule, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig
    import smdebug_rulesconfig as rule_configs

.. code:: ipython3

    # define the entrypoint script
    # Below script has 5 minutes sleep, we will create a stalledTrainingRule with 3 minutes of threshold.
    entrypoint_script='src/simple_stalled_training.py'
    
    # these hyperparameters ensure that vanishing gradient will trigger for our tensorflow mnist script
    hyperparameters = {
        "num_epochs": "10",
        "lr": "10.00"
    }

Create unique training job prefix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will create unique training job name prefix. this prefix would be
passed to StalledTrainingRule to identify which training job, rule
should take action on once the stalled training rule condition is met.
Note that, this prefix needs to be unique. If rule doesn’t find exactly
one job with provided prefix, it will fallback to safe mode and not take
action of stop training job. Rule will still emit a cloudwatch event if
the rule condition is met. To see details about cloud watch event, check
`here <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger/tensorflow_action_on_rule/tf-mnist-stop-training-job.ipynb>`__.

.. code:: ipython3

    import time
    print(int(time.time()))
    # Note that sagemaker appends date to your training job and truncates the provided name to 39 character. So, we will make 
    # sure that we use less than 39 character in below prefix. Appending time is to provide a unique id
    base_job_name_prefix= 'smdebug-stalled-demo-' + str(int(time.time()))
    base_job_name_prefix = base_job_name_prefix[:34]
    print(base_job_name_prefix)

.. code:: ipython3

    stalled_training_job_rule = Rule.sagemaker(
        base_config={
                        'DebugRuleConfiguration': {
                            'RuleConfigurationName': 'StalledTrainingRule', 
                            'RuleParameters': {'rule_to_invoke': 'StalledTrainingRule'}
                        }
                     },
        rule_parameters={
            'threshold': '120',
            'training_job_name_prefix': base_job_name_prefix,
            'stop_training_on_fire' : 'True'
        },    
    )

.. code:: ipython3

    estimator = TensorFlow(
        role=sagemaker.get_execution_role(),
        base_job_name=base_job_name_prefix,
        train_instance_count=1,
        train_instance_type='ml.m5.4xlarge',
        entry_point=entrypoint_script,
        #source_dir = 'src',
        framework_version='1.15.0',
        py_version='py3',
        train_max_run=3600,
        script_mode=True,
        ## New parameter
        rules = [stalled_training_job_rule]
    )


.. code:: ipython3

    # After calling fit, SageMaker will spin off 1 training job and 1 rule job for you
    # The rule evaluation status(es) will be visible in the training logs
    # at regular intervals
    # wait=False makes this a fire and forget function. To stream the logs in the notebook leave this out
    
    estimator.fit(wait=True)

Monitoring
----------

SageMaker kicked off rule evaluation job ``StalledTrainingRule`` as
specified in the estimator. Given that we’ve stalled our training script
for 10 minutes such that ``StalledTrainingRule`` is bound to fire and
take action StopTrainingJob, we should expect to see the
``TrainingJobStatus`` as ``Stopped`` once the ``RuleEvaluationStatus``
for ``StalledTrainingRule`` changes to ``IssuesFound``

.. code:: ipython3

    # rule job summary gives you the summary of the rule evaluations. You might have to run it over 
    # a few times before you start to see all values populated/changing
    estimator.latest_training_job.rule_job_summary()

.. code:: ipython3

    # This utility gives the link to monitor the CW event
    def _get_rule_job_name(training_job_name, rule_configuration_name, rule_job_arn):
            """Helper function to get the rule job name"""
            return "{}-{}-{}".format(
                training_job_name[:26], rule_configuration_name[:26], rule_job_arn[-8:]
            )
        
    def _get_cw_url_for_rule_job(rule_job_name, region):
        return "https://{}.console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/ProcessingJobs;prefix={};streamFilter=typeLogStreamPrefix".format(region, region, rule_job_name)
    
    
    def get_rule_jobs_cw_urls(estimator):
        region = boto3.Session().region_name
        training_job = estimator.latest_training_job
        training_job_name = training_job.describe()["TrainingJobName"]
        rule_eval_statuses = training_job.describe()["DebugRuleEvaluationStatuses"]
        
        result={}
        for status in rule_eval_statuses:
            if status.get("RuleEvaluationJobArn", None) is not None:
                rule_job_name = _get_rule_job_name(training_job_name, status["RuleConfigurationName"], status["RuleEvaluationJobArn"])
                result[status["RuleConfigurationName"]] = _get_cw_url_for_rule_job(rule_job_name, region)
        return result
    
    get_rule_jobs_cw_urls(estimator)

After running the last two cells over and until ``VanishingGradient``
reports ``IssuesFound``, we’ll attempt to describe the
``TrainingJobStatus`` for our training job.

.. code:: ipython3

    estimator.latest_training_job.describe()["TrainingJobStatus"]

Result
------

This notebook attempted to show a very simple setup of how you can use
CloudWatch events for your training job to take action on rule
evaluation status changes. Learn more about Amazon SageMaker Debugger in
the `GitHub
Documentation <https://github.com/awslabs/sagemaker-debugger>`__.
