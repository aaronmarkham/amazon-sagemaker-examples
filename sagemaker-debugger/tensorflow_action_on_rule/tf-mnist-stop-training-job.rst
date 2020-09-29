Amazon SageMaker Debugger - Reacting to Cloudwatch Events from Rules
====================================================================

`Amazon SageMaker <https://aws.amazon.com/sagemaker/>`__ is managed
platform to build, train and host maching learning models. Amazon
SageMaker Debugger is a new feature which offers the capability to debug
machine learning models during training by identifying and detecting
problems with the models in near real time.

In this notebook, we’ll show you how you can react off rule triggers and
take some action, e.g. stop the training job through CloudWatch Events.

How does Amazon SageMaker Debugger work?
----------------------------------------

Amazon SageMaker Debugger lets you go beyond just looking at scalars
like losses and accuracies during training and gives you full visibility
into all tensors ‘flowing through the graph’ during training.
Furthermore, it helps you monitor your training in near real time using
rules and provides you alerts, once it has detected inconsistency in
training flow.

Concepts
~~~~~~~~

-  **Tensors**: These represent the state of the training network at
   intermediate points during its execution
-  **Debug Hook**: Hook is the construct with which Amazon SageMaker
   Debugger looks into the training process and captures the tensors
   requested at the desired step intervals
-  **Rule**: A logical construct, implemented as Python code, which
   helps analyze the tensors captured by the hook and report anomalies,
   if at all

With these concepts in mind, let’s understand the overall flow of things
that Amazon SageMaker Debugger uses to orchestrate debugging.

Saving tensors during training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tensors captured by the debug hook are stored in the S3 location
specified by you. There are two ways you can configure Amazon SageMaker
Debugger to save tensors:

With no changes to your training script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use one of the SageMaker provided `Deep Learning
Containers <https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html>`__
for 1.15, then you don’t need to make any changes to your training
script for the tensors to be stored. SageMaker Debugger will use the
configuration you provide through the SageMaker SDK’s Tensorflow
``Estimator`` when creating your job to save the tensors in the fashion
you specify. You can review the script we are going to use at
`src/mnist_zerocodechange.py <src/mnist_zerocodechange.py>`__. You will
note that this is an untouched TensorFlow script which uses the
Estimator interface. Please note that SageMaker Debugger only supports
``tf.keras``, ``tf.Estimator`` and ``tf.MonitoredSession`` interfaces.
Full description of support is available at `SageMaker Debugger with
TensorFlow <https://github.com/awslabs/sagemaker-debugger/tree/master/docs/tensorflow.md>`__

Orchestrating your script to store tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For other containers, you need to make couple of lines of changes to
your training script. SageMaker Debugger exposes a library ``smdebug``
which allows you to capture these tensors and save them for analysis.
It’s highly customizable and allows to save the specific tensors you
want at different frequencies and possibly with other configurations.
Refer
`DeveloperGuide <https://github.com/awslabs/sagemaker-debugger/tree/master/docs>`__
for details on how to use SageMaker Debugger library with your choice of
framework in your training script. Here we have an example script
orchestrated at `src/mnist_byoc <src/mnist_byoc.py>`__. You also need to
ensure that your container has the ``smdebug`` library installed.

Analysis of tensors
~~~~~~~~~~~~~~~~~~~

Once the tensors are saved, Amazon SageMaker Debugger can be configured
to run debugging **Rules** on them. At a very broad level, a rule is
Python code used to detect certain conditions during training. Some of
the conditions that a data scientist training an algorithm may care
about are monitoring for gradients getting too large or too small,
detecting overfitting, and so on. Sagemaker Debugger comes pre-packaged
with certain built-in rules. Users can write their own rules using the
Sagemaker Debugger APIs. You can also analyze raw tensor data outside of
the Rules construct in say, a Sagemaker notebook, using Amazon Sagemaker
Debugger’s full set of APIs. Please refer `Analysis Developer
Guide <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md>`__
for more on these APIs.

Cloudwatch Events for Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rule status changes in a training job trigger CloudWatch Events. These
events can be acted upon by configuring a CloudWatch Rule (different
from Amazon SageMaker Debugger Rule) to trigger each time a Debugger
Rule changes status. In this notebook we’ll go through how you can
create a CloudWatch Rule to direct Training Job State change events to a
lambda function that stops the training job in case a rule triggers and
has status ``"IssuesFound"``

Lambda Function
^^^^^^^^^^^^^^^

-  In your AWS console, go to Lambda Management Console,
-  Create a new function by hitting Create Function,
-  Choose the language as Python 3.7 and put in the following sample
   code for stopping the training job if one of the Rule statuses is
   ``"IssuesFound"``:

.. code:: python

   import json
   import boto3
   import logging

   def lambda_handler(event, context):
       training_job_name = event.get("detail").get("TrainingJobName")
       eval_statuses = event.get("detail").get("DebugRuleEvaluationStatuses", None)

       if eval_statuses is None or len(eval_statuses) == 0:
           logging.info("Couldn't find any debug rule statuses, skipping...")
           return {
               'statusCode': 200,
               'body': json.dumps('Nothing to do')
           }

       client = boto3.client('sagemaker')

       for status in eval_statuses:
           if status.get("RuleEvaluationStatus") == "IssuesFound":
               logging.info(
                   'Evaluation of rule configuration {} resulted in "IssuesFound". '
                   'Attempting to stop training job {}'.format(
                       status.get("RuleConfigurationName"), training_job_name
                   )
               )
               try:
                   client.stop_training_job(
                       TrainingJobName=training_job_name
                   )
               except Exception as e:
                   logging.error(
                       "Encountered error while trying to "
                       "stop training job {}: {}".format(
                           training_job_name, str(e)
                       )
                   )
                   raise e
       return None

-  Create a new execution role for the Lambda, and
-  In your IAM console, search for the role and attach
   “AmazonSageMakerFullAccess” policy to the role. This is needed for
   the code in your Lambda function to stop the training job.

Create a CloudWatch Rule
^^^^^^^^^^^^^^^^^^^^^^^^

-  In your AWS Console, go to CloudWatch and select Rule from the left
   column,
-  Hit Create Rule. The console will redirect you to the Rule creation
   page,
-  For the Service Name, select “SageMaker”.
-  For the Event Type, select “SageMaker Training Job State Change”.
-  In the Targets select the Lambda function you created above, and
-  For this example notebook, we’ll leave everything as is.

.. code:: ipython3

    import boto3
    import os
    import sagemaker
    from sagemaker.tensorflow import TensorFlow

.. code:: ipython3

    from sagemaker.debugger import Rule, rule_configs

.. code:: ipython3

    # define the entrypoint script
    entrypoint_script='src/mnist_zerocodechange.py'
    
    # these hyperparameters ensure that vanishing gradient will trigger for our tensorflow mnist script
    hyperparameters = {
        "num_epochs": "10",
        "lr": "10.00"
    }

.. code:: ipython3

    rules=[
        Rule.sagemaker(rule_configs.vanishing_gradient()), 
        Rule.sagemaker(rule_configs.loss_not_decreasing())
    ]
    
    estimator = TensorFlow(
        role=sagemaker.get_execution_role(),
        base_job_name='smdebugger-demo-mnist-tensorflow',
        train_instance_count=1,
        train_instance_type='ml.m4.xlarge',
        entry_point=entrypoint_script,
        framework_version='1.15',
        train_volume_size=400,
        py_version='py3',
        train_max_run=3600,
        script_mode=True,
        hyperparameters=hyperparameters,
        ## New parameter
        rules = rules
    )

.. code:: ipython3

    # After calling fit, SageMaker will spin off 1 training job and 1 rule job for you
    # The rule evaluation status(es) will be visible in the training logs
    # at regular intervals
    # wait=False makes this a fire and forget function. To stream the logs in the notebook leave this out
    
    estimator.fit(wait=False)

Monitoring
----------

SageMaker kicked off rule evaluation jobs, one for each of the SageMaker
rules - ``VanishingGradient`` and ``LossNotDecreasing`` as specified in
the estimator. Given that we’ve tweaked the hyperparameters of our
training script such that ``VanishingGradient`` is bound to fire, we
should expect to see the ``TrainingJobStatus`` as ``Stopped`` once the
``RuleEvaluationStatus`` for ``VanishingGradient`` changes to
``IssuesFound``

.. code:: ipython3

    # rule job summary gives you the summary of the rule evaluations. You might have to run it over 
    # a few times before you start to see all values populated/changing
    estimator.latest_training_job.rule_job_summary()




.. parsed-literal::

    [{'RuleConfigurationName': 'VanishingGradient',
      'RuleEvaluationJobArn': 'arn:aws:sagemaker:us-east-2:072677473360:processing-job/smdebugger-demo-mnist-tens-vanishinggradient-e23301a8',
      'RuleEvaluationStatus': 'IssuesFound',
      'StatusDetails': 'RuleEvaluationConditionMet: Evaluation of the rule VanishingGradient at step 500 resulted in the condition being met\n',
      'LastModifiedTime': datetime.datetime(2019, 12, 1, 7, 20, 55, 495000, tzinfo=tzlocal())},
     {'RuleConfigurationName': 'LossNotDecreasing',
      'RuleEvaluationJobArn': 'arn:aws:sagemaker:us-east-2:072677473360:processing-job/smdebugger-demo-mnist-tens-lossnotdecreasing-27ee2da1',
      'RuleEvaluationStatus': 'InProgress',
      'LastModifiedTime': datetime.datetime(2019, 12, 1, 7, 20, 55, 495000, tzinfo=tzlocal())}]



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




.. parsed-literal::

    {'VanishingGradient': 'https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2#logStream:group=/aws/sagemaker/ProcessingJobs;prefix=smdebugger-demo-mnist-tens-VanishingGradient-e23301a8;streamFilter=typeLogStreamPrefix',
     'LossNotDecreasing': 'https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2#logStream:group=/aws/sagemaker/ProcessingJobs;prefix=smdebugger-demo-mnist-tens-LossNotDecreasing-27ee2da1;streamFilter=typeLogStreamPrefix'}



After running the last two cells over and until ``VanishingGradient``
reports ``IssuesFound``, we’ll attempt to describe the
``TrainingJobStatus`` for our training job.

.. code:: ipython3

    estimator.latest_training_job.describe()["TrainingJobStatus"]




.. parsed-literal::

    'Stopped'



Result
------

This notebook attempted to show a very simple setup of how you can use
CloudWatch events for your training job to take action on rule
evaluation status changes. Learn more about Amazon SageMaker Debugger in
the `GitHub
Documentation <https://github.com/awslabs/sagemaker-debugger>`__.
