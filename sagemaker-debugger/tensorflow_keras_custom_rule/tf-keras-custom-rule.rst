Amazon SageMaker - Debugging with custom rules
==============================================

`Amazon SageMaker <https://aws.amazon.com/sagemaker/>`__ is managed
platform to build, train and host maching learning models. Amazon
SageMaker Debugger is a new feature which offers the capability to debug
machine learning models during training by identifying and detecting
problems with the models in near real-time.

In this notebook, we’ll show you how to use a custom rule to monitor
your training job. All through a tf.keras ResNet example.

How does Amazon SageMaker Debugger work?
----------------------------------------

Amazon SageMaker Debugger lets you go beyond just looking at scalars
like losses and accuracies during training and gives you full visibility
into all tensors ‘flowing through the graph’ during training.
Furthermore, it helps you monitor your training in near real-time using
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

If you use one of the Amazon SageMaker provided `Deep Learning
Containers <https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html>`__
for 1.15, then you don’t need to make any changes to your training
script for the tensors to be stored. Amazon SageMaker Debugger will use
the configuration you provide through the Amazon SageMaker SDK’s
Tensorflow ``Estimator`` when creating your job to save the tensors in
the fashion you specify. You can review the script we are going to use
at
`src/tf_keras_resnet_zerocodechange.py <src/tf_keras_resnet_zerocodechange.py>`__.
You will note that this is an untouched TensorFlow Keras script which
uses the ``tf.keras`` interface. Please note that Amazon SageMaker
Debugger only supports ``tf.keras``, ``tf.estimator`` and
``tf.MonitoredSession`` interfaces for the zero script change
experience. Full description of support is available at `Amazon
SageMaker Debugger with
TensorFlow <https://github.com/awslabs/sagemaker-debugger/tree/master/docs/tensorflow.md>`__

Orchestrating your script to store tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For other containers, you need to make couple of lines of changes to
your training script. Amazon SageMaker Debugger exposes a library
``smdebug`` which allows you to capture these tensors and save them for
analysis. It’s highly customizable and allows to save the specific
tensors you want at different frequencies and possibly with other
configurations. Refer
`DeveloperGuide <https://github.com/awslabs/sagemaker-debugger/tree/master/docs>`__
for details on how to use Amazon SageMaker Debugger library with your
choice of framework in your training script. Here we have an example
script orchestrated at
`src/tf_keras_resnet_byoc.py <src/tf_keras_resnet_byoc.py>`__. In
addition to this, you will need to ensure that your container has the
``smdebug`` library installed in this case, and specify your container
image URI when creating the SageMaker Estimator below. Please refer
`SageMaker
Documentation <https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html>`__
on how to do that.

Analysis of tensors
~~~~~~~~~~~~~~~~~~~

Amazon SageMaker Debugger can be configured to run debugging **Rules**
on the tensors saved from the training job. At a very broad level, a
rule is Python code used to detect certain conditions during training.
Some of the conditions that a data scientist training an algorithm may
care about are monitoring for gradients getting too large or too small,
detecting overfitting, and so on. Amazon SageMaker Debugger comes
pre-packaged with certain built-in rules. Users can write their own
rules using the APIs provided by Amazon SageMaker Debugger through the
``smdebug`` library. You can also analyze raw tensor data outside of the
Rules construct in say, a SageMaker notebook, using these APIs. Please
refer `Analysis Developer
Guide <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md>`__
for more on these APIs.

Training TensorFlow Keras models with Amazon SageMaker Debugger
---------------------------------------------------------------

Amazon SageMaker TensorFlow as a framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a TensorFlow Keras model in this notebook with Amazon Sagemaker
Debugger enabled and monitor the training jobs with rules. This is done
using Amazon SageMaker `TensorFlow
1.15.0 <https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html>`__
Container as a framework

.. code:: ipython3

    import boto3
    import os
    import sagemaker
    from sagemaker.tensorflow import TensorFlow

Import the libraries needed for the demo of Amazon SageMaker Debugger.

.. code:: ipython3

    from sagemaker.debugger import Rule, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig
    import smdebug_rulesconfig as rule_configs

Now define the entry point for the training script

.. code:: ipython3

    # define the entrypoint script
    entrypoint_script='src/tf_keras_resnet_zerocodechange.py'

Setting up the Estimator
~~~~~~~~~~~~~~~~~~~~~~~~

Now it’s time to setup our SageMaker TensorFlow Estimator. There are new
parameters with the estimator to enable your training job for debugging
through Amazon SageMaker Debugger. These new parameters are explained
below

-  **debugger_hook_config**: This new parameter accepts a local path
   where you wish your tensors to be written to and also accepts the S3
   URI where you wish your tensors to be uploaded to. It also accepts
   CollectionConfigurations which specify which tensors will be saved
   from the training job.
-  **rules**: This new parameter will accept a list of rules you wish to
   evaluate against the tensors output by this training job. For rules,

Amazon SageMaker Debugger supports two types of rules \* **Amazon
SageMaker Rules**: These are rules curated by the Amazon SageMaker team
and you can choose to evaluate them against your training job. \*
**Custom Rules**: You can optionally choose to write your own rule as a
Python source file and have it evaluated against your training job. To
provide SageMaker Debugger to evaluate this rule, you would have to
provide the S3 location of the rule source and the evaluator image.

Creating your own custom rule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us look at how you can create your custom rule briefly before
proceeding to use it with your training job. Please see the
`documentation <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md>`__
to learn more about structuring your rules and other related concepts.

**Summary of what the custom rule evaluates**
'''''''''''''''''''''''''''''''''''''''''''''

For demonstration purposes, below is a rule that tries to track whether
gradients are too large. The custom rule looks at the tensors in the
collection “gradients” saved during training and attempt to get the
absolute value of the gradients in each step of the training. If the
mean of the absolute values of gradients in any step is greater than a
specified threshold, mark the rule as ‘triggering’. Let us look at how
to structure the rule source.

Any custom rule logic you want to be evaluated should extend the
``Rule`` interface provided by Amazon SageMaker Debugger

.. code:: python

   from smdebug.rules.rule import Rule

   class CustomGradientRule(Rule):

Now implement the class methods for the rule. Doing this allows Amazon
SageMaker to understand the intent of the rule and evaluate it against
your training tensors.

Rule class constructor
''''''''''''''''''''''

In order for Amazon SageMaker to instantiate your rule, your rule class
constructor must conform to the following signature.

.. code:: python

       def __init__(self, base_trial, other_trials, <other parameters>)

Arguments
         

-  ``base_trial (Trial)``: This defines the primary
   `Trial <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#trial>`__
   that your rule is anchored to. This is an object of class type
   ``Trial``.

-  ``other_trials (list[Trial])``: *(Optional)* This defines a list of
   ‘other’ trials you want your rule to look at. This is useful in the
   scenarios when you’re comparing tensors from the base_trial to
   tensors from some other trials.

-  ``<other parameters>``: This is similar to ``**kwargs`` where you can
   pass in however many string parameters in your constructor signature.
   Note that SageMaker would only be able to support supplying string
   types for these values at runtime (see how, later).

Defining the rule logic to be invoked at each step:
'''''''''''''''''''''''''''''''''''''''''''''''''''

This defines the logic to invoked for each step. Essentially, this is
where you decide whether the rule should trigger or not. In this case,
you’re concerned about the gradients getting too large. So, get the
`tensor
reduction <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#reduction_value>`__
“mean” for each step and see if it’s value is larger than a threshold.

.. code:: python

       def invoke_at_step(self, step):
           for tname in self.base_trial.tensor_names(collection="gradients"):
               t = self.base_trial.tensor(tname)
               abs_mean = t.reduction_value(step, "mean", abs=True)
               if abs_mean > self.threshold:
                   return True
           return False

Using your custom rule with SageMaker Estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below we create the rule configuration using the ``Rule.custom`` method,
and then pass it to the SageMaker TensorFlow estimator to kick off the
job. Note that you need to pass the rule evaluator container image for
custom rules. Please refer AWS Documentation on SageMaker documentation
to find the image URI for your region. We will soon have this be
automatically taken care of by the SageMaker SDK. You can also provide
your own image, please refer to `this
repository <https://github.com/awslabs/sagemaker-debugger-rules-container>`__
for instructions on how to build such a container.

.. code:: ipython3

    custom_rule = Rule.custom(
        name='MyCustomRule', # used to identify the rule
        # rule evaluator container image
        image_uri='759209512951.dkr.ecr.us-west-2.amazonaws.com/sagemaker-debugger-rule-evaluator:latest', 
        instance_type='ml.t3.medium', # instance type to run the rule evaluation on
        source='rules/my_custom_rule.py', # path to the rule source file
        rule_to_invoke='CustomGradientRule', # name of the class to invoke in the rule source file
        volume_size_in_gb=30, # EBS volume size required to be attached to the rule evaluation instance
        collections_to_save=[CollectionConfig("gradients")], 
        # collections to be analyzed by the rule. since this is a first party collection we fetch it as above
        rule_parameters={
          "threshold": "20.0" # this will be used to intialize 'threshold' param in your constructor
        }
    )


Before you proceed and create our training job, let us take a closer
look at the parameters used to create the Rule configuration above:

-  ``name``: This is used to identify this particular rule among the
   suite of rules you specified to be evaluated.
-  ``image_uri``: This is the image of the container that has the logic
   of understanding your custom rule sources and evaluating them against
   the collections you save in the training job. You can get the list of
   open sourced SageMaker rule evaluator images
   `here <https://docs.aws.amazon.com/sagemaker/latest/dg/debuger-custom-rule-registry-ids.html>`__
-  ``instance_type``: The type of the instance you want to run the rule
   evaluation on
-  ``source``: This is the local path or the Amazon S3 URI of your rule
   source file.
-  ``rule_to_invoke``: This specifies the particular Rule class
   implementation in your source file which you want to be evaluated.
   SageMaker supports only 1 rule to be evaluated at a time in a rule
   job. Your source file can have multiple Rule class implementations,
   though.
-  ``collections_to_save``: This specifies which collections are
   necessary to be saved for this rule to run. Note that providing this
   collection does not necessarily mean the rule will actually use these
   collections. You might want to take such parameters for the rule
   through the next argument ``rule_parameters``.
-  ``rule_parameters``: This provides the runtime values of the
   parameter in your constructor. You can still choose to pass in other
   values which may be necessary for your rule to be evaluated. Any
   value in this map is available as an environment variable and can be
   accessed by your rule script using ``$<rule_parameter_key>``

You can read more about custom rule evaluation in Amazon SageMaker in
this
`documentation <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md>`__

Let us now create the estimator and call ``fit()`` on our estimator to
start the training job and rule evaluation job in parallel.

.. code:: ipython3

    estimator = TensorFlow(
        role=sagemaker.get_execution_role(),
        base_job_name='smdebug-custom-rule-demo-tf-keras',
        train_instance_count=1,
        train_instance_type='ml.p2.xlarge',
        entry_point=entrypoint_script,
        framework_version='1.15',
        py_version='py3',
        train_max_run=3600,
        script_mode=True,
        ## New parameter
        rules = [custom_rule]
    )
    
    # After calling fit, Amazon SageMaker starts one training job and one rule job for you.
    # The rule evaluation status is visible in the training logs
    # at regular intervals
    
    estimator.fit(wait=False)

Result
------

As a result of calling the ``fit(wait=False)``, two jobs were kicked off
in the background. Amazon SageMaker Debugger kicked off a rule
evaluation job for our custom gradient logic in parallel with the
training job. You can review the status of the above rule job as
follows.

.. code:: ipython3

    import time
    status = estimator.latest_training_job.rule_job_summary()
    while status[0]['RuleEvaluationStatus'] == 'InProgress':
        status = estimator.latest_training_job.rule_job_summary()
        print(status)
        time.sleep(10)
        

Once the rule job starts and you see the RuleEvaluationJobArn above, we
can see the logs for the rule job in Cloudwatch. To do that, we’ll use
this utlity function to get a link to the rule job logs.

.. code:: ipython3

    def _get_rule_job_name(training_job_name, rule_configuration_name, rule_job_arn):
            """Helper function to get the rule job name with correct casing"""
            return "{}-{}-{}".format(
                training_job_name[:26], rule_configuration_name[:26], rule_job_arn[-8:]
            )
        
    def _get_cw_url_for_rule_job(rule_job_name, region):
        return "https://{}.console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/ProcessingJobs;prefix={};streamFilter=typeLogStreamPrefix".format(region, region, rule_job_name)
    
    
    def get_rule_jobs_cw_urls(estimator):
        training_job = estimator.latest_training_job
        training_job_name = training_job.describe()["TrainingJobName"]
        rule_eval_statuses = training_job.describe()["DebugRuleEvaluationStatuses"]
        
        result={}
        for status in rule_eval_statuses:
            if status.get("RuleEvaluationJobArn", None) is not None:
                rule_job_name = _get_rule_job_name(training_job_name, status["RuleConfigurationName"], status["RuleEvaluationJobArn"])
                result[status["RuleConfigurationName"]] = _get_cw_url_for_rule_job(rule_job_name, boto3.Session().region_name)
        return result
    
    get_rule_jobs_cw_urls(estimator)
