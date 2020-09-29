Amazon SageMaker - Tensorflow 2.1
=================================

`Amazon SageMaker <https://aws.amazon.com/sagemaker/>`__ is managed
platform to build, train and host maching learning models. `Amazon
SageMaker Debugger <https://github.com/awslabs/sagemaker-debugger>`__ is
a new feature which offers the capability to debug machine learning
models during training by identifying and detecting problems with the
models in real-time.

Experimental support for TF 2.x was introduced in v0.7.1 of the
Debugger, which enables the Debugger to save losses, metrics, weights,
and biases. Full description of support is available at `Amazon
SageMaker Debugger with
TensorFlow <https://github.com/awslabs/sagemaker-debugger/tree/master/docs/tensorflow.md>`__

In this notebook, we’ll show you how to use the SageMaker Debugger
without making any changes to the training script and using a built-in
rule to monitor your training job using a tf.keras (TF 2.1.0) ResNet
example.

Training TensorFlow Keras models with Amazon SageMaker Debugger
---------------------------------------------------------------

Amazon SageMaker TensorFlow as a framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a TensorFlow Keras model in this notebook with Amazon Sagemaker
Debugger enabled and monitor the training jobs with rules. This is done
using Amazon SageMaker `TensorFlow
2.1.0 <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html>`__
Container as a framework.

Setup
-----

Follow this one time setup to get your notebook up and running to use
Amazon SageMaker Debugger. This is only needed because we plan to
perform interactive analysis using this library in the notebook.

.. code:: ipython3

    ! pip install smdebug

.. code:: ipython3

    import boto3
    import os
    import sagemaker
    from sagemaker.tensorflow import TensorFlow

Import the libraries needed for the demo of Amazon SageMaker Debugger.

.. code:: ipython3

    from sagemaker.debugger import Rule, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig, rule_configs

Now define the entry point for the training script and hyperparameters
for the training script

.. code:: ipython3

    # define the entrypoint script
    entrypoint_script='src/keras_fit_zero_code_change.py'
    
    hyperparameters = {
        "batch_size": 64
    }

Setting up the Estimator
~~~~~~~~~~~~~~~~~~~~~~~~

Now it’s time to setup our TensorFlow estimator. We’ve added new
parameters to the estimator to enable your training job for debugging
through Amazon SageMaker Debugger. These new parameters are explained
below.

-  **debugger_hook_config**: This new parameter accepts a local path
   where you wish your tensors to be written to and also accepts the S3
   URI where you wish your tensors to be uploaded to. SageMaker will
   take care of uploading these tensors transparently during execution.
-  **rules**: This new parameter will accept a list of rules you wish to
   evaluate against the tensors output by this training job. For rules,
   Amazon SageMaker Debugger supports two types:
-  **SageMaker Rules**: These are rules specially curated by the data
   science and engineering teams in Amazon SageMaker which you can opt
   to evaluate against your training job.
-  **Custom Rules**: You can optionally choose to write your own rule as
   a Python source file and have it evaluated against your training job.
   To provide Amazon SageMaker Debugger to evaluate this rule, you would
   have to provide the S3 location of the rule source and the evaluator
   image.

.. code:: ipython3

    hook_config = DebuggerHookConfig(
        hook_parameters={
            "save_interval": "100"
        },
        collection_configs=[
            CollectionConfig("weights"),
            CollectionConfig(
                name="biases",
                parameters={
                    "save_interval": "10",
                    "end_step": "500"
                }
            ),
        ]
    )

Using Amazon SageMaker Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example we’ll demonstrate how to use SageMaker rules to be
evaluated against your training. You can find the list of SageMaker
rules and the configurations best suited for using them
`here <https://github.com/awslabs/sagemaker-debugger-rulesconfig>`__.

The rules we’ll use are **VanishingGradient** and **LossNotDecreasing**.
As the names suggest, the rules will attempt to evaluate if there are
vanishing gradients in the tensors captured by the debugging hook during
training and also if the loss is not decreasing.

.. code:: ipython3

    rules = [
        Rule.sagemaker(rule_configs.vanishing_gradient()), 
        Rule.sagemaker(rule_configs.loss_not_decreasing())
    ]

Let us now create the estimator and call ``fit()`` on our estimator to
start the training job and rule evaluation job in parallel.

.. code:: ipython3

    estimator = TensorFlow(
        role=sagemaker.get_execution_role(),
        base_job_name='smdebug-demo-tf2-keras',
        train_instance_count=1,
        train_instance_type='ml.p2.xlarge',
        entry_point=entrypoint_script,
        framework_version='2.1.0',
        py_version='py3',
        train_max_run=3600,
        script_mode=True,
        ## New parameter
        rules = rules,
        debugger_hook_config=hook_config
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

    estimator.latest_training_job.rule_job_summary()    

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




.. parsed-literal::

    {'VanishingGradient': 'https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logStream:group=/aws/sagemaker/ProcessingJobs;prefix=smdebug-demo-tf2-keras-202-VanishingGradient-99e0011e;streamFilter=typeLogStreamPrefix',
     'LossNotDecreasing': 'https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logStream:group=/aws/sagemaker/ProcessingJobs;prefix=smdebug-demo-tf2-keras-202-LossNotDecreasing-45295edc;streamFilter=typeLogStreamPrefix'}



Data Analysis - Interactive Exploration
---------------------------------------

Now that we have trained a job, and looked at automated analysis through
rules, let us also look at another aspect of Amazon SageMaker Debugger.
It allows us to perform interactive exploration of the tensors saved in
real time or after the job. Here we focus on after-the-fact analysis of
the above job. We import the ``smdebug`` library, which defines a
concept of Trial that represents a single training run. Note how we
fetch the path to debugger artifacts for the above job.

.. code:: ipython3

    from smdebug.trials import create_trial
    trial = create_trial(estimator.latest_job_debugger_artifacts_path())

We can list all the tensors that were recorded to know what we want to
plot. Each one of these names is the name of a tensor, which is
auto-assigned by TensorFlow. In some frameworks where such names are not
available, we try to create a name based on the layer’s name and whether
it is weight, bias, gradient, input or output.

Note: As part of this experimental support fot TF 2.x, gradients,
inputs, outputs are not saved by Sagemaker Debugger

.. code:: ipython3

    len(trial.tensor_names())


.. parsed-literal::

    [2020-05-04 04:31:16.548 ip-172-16-189-249:11978 INFO trial.py:198] Training has ended, will refresh one final time in 1 sec.
    [2020-05-04 04:31:17.569 ip-172-16-189-249:11978 INFO trial.py:210] Loaded all steps




.. parsed-literal::

    326



.. code:: ipython3

    len(trial.tensor_names(collection="weights"))




.. parsed-literal::

    266



.. code:: ipython3

    trial.tensor_names(collection="metrics")




.. parsed-literal::

    ['accuracy', 'batch', 'size']



.. code:: ipython3

    trial.tensor_names(collection="losses")




.. parsed-literal::

    ['loss', 'val_loss']



