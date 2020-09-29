Using Amazon SageMaker Debugger with your own PyTorch container
===============================================================

Amazon SageMaker is a managed platform to build, train and host machine
learning models. Amazon SageMaker Debugger is a new feature which offers
capability to debug machine learning and deep learning models during
training by identifying and detecting problems with the models in real
time.

Amazon SageMaker also gives you the option of bringing your own
algorithms packaged in a custom container, that can then be trained and
deployed in the Amazon SageMaker environment.

This notebook guides you through an example of using your own container
with PyTorch for training, along with the recently added feature, Amazon
SageMaker Debugger.

How does Amazon SageMaker Debugger work?
----------------------------------------

Amazon SageMaker Debugger lets you go beyond just looking at scalars
like losses and accuracies during training and gives you full visibility
into all tensors ‘flowing through the graph’ during training.
Furthermore, it helps you monitor your training in real time using rules
and CloudWatch events and react to issues like, for example, common
training issues such as vanishing gradients or poor weight
initialization.

Concepts
~~~~~~~~

-  **Tensor**: These are the artifacts that define the state of the
   training job at any particular instant in its lifecycle.
-  **Debug Hook**: Captures the tensors flowing through the training
   computational graph every N steps.
-  **Debugging Rule**: Logic to analyze the tensors captured by the hook
   and report anomalies.

With these concepts in mind, let’s understand the overall flow of things
which Amazon SageMaker Debugger uses to orchestrate debugging.

It operates in two steps - saving tensors and analysis.

Saving tensors
~~~~~~~~~~~~~~

Tensors that debug hook captures are stored in S3 location specified by
you. There are two ways you can configure Amazon SageMaker Debugger for
storage:

1. With no changes to your training script: If you use any of SageMaker
   provided `Deep Learning
   containers <https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html>`__
   then you don’t need to make any changes to your training script for
   tensors to be stored. Amazon SageMaker Debugger will use the
   configuration you provide in the framework ``Estimator`` to save
   tensors in the fashion you specify.
2. Orchestrating your script to store tensors: Amazon SageMaker Debugger
   exposes a library which allows you to capture these tensors and save
   them for analysis. It’s highly customizable and allows to save the
   specific tensors you want at different frequencies and
   configurations. Refer to the
   `DeveloperGuide <https://github.com/awslabs/sagemaker-debugger/tree/master/docs>`__
   for details on how to use Amazon SageMaker Debugger with your choice
   of framework in your training script.

Analysis of tensors
~~~~~~~~~~~~~~~~~~~

Once tensors are saved, Amazon SageMaker Debugger can be configured to
run debugging **Rules** on them. On a very broad level, a rule is a
python script used to detect certain conditions during training. Some of
the conditions that a data scientist training an algorithm might be
interested in are monitoring for gradients getting too large or too
small, detecting overfitting, and so on. Amazon SageMaker Debugger comes
pre-packaged with certain built-in rules. You can also write your own
rules using the Amazon SageMaker Debugger APIs. You can also analyze raw
tensor data outside of the Rules construct in a notebook, using Amazong
Sagemaker Debugger’s full set of APIs.

Setup
-----

To successfully execute this example, the following packages need to be
installed in your container:

-  PyTorch v1.3.1
-  Torchvision v0.4.2
-  Amazon SageMaker Debugger (smdebug)

``!python -m pip install smdebug``

Bring Your Own PyTorch for training
-----------------------------------

In this notebook, we will train a PyTorch model with Amazon SageMaker
Debugger enabled. We can do that by using custom PyTorch container,
enabling Amazon SageMaker Debugger in the training script, and bringing
it to Amazon SageMaker for training.

Note: The changes to the training script that are mentioned in this
notebook are only required when a custom container is used. Amazon
SageMaker Debugger will be automatically enabled (and not require any
changes to training script) if you use the SageMaker Deep Learning
Container for PyTorch.

We will focus on how to modify a training script to save tensors by
registering debug hooks and specifying which tensors to save.

The model used for this notebook is trained with the MNIST dataset. The
example is based on
https://github.com/pytorch/examples/blob/master/mnist/main.py

Modifying the training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we define the Estimator object and start training, we will
explore parts of the training script in detail. (The entire training
script can be found at
`./scripts/pytorch_mnist.py <./scripts/pytorch_mnist.py>`__).

Step 1: Import Amazon SageMaker Debugger.

.. code:: python

   import smdebug.pytorch as smd

Step 2: Create a debugger hook to save tensors of specified collections.
Apart from a list of collections, the hook takes the save config and
output directory as parameters. The output directory is a mandatory
parameter. All these parameters can be specified in the config json
file.

.. code:: python

   def create_smdebug_hook():
       # This allows you to create the hook from the configuration you pass to the SageMaker pySDK
       hook = smd.Hook.create_from_json_file()
       return hook

Step 3: Register the hook for all layers in the model

.. code:: python

   hook.register_hook(model)

Step 4: For PyTorch, if you use a Loss module for loss, add a step to
register loss

.. code:: python

   hook.register_loss(criterion)

Once these changes are made in the training script, Amazon SageMaker
Debugger will start saving tensors, belonging to the specified
collections, during training into the specfied output directory.

Now, we will setup the Estimator and start training using modified
training script.

.. code:: ipython3

    from __future__ import absolute_import
    
    import boto3
    import pytest
    from sagemaker.pytorch import PyTorch
    from sagemaker import get_execution_role
    from sagemaker.debugger import Rule, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig, rule_configs

Define the configuration of training to run. ``ecr_image`` is where you
can provide link to your bring-your-own-container. ``hyperparameters``
are fed into the training script with data directory (directory where
the training dataset is stored) and smdebug directory (directory where
the tensors will be saved) are mandatory fields.

.. code:: ipython3

    role = get_execution_role()
    training_dir = '/tmp/pytorch-smdebug'
    smdebug_mnist_script = 'scripts/pytorch_mnist.py'
    
    hyperparameters = {'random_seed': True, 'num_steps': 50, 'epochs': 5,
                       'data_dir':training_dir}

“rules” is a new parameter that will accept a list of rules you wish to
evaluate the tensors output against. For rules, Amazon SageMaker
Debugger supports two types: \* SageMaker Rules: These are rules
specially curated by the data science and engineering teams in Amazon
SageMaker which you can opt to evaluate against your training job. \*
Custom Rules: You can optionally choose to write your own rule as a
Python source file and have it evaluated against your training job. To
provide Amazon SageMaker Debugger to evaluate this rule, you would have
to provide the S3 location of the rule source and the evaluator image.

In this example, we will use the VanishingGradient which will attempt to
evaluate if there are vanishing gradients. Alternatively, you could
write your own custom rule, as demonstrated in
`this <https://github.com/aws/amazon-sagemaker-examples-staging/blob/master/sagemaker-debugger/tensorflow_keras_custom_rule/tf-keras-custom-rule.ipynb>`__
example.

.. code:: ipython3

    rules = [
        Rule.sagemaker(rule_configs.vanishing_gradient())
    ]
    
    estimator = PyTorch(entry_point=smdebug_mnist_script,
                      base_job_name='smdebugger-demo-mnist-pytorch',
                      role=role,
                      train_instance_count=1,
                      train_instance_type='ml.m4.xlarge',
                      train_volume_size=400,
                      train_max_run=3600,
                      hyperparameters=hyperparameters,
                      framework_version='1.3.1',
                      py_version='py3',
                      ## New parameter
                      rules = rules
                     )

*Note that Amazon Sagemaker Debugger is only supported for
py_version=‘py3’.*

With the next step we kick off traning job using Estimator object we
created above. Note that the way training job starts here is
asynchronous. That means that notebook is not blocked and control flow
is passed to next cell.

.. code:: ipython3

    estimator.fit(wait=False)

Result
~~~~~~

As a result of calling the fit() Amazon SageMaker Debugger kicked off a
rule evaluation job to monitor loss decrease, in parallel with the
training job. The rule evaluation status(es) will be visible in the
training logs at regular intervals. As you can see, in the summary,
there was no step in the training which reported vanishing gradients in
the tensors. Although, the loss was not found to be decreasing at step
1900.

.. code:: ipython3

    estimator.latest_training_job.rule_job_summary()




.. parsed-literal::

    [{'RuleConfigurationName': 'VanishingGradient',
      'RuleEvaluationJobArn': 'arn:aws:sagemaker:us-west-2:072677473360:processing-job/smdebugger-demo-mnist-pyto-vanishinggradient-52ca2f8e',
      'RuleEvaluationStatus': 'NoIssuesFound',
      'LastModifiedTime': datetime.datetime(2019, 12, 3, 0, 50, 53, 50000, tzinfo=tzlocal())}]



.. code:: ipython3

    def _get_rule_job_name(training_job_name, rule_configuration_name, rule_job_arn):
            """Helper function to get the rule job name with correct casing"""
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

    {'VanishingGradient': 'https://us-west-2.console.aws.amazon.com/cloudwatch/home?region=us-west-2#logStream:group=/aws/sagemaker/ProcessingJobs;prefix=smdebugger-demo-mnist-pyto-VanishingGradient-52ca2f8e;streamFilter=typeLogStreamPrefix'}



Analysis
~~~~~~~~

Another aspect of the Amazon SageMaker Debugger is analysis. It allows
us to perform interactive exploration of the tensors saved in real time
or after the job. Here we focus on after-the-fact analysis of the above
job. We import the smdebug library, which defines a concept of Trial
that represents a single training run. Note how we fetch the path to
debugger artifacts for the above job.

.. code:: ipython3

    from smdebug.trials import create_trial
    trial = create_trial(estimator.latest_job_debugger_artifacts_path())


.. parsed-literal::

    [2019-12-03 00:50:59.439 ip-172-16-56-202:4023 INFO s3_trial.py:42] Loading trial debug-output at path s3://sagemaker-us-west-2-072677473360/smdebugger-demo-mnist-pytorch-2019-12-03-00-44-45-065/debug-output


We can list all the tensors that were recorded to know what we want to
plot.

.. code:: ipython3

    trial.tensor_names()


.. parsed-literal::

    [2019-12-03 00:51:01.336 ip-172-16-56-202:4023 INFO trial.py:197] Training has ended, will refresh one final time in 1 sec.
    [2019-12-03 00:51:02.375 ip-172-16-56-202:4023 INFO trial.py:209] Loaded all steps




.. parsed-literal::

    ['CrossEntropyLoss_input_0',
     'CrossEntropyLoss_input_1',
     'CrossEntropyLoss_output_0',
     'gradient/Net_conv1.bias',
     'gradient/Net_conv1.weight',
     'gradient/Net_conv2.bias',
     'gradient/Net_conv2.weight',
     'gradient/Net_fc1.bias',
     'gradient/Net_fc1.weight',
     'gradient/Net_fc2.bias',
     'gradient/Net_fc2.weight']



We can also retrieve tensors by some default collections that smdebug
creates from your training job. Here we are interested in the losses
collection, so we can retrieve the names of tensors in losses collection
as follows. Amazon SageMaker Debugger creates default collections such
as weights, gradients, biases, losses automatically. You can also create
custom collections from your tensors.

.. code:: ipython3

    trial.tensor_names(collection="losses")




.. parsed-literal::

    ['CrossEntropyLoss_input_0',
     'CrossEntropyLoss_input_1',
     'CrossEntropyLoss_output_0']



