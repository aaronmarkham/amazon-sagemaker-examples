MNIST Training using PyTorch and Step Functions
===============================================

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Data <#Data>`__
4. `Train <#Train>`__

--------------

Background
----------

MNIST is a widely used dataset for handwritten digit classification. It
consists of 70,000 labeled 28x28 pixel grayscale images of hand-written
digits. The dataset is split into 60,000 training images and 10,000 test
images. There are 10 classes (one for each of the 10 digits). This
tutorial will show how to train and test an MNIST model on SageMaker
using PyTorch.

For more information about PyTorch in SageMaker, please visit
`sagemaker-pytorch-containers <https://github.com/aws/sagemaker-pytorch-containers>`__
and
`sagemaker-python-sdk <https://github.com/aws/sagemaker-python-sdk>`__
github repositories.

--------------

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install --upgrade stepfunctions

Setup
-----

Add a policy to your SageMaker role in IAM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**If you are running this notebook on an Amazon SageMaker notebook
instance**, the IAM role assumed by your notebook instance needs
permission to create and run workflows in AWS Step Functions. To provide
this permission to the role, do the following.

1. Open the Amazon `SageMaker
   console <https://console.aws.amazon.com/sagemaker/>`__.
2. Select **Notebook instances** and choose the name of your notebook
   instance
3. Under **Permissions and encryption** select the role ARN to view the
   role on the IAM console
4. Choose **Attach policies** and search for
   ``AWSStepFunctionsFullAccess``.
5. Select the check box next to ``AWSStepFunctionsFullAccess`` and
   choose **Attach policy**

If you are running this notebook in a local environment, the SDK will
use your configured AWS CLI configuration. For more information, see
`Configuring the AWS
CLI <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`__.

Next, create an execution role in IAM for Step Functions.

Create an execution role for Step Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need an execution role so that you can create and execute workflows
in Step Functions.

1. Go to the `IAM console <https://console.aws.amazon.com/iam/>`__
2. Select **Roles** and then **Create role**.
3. Under **Choose the service that will use this role** select **Step
   Functions**
4. Choose **Next** until you can enter a **Role name**
5. Enter a name such as ``StepFunctionsWorkflowExecutionRole`` and then
   select **Create role**

Attach a policy to the role you created. The following steps attach a
policy that provides full access to Step Functions, however as a good
practice you should only provide access to the resources you need.

1. Under the **Permissions** tab, click **Add inline policy**
2. Enter the following in the **JSON** tab

.. code:: json

   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "sagemaker:CreateTransformJob",
                   "sagemaker:DescribeTransformJob",
                   "sagemaker:StopTransformJob",
                   "sagemaker:CreateTrainingJob",
                   "sagemaker:DescribeTrainingJob",
                   "sagemaker:StopTrainingJob",
                   "sagemaker:CreateHyperParameterTuningJob",
                   "sagemaker:DescribeHyperParameterTuningJob",
                   "sagemaker:StopHyperParameterTuningJob",
                   "sagemaker:CreateModel",
                   "sagemaker:CreateEndpointConfig",
                   "sagemaker:CreateEndpoint",
                   "sagemaker:DeleteEndpointConfig",
                   "sagemaker:DeleteEndpoint",
                   "sagemaker:UpdateEndpoint",
                   "sagemaker:ListTags",
                   "lambda:InvokeFunction",
                   "sqs:SendMessage",
                   "sns:Publish",
                   "ecs:RunTask",
                   "ecs:StopTask",
                   "ecs:DescribeTasks",
                   "dynamodb:GetItem",
                   "dynamodb:PutItem",
                   "dynamodb:UpdateItem",
                   "dynamodb:DeleteItem",
                   "batch:SubmitJob",
                   "batch:DescribeJobs",
                   "batch:TerminateJob",
                   "glue:StartJobRun",
                   "glue:GetJobRun",
                   "glue:GetJobRuns",
                   "glue:BatchStopJobRun"
               ],
               "Resource": "*"
           },
           {
               "Effect": "Allow",
               "Action": [
                   "iam:PassRole"
               ],
               "Resource": "*",
               "Condition": {
                   "StringEquals": {
                       "iam:PassedToService": "sagemaker.amazonaws.com"
                   }
               }
           },
           {
               "Effect": "Allow",
               "Action": [
                   "events:PutTargets",
                   "events:PutRule",
                   "events:DescribeRule"
               ],
               "Resource": [
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTuningJobsRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForECSTaskRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForBatchJobsRule"
               ]
           }
       ]
   }

3. Choose **Review policy** and give the policy a name such as
   ``StepFunctionsWorkflowExecutionPolicy``
4. Choose **Create policy**. You will be redirected to the details page
   for the role.
5. Copy the **Role ARN** at the top of the **Summary**

Import the required modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now import the required modules from the Step Functions SDK and AWS
SageMaker, configure an S3 bucket, and get the AWS SageMaker execution
role.

.. code:: ipython3

    import sagemaker
    import stepfunctions
    import logging
    
    from stepfunctions.template.pipeline import TrainingPipeline
    
    sagemaker_session = sagemaker.Session()
    stepfunctions.set_stream_logger(level=logging.INFO)
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-mnist'
    
    # SageMaker Execution Role
    # You can use sagemaker.get_execution_role() if running inside sagemaker's notebook instance
    sagemaker_execution_role = sagemaker.get_execution_role() #Replace with ARN if not in an AWS SageMaker notebook
    
    # paste the StepFunctionsWorkflowExecutionRole ARN from above
    workflow_execution_role = "<execution-role-arn>" 

Data
----

Getting the data
~~~~~~~~~~~~~~~~

.. code:: ipython3

    from torchvision import datasets, transforms
    
    datasets.MNIST('data', download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

.. code:: ipython3

    inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
    print('input spec (in this case, just an S3 path): {}'.format(inputs))

Uploading the data to S3
~~~~~~~~~~~~~~~~~~~~~~~~

We are going to use the ``sagemaker.Session.upload_data`` function to
upload our datasets to an S3 location. The return value inputs
identifies the location – we will use later when we start the training
job.

Train
-----

Training script
~~~~~~~~~~~~~~~

The ``mnist.py`` script provides all the code we need for training and
hosting a SageMaker model (``model_fn`` function to load a model). The
training script is very similar to a training script you might run
outside of SageMaker, but you can access useful properties about the
training environment through various environment variables, such as:

-  ``SM_MODEL_DIR``: A string representing the path to the directory to
   write model artifacts to. These artifacts are uploaded to S3 for
   model hosting.
-  ``SM_NUM_GPUS``: The number of gpus available in the current
   container.
-  ``SM_CURRENT_HOST``: The name of the current container on the
   container network.
-  ``SM_HOSTS``: JSON encoded list containing all the hosts .

Supposing one input channel, ‘training’, was used in the call to the
PyTorch estimator’s ``fit()`` method, the following will be set,
following the format ``SM_CHANNEL_[channel_name]``:

-  ``SM_CHANNEL_TRAINING``: A string representing the path to the
   directory containing data in the ‘training’ channel.

For more information about training environment variables, please visit
`SageMaker Containers <https://github.com/aws/sagemaker-containers>`__.

A typical training script loads data from the input channels, configures
training with hyperparameters, trains a model, and saves a model to
``model_dir`` so that it can be hosted later. Hyperparameters are passed
to your script as arguments and can be retrieved with an
``argparse.ArgumentParser`` instance.

Because the SageMaker imports the training script, you should put your
training code in a main guard (``if __name__=='__main__':``) if you are
using the same script to host your model as we do in this example, so
that SageMaker does not inadvertently run your training code at the
wrong point in execution.

For example, the script run by this notebook:

.. code:: ipython3

    !pygmentize mnist.py

Use Step Functions to run training in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``PyTorch`` class allows us to run our training function as a
training job on SageMaker. We need to configure it with our training
script, an IAM role, the number of training instances, the training
instance type, and hyperparameters. In this case we are going to run our
training job on 2 ``ml.c4.xlarge`` instances. But this example can be
ran on one or multiple, cpu or gpu instances (`full list of available
instances <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__).
The hyperparameters parameter is a dict of values that will be passed to
your training script – you can see how to access these values in the
``mnist.py`` script above.

.. code:: ipython3

    from sagemaker.pytorch import PyTorch
    
    estimator = PyTorch(entry_point='mnist.py',
                        role=sagemaker_execution_role,
                        framework_version='1.2.0',
                        train_instance_count=2,
                        train_instance_type='ml.c4.xlarge',
                        hyperparameters={
                            'epochs': 6,
                            'backend': 'gloo'
                        })

Build a training pipeline with the Step Functions SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A typical task for a data scientist is to train a model and deploy that
model to an endpoint. Without the Step Functions SDK, this is a four
step process on SageMaker that includes the following.

1. Training the model
2. Creating the model on SageMaker
3. Creating an endpoint configuration
4. Deploying the trained model to the configured endpoint

The Step Functions SDK provides the
`TrainingPipeline <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/pipelines.html#stepfunctions.template.pipeline.train.TrainingPipeline>`__
API to simplify this procedure. The following configures ``pipeline``
with the necessary parameters to define a training pipeline.

.. code:: ipython3

    pipeline = TrainingPipeline(
        estimator=estimator,
        role=workflow_execution_role,
        inputs=inputs,
        s3_bucket=bucket
    )

Visualize the pipeline
~~~~~~~~~~~~~~~~~~~~~~

You can now view the workflow definition, and also visualize it as a
graph. This workflow and graph represent your training pipeline.

View the workflow definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    print(pipeline.workflow.definition.to_json(pretty=True))

Visualize the workflow graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    pipeline.render_graph()

Create and execute the pipeline on AWS Step Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the pipeline in AWS Step Functions with
`create <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.create>`__.

.. code:: ipython3

    pipeline.create()

Run the workflow with
`execute <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.execute>`__.
A link will be provided after the following cell is executed. Following
this link, you can monitor your pipeline execution on Step Functions’
console.

.. code:: ipython3

    pipeline.execute()
