AWS Step Functions Data Science SDK - Hello World
=================================================

1. `Introduction <#Introduction>`__
2. `Setup <#Setup>`__
3. `Create steps for your workflow <#Create-steps-for-your-workflow>`__
4. `Define the workflow instance <#Define-the-workflow-instance>`__
5. `Review the Amazon States Language code for your
   workflow <#Review-the-Amazon-States-Language-code-for-your-workflow>`__
6. `Create the workflow on AWS Step
   Functions <#Create-the-workflow-on-AWS-Step-Functions>`__
7. `Execute the workflow <#Execute-the-workflow>`__
8. `Review the execution progress <#Review-the-execution-progress>`__
9. `Review the execution history <#Review-the-execution-history>`__

Introduction
------------

This notebook describes using the AWS Step Functions Data Science SDK to
create and manage workflows. The Step Functions SDK is an open source
library that allows data scientists to easily create and execute machine
learning workflows using AWS Step Functions and Amazon SageMaker. For
more information, see the following. \* `AWS Step
Functions <https://aws.amazon.com/step-functions/>`__ \* `AWS Step
Functions Developer
Guide <https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html>`__
\* `AWS Step Functions Data Science
SDK <https://aws-step-functions-data-science-sdk.readthedocs.io>`__

In this notebook we will use the SDK to create steps, link them together
to create a workflow, and execute the workflow in AWS Step Functions.

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

Import the required modules from the SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import stepfunctions
    import logging
    
    from stepfunctions.steps import *
    from stepfunctions.workflow import Workflow
    
    stepfunctions.set_stream_logger(level=logging.INFO)
    
    workflow_execution_role = "<execution-role-arn>" # paste the StepFunctionsWorkflowExecutionRole ARN from above

Create basic workflow
---------------------

In the following cell, you will define the step that you will use in our
first workflow. Then you will create, visualize and execute the
workflow.

Steps relate to states in AWS Step Functions. For more information, see
`States <https://docs.aws.amazon.com/step-functions/latest/dg/concepts-states.html>`__
in the *AWS Step Functions Developer Guide*. For more information on the
AWS Step Functions Data Science SDK APIs, see:
https://aws-step-functions-data-science-sdk.readthedocs.io.

Pass state
~~~~~~~~~~

A ``Pass`` state in Step Functions simply passes its input to its
output, without performing work. See
`Pass <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Pass>`__
in the AWS Step Functions Data Science SDK documentation.

.. code:: ipython3

    start_pass_state = Pass(
        state_id="MyPassState"             
    )

Chain together steps for the basic path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following cell links together the steps you’ve created into a
sequential group called ``basic_path``. We will chain a single step to
create our basic path. See
`Chain <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Chain>`__
in the AWS Step Functions Data Science SDK documentation.

After chaining together the steps for the basic path, in this case only
one step, we will visualize the basic path.

.. code:: ipython3

    # First we chain the start pass state
    basic_path=Chain([start_pass_state])

Define the workflow instance
----------------------------

The following cell defines the
`workflow <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow>`__
with the path we just defined.

After defining the workflow, we will render the graph to see what our
workflow looks like.

.. code:: ipython3

    # Next, we define the workflow
    basic_workflow = Workflow(
        name="MyWorkflow_Simple",
        definition=basic_path,
        role=workflow_execution_role
    )
    
    #Render the workflow
    basic_workflow.render_graph()

Review the Amazon States Language code for your workflow
--------------------------------------------------------

The following renders the JSON of the `Amazon States
Language <https://docs.aws.amazon.com/step-functions/latest/dg/concepts-amazon-states-language.html>`__
definition of the workflow you created.

.. code:: ipython3

    print(basic_workflow.definition.to_json(pretty=True))

Create the workflow on AWS Step Functions
-----------------------------------------

Create the workflow in AWS Step Functions with
`create <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.create>`__.

.. code:: ipython3

    basic_workflow.create()

Execute the workflow
--------------------

Run the workflow with
`execute <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.execute>`__.
Since the workflow only has a pass state, it will succeed immediately.

.. code:: ipython3

    basic_workflow_execution = basic_workflow.execute()

Review the execution progress
-----------------------------

Render workflow progress with the
`render_progress <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.render_progress>`__.

This generates a snapshot of the current state of your workflow as it
executes. This is a static image. Run the cell again to check progress.

.. code:: ipython3

    basic_workflow_execution.render_progress()

Review the execution history
----------------------------

Use
`list_events <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.list_events>`__
to list all events in the workflow execution.

.. code:: ipython3

    basic_workflow_execution.list_events(html=True)

Create additional steps for your workflow
-----------------------------------------

In the following cells, you will define additional steps that you will
use in our workflows. Steps relate to states in AWS Step Functions. For
more information, see
`States <https://docs.aws.amazon.com/step-functions/latest/dg/concepts-states.html>`__
in the *AWS Step Functions Developer Guide*. For more information on the
AWS Step Functions Data Science SDK APIs, see:
https://aws-step-functions-data-science-sdk.readthedocs.io.

Choice state
~~~~~~~~~~~~

A ``Choice`` state in Step Functions adds branching logic to your
workflow. See
`Choice <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Choice>`__
in the AWS Step Functions Data Science SDK documentation.

.. code:: ipython3

    choice_state = Choice(
        state_id="Is this Hello World example?"
    )

First create the steps for the “happy path”.

Wait state
~~~~~~~~~~

A ``Wait`` state in Step Functions waits a specific amount of time. See
`Wait <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Wait>`__
in the AWS Step Functions Data Science SDK documentation.

.. code:: ipython3

    wait_state = Wait(
        state_id="Wait for 3 seconds",
        seconds=3
    )

Parallel state
~~~~~~~~~~~~~~

A ``Parallel`` state in Step Functions is used to create parallel
branches of execution in your workflow. This creates the ``Parallel``
step and adds two branches: ``Hello`` and ``World``. See
`Parallel <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Parallel>`__
in the AWS Step Functions Data Science SDK documentation.

.. code:: ipython3

    parallel_state = Parallel("MyParallelState")
    parallel_state.add_branch(
        Pass(state_id="Hello")
    )
    parallel_state.add_branch(
        Pass(state_id="World")
    )

Lambda Task state
~~~~~~~~~~~~~~~~~

A ``Task`` State in Step Functions represents a single unit of work
performed by a workflow. Tasks can call Lambda functions and orchestrate
other AWS services. See `AWS Service
Integrations <https://docs.aws.amazon.com/step-functions/latest/dg/concepts-service-integrations.html>`__
in the *AWS Step Functions Developer Guide*.

Create a Lambda function
^^^^^^^^^^^^^^^^^^^^^^^^

The Lambda task state in this workflow uses a simple Lambda function
**(Python 3.x)** that returns a base64 encoded string. Create the
following function in the `Lambda
console <https://console.aws.amazon.com/lambda/>`__.

.. code:: python

   import json
   import base64
    
   def lambda_handler(event, context):
       return {
           'statusCode': 200,
           'input': event['input'],
           'output': base64.b64encode(event['input'].encode()).decode('UTF-8')
       }

Define the Lambda Task state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following creates a
`LambdaStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/compute.html#stepfunctions.steps.compute.LambdaStep>`__
called ``lambda_state``, and then configures the options to
`Retry <https://docs.aws.amazon.com/step-functions/latest/dg/concepts-error-handling.html#error-handling-retrying-after-an-error>`__
if the Lambda function fails.

.. code:: ipython3

    lambda_state = LambdaStep(
        state_id="Convert HelloWorld to Base64",
        parameters={  
            "FunctionName": "<lambda-function-name>", #replace with the name of the function you created
            "Payload": {  
               "input": "HelloWorld"
            }
        }
    )
    
    lambda_state.add_retry(Retry(
        error_equals=["States.TaskFailed"],
        interval_seconds=15,
        max_attempts=2,
        backoff_rate=4.0
    ))
    
    lambda_state.add_catch(Catch(
        error_equals=["States.TaskFailed"],
        next_step=Fail("LambdaTaskFailed")
    ))

Succeed state
~~~~~~~~~~~~~

A ``Succeed`` state in Step Functions stops an execution successfully.
See
`Succeed <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Succeed>`__
in the AWS Step Functions Data Science SDK documentation.

.. code:: ipython3

    succeed_state = Succeed("HelloWorldSuccessful")

Chain together steps for the happy path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following cell links together the steps you’ve created above into a
sequential group called ``happy_path``. The new path sequentially
includes the Wait state, the Parallel state, the Lambda state, and the
Succeed state that you created earlier.

After chaining together the steps for the happy path, we will define the
workflow and visualize the happy path.

.. code:: ipython3

    happy_path = Chain([wait_state, parallel_state, lambda_state, succeed_state])

.. code:: ipython3

    # Next, we define the workflow
    happy_workflow = Workflow(
        name="MyWorkflow_Happy",
        definition=happy_path,
        role=workflow_execution_role
    )
    
    happy_workflow.render_graph()

For the sad path, we simply end the workflow using a ``Fail`` state. See
`Fail <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Fail>`__
in the AWS Step Functions Data Science SDK documentation.

.. code:: ipython3

    sad_state = Fail("HelloWorldFailed")

Choice state
~~~~~~~~~~~~

Now, attach branches to the Choice state you created earlier. See
*Choice Rules* in the `AWS Step Functions Data Science SDK
documentation <https://aws-step-functions-data-science-sdk.readthedocs.io>`__
.

.. code:: ipython3

    choice_state.add_choice(
        rule=ChoiceRule.BooleanEquals(variable=start_pass_state.output()["IsHelloWorldExample"], value=True),
        next_step=happy_path
    )
    choice_state.add_choice(
        ChoiceRule.BooleanEquals(variable=start_pass_state.output()["IsHelloWorldExample"], value=False),
        next_step=sad_state
    )

Chain together two steps
------------------------

In the next cell, you will chain the start_pass_state with the
choice_state and define the workflow.

.. code:: ipython3

    # First we chain the start pass state and the choice state
    branching_workflow_definition=Chain([start_pass_state, choice_state])
    
    # Next, we define the workflow
    branching_workflow = Workflow(
        name="MyWorkflow_v2",
        definition=branching_workflow_definition,
        role=workflow_execution_role
    )

.. code:: ipython3

    # Review the Amazon States Language code for your workflow
    print(branching_workflow.definition.to_json(pretty=True))

Review a visualization for your workflow
----------------------------------------

The following cell generates a graphical representation of your
workflow.

.. code:: ipython3

    branching_workflow.render_graph(portrait=False)

Create the workflow and execute
-------------------------------

In the next cells, we will create the branching happy workflow in AWS
Step Functions with
`create <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.create>`__
and execute it with
`execute <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.execute>`__.

Since IsHelloWorldExample is set to True, your execution should follow
the happy path.

.. code:: ipython3

    branching_workflow.create()

.. code:: ipython3

    branching_workflow_execution = branching_workflow.execute(inputs={
            "IsHelloWorldExample": True
    })

Review the progress
-------------------

Review the workflow progress with the
`render_progress <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.render_progress>`__.

Review the execution history by calling
`list_events <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.list_events>`__
to list all events in the workflow execution.

.. code:: ipython3

    branching_workflow_execution.render_progress()

.. code:: ipython3

    branching_workflow_execution.list_events(html=True)
