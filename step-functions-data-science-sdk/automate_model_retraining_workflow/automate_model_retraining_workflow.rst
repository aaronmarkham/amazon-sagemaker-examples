Automate Model Retraining & Deployment Using the AWS Step Functions Data Science SDK
====================================================================================

1. `Introduction <#Introduction>`__
2. `Setup <#Setup>`__
3. `Create Resources <#Create-Resources>`__
4. `Build a Machine Learning
   Workflow <#Build-a-Machine-Learning-Workflow>`__
5. `Run the Workflow <#Run-the-Workflow>`__
6. `Clean Up <#Clean-Up>`__

Introduction
------------

This notebook describes how to use the AWS Step Functions Data Science
SDK to create a machine learning model retraining workflow. The Step
Functions SDK is an open source library that allows data scientists to
easily create and execute machine learning workflows using AWS Step
Functions and Amazon SageMaker. For more information, please see the
following resources: \* `AWS Step
Functions <https://aws.amazon.com/step-functions/>`__ \* `AWS Step
Functions Developer
Guide <https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html>`__
\* `AWS Step Functions Data Science
SDK <https://aws-step-functions-data-science-sdk.readthedocs.io>`__

In this notebook, we will use the SDK to create steps that capture and
transform data using AWS Glue, encorporate this data into the training
of a machine learning model, deploy the model to a SageMaker endpoint,
link these steps together to create a workflow, and then execute the
workflow in AWS Step Functions.

Setup
-----

First, we’ll need to install and load all the required modules. Then
we’ll create fine-grained IAM roles for the Lambda, Glue, and Step
Functions resources that we will create. The IAM roles grant the
services permissions within your AWS environment.

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install --upgrade stepfunctions

Import the Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import uuid
    import logging
    import stepfunctions
    import boto3
    import sagemaker
    
    from sagemaker.amazon.amazon_estimator import get_image_uri
    from sagemaker import s3_input
    from sagemaker.s3 import S3Uploader
    from stepfunctions import steps
    from stepfunctions.steps import TrainingStep, ModelStep
    from stepfunctions.inputs import ExecutionInput
    from stepfunctions.workflow import Workflow
    
    session = sagemaker.Session()
    stepfunctions.set_stream_logger(level=logging.INFO)
    
    region = boto3.Session().region_name
    bucket = session.default_bucket()
    id = uuid.uuid4().hex
    
    #Create a unique name for the AWS Glue job to be created. If you change the 
    #default name, you may need to change the Step Functions execution role.
    job_name = 'glue-customer-churn-etl-{}'.format(id)
    
    #Create a unique name for the AWS Lambda function to be created. If you change
    #the default name, you may need to change the Step Functions execution role.
    function_name = 'query-training-status-{}'.format(id)

Next, we’ll create fine-grained IAM roles for the Lambda, Glue, and Step
Functions resources. The IAM roles grant the services permissions within
your AWS environment.

Add permissions to your notebook role in IAM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IAM role assumed by your notebook requires permission to create and
run workflows in AWS Step Functions. If this notebook is running on a
SageMaker notebook instance, do the following to provide IAM permissions
to the notebook:

1. Open the Amazon `SageMaker
   console <https://console.aws.amazon.com/sagemaker/>`__.
2. Select **Notebook instances** and choose the name of your notebook
   instance.
3. Under **Permissions and encryption** select the role ARN to view the
   role on the IAM console.
4. Copy and save the IAM role ARN for later use.
5. Choose **Attach policies** and search for
   ``AWSStepFunctionsFullAccess``.
6. Select the check box next to ``AWSStepFunctionsFullAccess`` and
   choose **Attach policy**.

We also need to provide permissions that allow the notebook instance the
ability to create an AWS Lambda function and AWS Glue job. We will edit
the managed policy attached to our role directly to encorporate these
specific permissions:

1. Under **Permisions policies** expand the
   AmazonSageMaker-ExecutionPolicy-*******\* policy and choose **Edit
   policy**.
2. Select **Add additional permissions**. Choose **IAM** for Service and
   **PassRole** for Actions.
3. Under Resources, choose **Specific**. Select **Add ARN** and enter
   ``query_training_status-role`` for **Role name with path**\ \* and
   choose **Add**. You will create this role later on in this notebook.
4. Select **Add additional permissions** a second time. Choose
   **Lambda** for Service, **Write** for Access level, and **All
   resources** for Resources.
5. Select **Add additional permissions** a final time. Choose **Glue**
   for Service, **Write** for Access level, and **All resources** for
   Resources.
6. Choose **Review policy** and then **Save changes**.

If you are running this notebook outside of SageMaker, the SDK will use
your configured AWS CLI configuration. For more information, see
`Configuring the AWS
CLI <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`__.

Next, let’s create an execution role in IAM for Step Functions.

Create an Execution Role for Step Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your Step Functions workflow requires an IAM role to interact with other
services in your AWS environment.

1. Go to the `IAM console <https://console.aws.amazon.com/iam/>`__.
2. Select **Roles** and then **Create role**.
3. Under **Choose the service that will use this role** select **Step
   Functions**.
4. Choose **Next** until you can enter a **Role name**.
5. Enter a name such as ``StepFunctionsWorkflowExecutionRole`` and then
   select **Create role**.

Next, create and attach a policy to the role you created. As a best
practice, the following steps will attach a policy that only provides
access to the specific resources and actions needed for this solution.

1. Under the **Permissions** tab, click **Attach policies** and then
   **Create policy**.
2. Enter the following in the **JSON** tab:

.. code:: json

   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": "iam:PassRole",
               "Resource": "NOTEBOOK_ROLE_ARN",
               "Condition": {
                   "StringEquals": {
                       "iam:PassedToService": "sagemaker.amazonaws.com"
                   }
               }
           },
           {
               "Effect": "Allow",
               "Action": [
                   "sagemaker:CreateModel",
                   "sagemaker:DeleteEndpointConfig",
                   "sagemaker:DescribeTrainingJob",
                   "sagemaker:CreateEndpoint",
                   "sagemaker:StopTrainingJob",
                   "sagemaker:CreateTrainingJob",
                   "sagemaker:UpdateEndpoint",
                   "sagemaker:CreateEndpointConfig",
                   "sagemaker:DeleteEndpoint"
               ],
               "Resource": [
                   "arn:aws:sagemaker:*:*:*"
               ]
           },
           {
               "Effect": "Allow",
               "Action": [
                   "events:DescribeRule",
                   "events:PutRule",
                   "events:PutTargets"
               ],
               "Resource": [
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule"
               ]
           },
           {
               "Effect": "Allow",
               "Action": [
                   "lambda:InvokeFunction"
               ],
               "Resource": [
                   "arn:aws:lambda:*:*:function:query-training-status*"
               ]
           },
           {
               "Effect": "Allow",
               "Action": [
                   "glue:StartJobRun",
                   "glue:GetJobRun",
                   "glue:BatchStopJobRun",
                   "glue:GetJobRuns"
               ],
               "Resource": "arn:aws:glue:*:*:job/glue-customer-churn-etl*"
           }
       ]
   }

3.  Replace **NOTEBOOK_ROLE_ARN** with the ARN for your notebook that
    you created in the previous step.
4.  Choose **Review policy** and give the policy a name such as
    ``StepFunctionsWorkflowExecutionPolicy``.
5.  Choose **Create policy**.
6.  Select **Roles** and search for your
    ``StepFunctionsWorkflowExecutionRole`` role.
7.  Under the **Permissions** tab, click **Attach policies**.
8.  Search for your newly created
    ``StepFunctionsWorkflowExecutionPolicy`` policy and select the check
    box next to it.
9.  Choose **Attach policy**. You will then be redirected to the details
    page for the role.
10. Copy the StepFunctionsWorkflowExecutionRole **Role ARN** at the top
    of the Summary.

Configure Execution Roles
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # paste the StepFunctionsWorkflowExecutionRole ARN from above
    workflow_execution_role = ''
    
    # SageMaker Execution Role
    # You can use sagemaker.get_execution_role() if running inside sagemaker's notebook instance
    sagemaker_execution_role = sagemaker.get_execution_role() #Replace with ARN if not in an AWS SageMaker notebook

Create a Glue IAM Role
^^^^^^^^^^^^^^^^^^^^^^

You need to create an IAM role so that you can create and execute an AWS
Glue Job on your data in Amazon S3.

1. Go to the `IAM console <https://console.aws.amazon.com/iam/>`__.
2. Select **Roles** and then **Create role**.
3. Under **Choose the service that will use this role** select **Glue**.
4. Choose **Next** until you can enter a **Role name**.
5. Enter a name such as ``AWS-Glue-S3-Bucket-Access`` and then select
   **Create role**.

Next, create and attach a policy to the role you created. The following
steps attach a managed policy that provides Glue access to the specific
S3 bucket holding your data.

1. Under the **Permissions** tab, click **Attach policies** and then
   **Create policy**.
2. Enter the following in the **JSON** tab:

.. code:: json

   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Sid": "ListObjectsInBucket",
               "Effect": "Allow",
               "Action": ["s3:ListBucket"],
               "Resource": ["arn:aws:s3:::BUCKET-NAME"]
           },
           {
               "Sid": "AllObjectActions",
               "Effect": "Allow",
               "Action": "s3:*Object",
               "Resource": ["arn:aws:s3:::BUCKET-NAME/*"]
           }
       ]
   }

3. Run the next cell (below) to retrieve the specific **S3 bucket name**
   that we will grant permissions to.

.. code:: ipython3

    session = sagemaker.Session()
    bucket = session.default_bucket()
    print(bucket)

4.  Copy the output of the above cell and replace the **two occurances**
    of **BUCKET-NAME** in the JSON text that you entered.
5.  Choose **Review policy** and give the policy a name such as
    ``S3BucketAccessPolicy``.
6.  Choose **Create policy**.
7.  Select **Roles**, then search for and select your
    ``AWS-Glue-S3-Bucket-Access`` role.
8.  Under the **Permissions** tab, click **Attach policies**.
9.  Search for your newly created ``S3BucketAccessPolicy`` policy and
    select the check box next to it.
10. Choose **Attach policy**. You will then be redirected to the details
    page for the role.
11. Copy the **Role ARN** at the top of the Summary tab.

.. code:: ipython3

    # paste the AWS-Glue-S3-Bucket-Access role ARN from above
    glue_role = ''

Create a Lambda IAM Role
^^^^^^^^^^^^^^^^^^^^^^^^

You also need to create an IAM role so that you can create and execute
an AWS Lambda function stored in Amazon S3.

1. Go to the `IAM console <https://console.aws.amazon.com/iam/>`__.
2. Select **Roles** and then **Create role**.
3. Under **Choose the service that will use this role** select
   **Lambda**.
4. Choose **Next** until you can enter a **Role name**.
5. Enter a name such as ``query_training_status-role`` and then select
   **Create role**.

Next, attach policies to the role you created. The following steps
attach policies that provides Lambda access to S3 and read-only access
to SageMaker.

1. Under the **Permissions** tab, click **Attach Policies**.
2. In the search box, type **SageMaker** and select
   **AmazonSageMakerReadOnly** from the populated list.
3. In the search box type **AWSLambda** and select
   **AWSLambdaBasicExecutionRole** from the populated list.
4. Choose **Attach policy**. You will then be redirected to the details
   page for the role.
5. Copy the **Role ARN** at the top of the **Summary**.

.. code:: ipython3

    # paste the query_training_status-role role ARN from above
    lambda_role = ''

Prepare the Dataset
~~~~~~~~~~~~~~~~~~~

This notebook uses the XGBoost algorithm to automate the classification
of unhappy customers for telecommunication service providers. The goal
is to identify customers who may cancel their service soon so that you
can entice them to stay. This is known as customer churn prediction.

The dataset we use is publicly available and was mentioned in the book
`Discovering Knowledge in
Data <https://www.amazon.com/dp/0470908742/>`__ by Daniel T. Larose. It
is attributed by the author to the University of California Irvine
Repository of Machine Learning Datasets.

.. code:: ipython3

    project_name = 'ml_deploy'
    
    data_source = S3Uploader.upload(local_path='./data/customer-churn.csv',
                                   desired_s3_uri='s3://{}/{}'.format(bucket, project_name),
                                   session=session)
    
    train_prefix = 'train'
    val_prefix = 'validation'
    
    train_data = 's3://{}/{}/{}/'.format(bucket, project_name, train_prefix)
    validation_data = 's3://{}/{}/{}/'.format(bucket, project_name, val_prefix)

Create Resources
----------------

In the following steps we’ll create the Glue job and Lambda function
that are called from the Step Functions workflow.

Create the AWS Glue Job
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    glue_script_location = S3Uploader.upload(local_path='./code/glue_etl.py',
                                   desired_s3_uri='s3://{}/{}'.format(bucket, project_name),
                                   session=session)
    glue_client = boto3.client('glue')
    
    response = glue_client.create_job(
        Name=job_name,
        Description='PySpark job to extract the data and split in to training and validation data sets',
        Role=glue_role, # you can pass your existing AWS Glue role here if you have used Glue before
        ExecutionProperty={
            'MaxConcurrentRuns': 2
        },
        Command={
            'Name': 'glueetl',
            'ScriptLocation': glue_script_location,
            'PythonVersion': '3'
        },
        DefaultArguments={
            '--job-language': 'python'
        },
        GlueVersion='1.0',
        WorkerType='Standard',
        NumberOfWorkers=2,
        Timeout=60
    )

Create the AWS Lambda Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import zipfile
    zip_name = 'query_training_status.zip'
    lambda_source_code = './code/query_training_status.py'
    
    zf = zipfile.ZipFile(zip_name, mode='w')
    zf.write(lambda_source_code, arcname=lambda_source_code.split('/')[-1])
    zf.close()
    
    
    S3Uploader.upload(local_path=zip_name, 
                      desired_s3_uri='s3://{}/{}'.format(bucket, project_name),
                      session=session)

.. code:: ipython3

    lambda_client = boto3.client('lambda')
    
    response = lambda_client.create_function(
        FunctionName=function_name,
        Runtime='python3.7',
        Role=lambda_role,
        Handler='query_training_status.lambda_handler',
        Code={
            'S3Bucket': bucket,
            'S3Key': '{}/{}'.format(project_name, zip_name)
        },
        Description='Queries a SageMaker training job and return the results.',
        Timeout=15,
        MemorySize=128
    )

Configure the AWS SageMaker Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    container = get_image_uri(region, 'xgboost')
    
    xgb = sagemaker.estimator.Estimator(container,
                                        sagemaker_execution_role, 
                                        train_instance_count=1, 
                                        train_instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/output'.format(bucket, project_name))
    
    xgb.set_hyperparameters(max_depth=5,
                            eta=0.2,
                            gamma=4,
                            min_child_weight=6,
                            subsample=0.8,
                            silent=0,
                            objective='binary:logistic',
                            eval_metric='error',
                            num_round=100)

Build a Machine Learning Workflow
---------------------------------

You can use a state machine workflow to create a model retraining
pipeline. The AWS Data Science Workflows SDK provides several AWS
SageMaker workflow steps that you can use to construct an ML pipeline.
In this tutorial you will create the following steps:

-  `ETLStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/compute.html#stepfunctions.steps.compute.GlueStartJobRunStep>`__
   - Starts an AWS Glue job to extract the latest data from our source
   database and prepare our data.
-  `TrainingStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.TrainingStep>`__
   - Creates the training step and passes the defined estimator.
-  `ModelStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.ModelStep>`__
   - Creates a model in SageMaker using the artifacts created during the
   TrainingStep.
-  `LambdaStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/compute.html#stepfunctions.steps.compute.LambdaStep>`__
   - Creates the task state step within our workflow that calls a Lambda
   function.
-  `ChoiceStateStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Choice>`__
   - Creates the choice state step within our workflow.
-  `EndpointConfigStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.EndpointConfigStep>`__
   - Creates the endpoint config step to define the new configuration
   for our endpoint.
-  `EndpointStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.EndpointStep>`__
   - Creates the endpoint step to update our model endpoint.
-  `FailStateStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Fail>`__
   - Creates fail state step within our workflow.

.. code:: ipython3

    # SageMaker expects unique names for each job, model and endpoint. 
    # If these names are not unique the execution will fail.
    execution_input = ExecutionInput(schema={
        'TrainingJobName': str,
        'GlueJobName': str,
        'ModelName': str,
        'EndpointName': str,
        'LambdaFunctionName': str
    })

Create an ETL step with AWS Glue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following cell, we create a Glue step thats runs an AWS Glue job.
The Glue job extracts the latest data from our source database, removes
unnecessary columns, splits the data in to training and validation sets,
and saves the data to CSV format in S3. Glue is performing this
extraction, transformation, and load (ETL) in a serverless fashion, so
there are no compute resources to configure and manage. See the
`GlueStartJobRunStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/compute.html#stepfunctions.steps.compute.GlueStartJobRunStep>`__
Compute step in the AWS Step Functions Data Science SDK documentation.

.. code:: ipython3

    etl_step = steps.GlueStartJobRunStep(
        'Extract, Transform, Load',
        parameters={"JobName": execution_input['GlueJobName'],
                    "Arguments":{
                        '--S3_SOURCE': data_source,
                        '--S3_DEST': 's3a://{}/{}/'.format(bucket, project_name),
                        '--TRAIN_KEY': train_prefix + '/',
                        '--VAL_KEY': val_prefix +'/'}
                   }
    )

Create a SageMaker Training Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following cell, we create the training step and pass the
estimator we defined above. See
`TrainingStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.TrainingStep>`__
in the AWS Step Functions Data Science SDK documentation to learn more.

.. code:: ipython3

    training_step = steps.TrainingStep(
        'Model Training', 
        estimator=xgb,
        data={
            'train': s3_input(train_data, content_type='csv'),
            'validation': s3_input(validation_data, content_type='csv')
        },
        job_name=execution_input['TrainingJobName'],
        wait_for_completion=True
    )

Create a Model Step
~~~~~~~~~~~~~~~~~~~

In the following cell, we define a model step that will create a model
in Amazon SageMaker using the artifacts created during the TrainingStep.
See
`ModelStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.ModelStep>`__
in the AWS Step Functions Data Science SDK documentation to learn more.

The model creation step typically follows the training step. The Step
Functions SDK provides the
`get_expected_model <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.TrainingStep.get_expected_model>`__
method in the TrainingStep class to provide a reference for the trained
model artifacts. Please note that this method is only useful when the
ModelStep directly follows the TrainingStep.

.. code:: ipython3

    model_step = steps.ModelStep(
        'Save Model',
        model=training_step.get_expected_model(),
        model_name=execution_input['ModelName'],
        result_path='$.ModelStepResults'
    )

Create a Lambda Step
~~~~~~~~~~~~~~~~~~~~

In the following cell, we define a lambda step that will invoke the
previously created lambda function as part of our Step Function
workflow. See
`LambdaStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/compute.html#stepfunctions.steps.compute.LambdaStep>`__
in the AWS Step Functions Data Science SDK documentation to learn more.

.. code:: ipython3

    lambda_step = steps.compute.LambdaStep(
        'Query Training Results',
        parameters={  
            "FunctionName": execution_input['LambdaFunctionName'],
            'Payload':{
                "TrainingJobName.$": '$.TrainingJobName'
            }
        }
    )

Create a Choice State Step
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following cell, we create a choice step in order to build a
dynamic workflow. This choice step branches based off of the results of
our SageMaker training step: did the training job fail or should the
model be saved and the endpoint be updated? We will add specfic rules to
this choice step later on in section 8 of this notebook.

.. code:: ipython3

    check_accuracy_step = steps.states.Choice(
        'Accuracy > 90%'
    )

Create an Endpoint Configuration Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following cell we create an endpoint configuration step. See
`EndpointConfigStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.sagemaker.EndpointConfigStep>`__
in the AWS Step Functions Data Science SDK documentation to learn more.

.. code:: ipython3

    endpoint_config_step = steps.EndpointConfigStep(
        "Create Model Endpoint Config",
        endpoint_config_name=execution_input['ModelName'],
        model_name=execution_input['ModelName'],
        initial_instance_count=1,
        instance_type='ml.m4.xlarge'
    )

Update the Model Endpoint Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following cell, we create the Endpoint step to deploy the new
model as a managed API endpoint, updating an existing SageMaker endpoint
if our choice state is sucessful.

.. code:: ipython3

    endpoint_step = steps.EndpointStep(
        'Update Model Endpoint',
        endpoint_name=execution_input['EndpointName'],
        endpoint_config_name=execution_input['ModelName'],
        update=False
    )

Create the Fail State Step
~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition, we create a Fail step which proceeds from our choice state
if the validation accuracy of our model is lower than the threshold we
define. See
`FailStateStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Fail>`__
in the AWS Step Functions Data Science SDK documentation to learn more.

.. code:: ipython3

    fail_step = steps.states.Fail(
        'Model Accuracy Too Low',
        comment='Validation accuracy lower than threshold'
    )

Add Rules to Choice State
~~~~~~~~~~~~~~~~~~~~~~~~~

In the cells below, we add a threshold rule to our choice state.
Therefore, if the validation accuracy of our model is below 0.90, we
move to the Fail State. If the validation accuracy of our model is above
0.90, we move to the save model step with proceeding endpoint update.
See
`here <https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst>`__
for more information on how XGBoost calculates classification error.

For binary classification problems the XGBoost algorithm defines the
model error as:

:raw-latex:`\begin{equation*}
\frac{incorret\:predictions}{total\:number\:of\:predictions}
\end{equation*}`

To achieve an accuracy of 90%, we need error <.10.

.. code:: ipython3

    threshold_rule = steps.choice_rule.ChoiceRule.NumericLessThan(variable=lambda_step.output()['Payload']['trainingMetrics'][0]['Value'], value=.1)
    
    check_accuracy_step.add_choice(rule=threshold_rule, next_step=endpoint_config_step)
    check_accuracy_step.default_choice(next_step=fail_step)

Link all the Steps Together
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, create your workflow definition by chaining all of the steps
together that we’ve created. See
`Chain <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/sagemaker.html#stepfunctions.steps.states.Chain>`__
in the AWS Step Functions Data Science SDK documentation to learn more.

.. code:: ipython3

    endpoint_config_step.next(endpoint_step)

.. code:: ipython3

    workflow_definition = steps.Chain([
        etl_step,
        training_step,
        model_step,
        lambda_step,
        check_accuracy_step
    ])

Run the Workflow
----------------

Create your workflow using the workflow definition above, and render the
graph with
`render_graph <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.render_graph>`__:

.. code:: ipython3

    workflow = Workflow(
        name='MyInferenceRoutine_{}'.format(id),
        definition=workflow_definition,
        role=workflow_execution_role,
        execution_input=execution_input
    )

.. code:: ipython3

    workflow.render_graph()

Create the workflow in AWS Step Functions with
`create <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.create>`__:

.. code:: ipython3

    workflow.create()

Run the workflow with
`execute <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.execute>`__:

.. code:: ipython3

    execution = workflow.execute(
        inputs={
            'TrainingJobName': 'regression-{}'.format(id), # Each Sagemaker Job requires a unique name,
            'GlueJobName': job_name,
            'ModelName': 'CustomerChurn-{}'.format(id), # Each Model requires a unique name,
            'EndpointName': 'CustomerChurn', # Each Endpoint requires a unique name
            'LambdaFunctionName': function_name
        }
    )

Render workflow progress with the
`render_progress <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.render_progress>`__.
This generates a snapshot of the current state of your workflow as it
executes. This is a static image therefore you must run the cell again
to check progress:

.. code:: ipython3

    execution.render_progress()

Use
`list_events <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.list_events>`__
to list all events in the workflow execution:

.. code:: ipython3

    execution.list_events(html=True)

Use
`list_executions <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.list_executions>`__
to list all executions for a specific workflow:

.. code:: ipython3

    workflow.list_executions(html=True)

Use
`list_workflows <https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.list_workflows>`__
to list all workflows in your AWS account:

.. code:: ipython3

    Workflow.list_workflows(html=True)

Clean Up
--------

When you are done, make sure to clean up your AWS account by deleting
resources you won’t be reusing. Uncomment the code below and run the
cell to delete the Glue job, Lambda function, and Step Function.

.. code:: ipython3

    #lambda_client.delete_function(FunctionName=function_name)
    #glue_client.delete_job(JobName=job_name)
    #workflow.delete()

--------------
