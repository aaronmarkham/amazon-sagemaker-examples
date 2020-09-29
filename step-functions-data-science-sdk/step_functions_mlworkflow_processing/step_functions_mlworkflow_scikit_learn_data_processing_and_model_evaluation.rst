Build machine learning workflows with Amazon SageMaker Processing and AWS Step Functions Data Science SDK
=========================================================================================================

With Amazon SageMaker Processing, you can leverage a simplified, managed
experience to run data pre- or post-processing and model evaluation
workloads on the Amazon SageMaker platform.

A processing job downloads input from Amazon Simple Storage Service
(Amazon S3), then uploads outputs to Amazon S3 during or after the
processing job.

The Step Functions SDK is an open source library that allows data
scientists to easily create and execute machine learning workflows using
AWS Step Functions and Amazon SageMaker. For more information, please
see the following resources: \* `AWS Step
Functions <https://aws.amazon.com/step-functions/>`__ \* `AWS Step
Functions Developer
Guide <https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html>`__
\* `AWS Step Functions Data Science
SDK <https://aws-step-functions-data-science-sdk.readthedocs.io>`__

SageMaker Processing Step
`ProcessingStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/stable/sagemaker.html#stepfunctions.steps.sagemaker.ProcessingStep>`__
in AWS Step Functions Data Science SDK allows the Machine Learning
engineers to directly integrate the SageMaker Processing with the AWS
Step Functions Workflows.

This notebook describes how to use the AWS Step Functions Data Science
SDK to create a machine learning workflow using SageMaker Processing
Jobs to perform data pre-processing, train the model and evaluate the
quality of the model. The high level steps include below -

1. Run a SageMaker processing job using ``ProcessingStep`` of AWS Step
   Functions Data Science SDK to run a scikit-learn script that cleans,
   pre-processes, performs feature engineering, and splits the input
   data into train and test sets.
2. Run a training job using ``TrainingStep`` of AWS Step Functions Data
   Science SDK on the pre-processed training data to train a model
3. Run a processing job on the pre-processed test data to evaluate the
   trained model’s performance using ``ProcessingStep`` of AWS Step
   Functions Data Science SDK

The dataset used here is the `Census-Income KDD
Dataset <https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29>`__.
You select features from this dataset, clean the data, and turn the data
into features that the training algorithm can use to train a binary
classification model, and split the data into train and test sets. The
task is to predict whether rows representing census responders have an
income greater than ``$50,000``, or less than ``$50,000``. The dataset
is heavily class imbalanced, with most records being labeled as earning
less than ``$50,000``. We train the model using logistic regression.

Setup
-----

.. code:: ipython3

    # Import the latest sagemaker, stepfunctions and boto3 SDKs
    import sys
    !{sys.executable} -m pip install --upgrade pip
    !{sys.executable} -m pip install -qU awscli boto3 "sagemaker==1.71.0"
    !{sys.executable} -m pip install -qU "stepfunctions==1.1.0"
    !{sys.executable} -m pip show sagemaker stepfunctions

Import the Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import io
    import logging
    import os
    import random
    import time
    import uuid
    
    import boto3
    import stepfunctions
    from stepfunctions import steps
    from stepfunctions.inputs import ExecutionInput
    from stepfunctions.steps import (
        Chain,
        ChoiceRule,
        ModelStep,
        ProcessingStep,
        TrainingStep,
        TransformStep,
    )
    from stepfunctions.template import TrainingPipeline
    from stepfunctions.template.utils import replace_parameters_with_jsonpath
    from stepfunctions.workflow import Workflow
    
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.s3 import S3Uploader
    from sagemaker.sklearn.processing import SKLearnProcessor
    
    # SageMaker Session
    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name
    
    # SageMaker Execution Role
    # You can use sagemaker.get_execution_role() if running inside sagemaker's notebook instance
    role = get_execution_role()

Next, we’ll create fine-grained IAM roles for the Step Functions and
SageMaker. The IAM roles grant the services permissions within your AWS
environment.

Add permissions to your notebook role in IAM
--------------------------------------------

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

If you are running this notebook outside of SageMaker, the SDK will use
your configured AWS CLI configuration. For more information, see
`Configuring the AWS
CLI <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`__.

Next, let’s create an execution role in IAM for Step Functions.

Create an Execution Role for Step Functions
-------------------------------------------

Your Step Functions workflow requires an IAM role to interact with other
services in your AWS environment.

1. Go to the `IAM console <https://console.aws.amazon.com/iam/>`__.
2. Select **Roles** and then **Create role**.
3. Under **Choose the service that will use this role** select **Step
   Functions**.
4. Choose **Next** until you can enter a **Role name**.
5. Enter a name such as ``StepFunctionsWorkflowExecutionRole`` and then
   select **Create role**.

Next, attach a AWS Managed IAM policy to the role you created as per
below steps.

1. Go to the `IAM console <https://console.aws.amazon.com/iam/>`__.
2. Select **Roles**
3. Search for ``StepFunctionsWorkflowExecutionRole`` IAM Role
4. Under the **Permissions** tab, click **Attach policies** and then
   search for ``CloudWatchEventsFullAccess`` IAM Policy managed by AWS.
5. Click on ``Attach Policy``

Next, create and attach another new policy to the role you created. As a
best practice, the following steps will attach a policy that only
provides access to the specific resources and actions needed for this
solution.

1. Under the **Permissions** tab, click **Attach policies** and then
   **Create policy**.
2. Enter the following in the **JSON** tab:

.. code:: json

   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Sid": "VisualEditor0",
               "Effect": "Allow",
               "Action": [
                   "events:PutTargets",
                   "events:DescribeRule",
                   "events:PutRule"
               ],
               "Resource": [
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTuningJobsRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForECSTaskRule",
                   "arn:aws:events:*:*:rule/StepFunctionsGetEventsForBatchJobsRule"
               ]
           },
           {
               "Sid": "VisualEditor1",
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
               "Sid": "VisualEditor2",
               "Effect": "Allow",
               "Action": [
                   "batch:DescribeJobs",
                   "batch:SubmitJob",
                   "batch:TerminateJob",
                   "dynamodb:DeleteItem",
                   "dynamodb:GetItem",
                   "dynamodb:PutItem",
                   "dynamodb:UpdateItem",
                   "ecs:DescribeTasks",
                   "ecs:RunTask",
                   "ecs:StopTask",
                   "glue:BatchStopJobRun",
                   "glue:GetJobRun",
                   "glue:GetJobRuns",
                   "glue:StartJobRun",
                   "lambda:InvokeFunction",
                   "sagemaker:CreateEndpoint",
                   "sagemaker:CreateEndpointConfig",
                   "sagemaker:CreateHyperParameterTuningJob",
                   "sagemaker:CreateModel",
                   "sagemaker:CreateProcessingJob",
                   "sagemaker:CreateTrainingJob",
                   "sagemaker:CreateTransformJob",
                   "sagemaker:DeleteEndpoint",
                   "sagemaker:DeleteEndpointConfig",
                   "sagemaker:DescribeHyperParameterTuningJob",
                   "sagemaker:DescribeProcessingJob",
                   "sagemaker:DescribeTrainingJob",
                   "sagemaker:DescribeTransformJob",
                   "sagemaker:ListProcessingJobs",
                   "sagemaker:ListTags",
                   "sagemaker:StopHyperParameterTuningJob",
                   "sagemaker:StopProcessingJob",
                   "sagemaker:StopTrainingJob",
                   "sagemaker:StopTransformJob",
                   "sagemaker:UpdateEndpoint",
                   "sns:Publish",
                   "sqs:SendMessage"
               ],
               "Resource": "*"
           }
       ]
   }

3.  Replace **NOTEBOOK_ROLE_ARN** with the ARN for your notebook that
    you created in the previous step in the above Policy.
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

.. code:: ipython3

    # paste the StepFunctionsWorkflowExecutionRole ARN from above
    workflow_execution_role = "" 

Create StepFunctions Workflow execution Input schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Generate unique names for Pre-Processing Job, Training Job, and Model Evaluation Job for the Step Functions Workflow
    training_job_name = "scikit-learn-training-{}".format(
        uuid.uuid1().hex
    )  # Each Training Job requires a unique name
    preprocessing_job_name = "scikit-learn-sm-preprocessing-{}".format(
        uuid.uuid1().hex
    )  # Each Preprocessing job requires a unique name,
    evaluation_job_name = "scikit-learn-sm-evaluation-{}".format(
        uuid.uuid1().hex
    )  # Each Evaluation Job requires a unique name

.. code:: ipython3

    # SageMaker expects unique names for each job, model and endpoint.
    # If these names are not unique the execution will fail. Pass these
    # dynamically for each execution using placeholders.
    execution_input = ExecutionInput(
        schema={
            "PreprocessingJobName": str,
            "TrainingJobName": str,
            "EvaluationProcessingJobName": str,
        }
    )

Data pre-processing and feature engineering
-------------------------------------------

Before introducing the script you use for data cleaning, pre-processing,
and feature engineering, inspect the first 20 rows of the dataset. The
target is predicting the ``income`` category. The features from the
dataset you select are ``age``, ``education``, ``major industry code``,
``class of worker``, ``num persons worked for employer``,
``capital gains``, ``capital losses``, and ``dividends from stocks``.

.. code:: ipython3

    import pandas as pd
    
    input_data = "s3://sagemaker-sample-data-{}/processing/census/census-income.csv".format(
        region
    )
    df = pd.read_csv(input_data, nrows=10)
    df.head(n=10)

To run the scikit-learn preprocessing script as a processing job, create
a ``SKLearnProcessor``, which lets you run scripts inside of processing
jobs using the scikit-learn image provided.

.. code:: ipython3

    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        max_runtime_in_seconds=1200,
    )

This notebook cell writes a file ``preprocessing.py``, which contains
the pre-processing script. You can update the script, and rerun this
cell to overwrite ``preprocessing.py``. You run this as a processing job
in the next cell. In this script, you

-  Remove duplicates and rows with conflicting data
-  transform the target ``income`` column into a column containing two
   labels.
-  transform the ``age`` and ``num persons worked for employer``
   numerical columns into categorical features by binning them
-  scale the continuous ``capital gains``, ``capital losses``, and
   ``dividends from stocks`` so they’re suitable for training
-  encode the ``education``, ``major industry code``,
   ``class of worker`` so they’re suitable for training
-  split the data into training and test datasets, and saves the
   training features and labels and test features and labels.

Our training script will use the pre-processed training features and
labels to train a model, and our model evaluation script will use the
trained model and pre-processed test features and labels to evaluate the
model.

.. code:: ipython3

    %%writefile preprocessing.py
    
    import argparse
    import os
    import warnings
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.compose import make_column_transformer
    
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings(action="ignore", category=DataConversionWarning)
    
    
    columns = [
        "age",
        "education",
        "major industry code",
        "class of worker",
        "num persons worked for employer",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "income",
    ]
    class_labels = [" - 50000.", " 50000+."]
    
    
    def print_shape(df):
        negative_examples, positive_examples = np.bincount(df["income"])
        print(
            "Data shape: {}, {} positive examples, {} negative examples".format(
                df.shape, positive_examples, negative_examples
            )
        )
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
        args, _ = parser.parse_known_args()
    
        print("Received arguments {}".format(args))
    
        input_data_path = os.path.join("/opt/ml/processing/input", "census-income.csv")
    
        print("Reading input data from {}".format(input_data_path))
        df = pd.read_csv(input_data_path)
        df = pd.DataFrame(data=df, columns=columns)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.replace(class_labels, [0, 1], inplace=True)
    
        negative_examples, positive_examples = np.bincount(df["income"])
        print(
            "Data after cleaning: {}, {} positive examples, {} negative examples".format(
                df.shape, positive_examples, negative_examples
            )
        )
    
        split_ratio = args.train_test_split_ratio
        print("Splitting data into train and test sets with ratio {}".format(split_ratio))
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0
        )
    
        preprocess = make_column_transformer(
            (
                ["age", "num persons worked for employer"],
                KBinsDiscretizer(encode="onehot-dense", n_bins=10),
            ),
            (
                ["capital gains", "capital losses", "dividends from stocks"],
                StandardScaler(),
            ),
            (
                ["education", "major industry code", "class of worker"],
                OneHotEncoder(sparse=False),
            ),
        )
        print("Running preprocessing and feature engineering transformations")
        train_features = preprocess.fit_transform(X_train)
        test_features = preprocess.transform(X_test)
    
        print("Train data shape after preprocessing: {}".format(train_features.shape))
        print("Test data shape after preprocessing: {}".format(test_features.shape))
    
        train_features_output_path = os.path.join(
            "/opt/ml/processing/train", "train_features.csv"
        )
        train_labels_output_path = os.path.join(
            "/opt/ml/processing/train", "train_labels.csv"
        )
    
        test_features_output_path = os.path.join(
            "/opt/ml/processing/test", "test_features.csv"
        )
        test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")
    
        print("Saving training features to {}".format(train_features_output_path))
        pd.DataFrame(train_features).to_csv(
            train_features_output_path, header=False, index=False
        )
    
        print("Saving test features to {}".format(test_features_output_path))
        pd.DataFrame(test_features).to_csv(
            test_features_output_path, header=False, index=False
        )
    
        print("Saving training labels to {}".format(train_labels_output_path))
        y_train.to_csv(train_labels_output_path, header=False, index=False)
    
        print("Saving test labels to {}".format(test_labels_output_path))
        y_test.to_csv(test_labels_output_path, header=False, index=False)


Upload the pre processing script.

.. code:: ipython3

    PREPROCESSING_SCRIPT_LOCATION = "preprocessing.py"
    
    input_code = sagemaker_session.upload_data(
        PREPROCESSING_SCRIPT_LOCATION,
        bucket=sagemaker_session.default_bucket(),
        key_prefix="data/sklearn_processing/code",
    )

S3 Locations of processing output and training data.

.. code:: ipython3

    s3_bucket_base_uri = "{}{}".format("s3://", sagemaker_session.default_bucket())
    output_data = "{}/{}".format(s3_bucket_base_uri, "data/sklearn_processing/output")
    preprocessed_training_data = "{}/{}".format(output_data, "train_data")

Create the ``ProcessingStep``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will now create the
`ProcessingStep <https://aws-step-functions-data-science-sdk.readthedocs.io/en/stable/sagemaker.html#stepfunctions.steps.sagemaker.ProcessingStep>`__
that will launch a SageMaker Processing Job.

This step will use the SKLearnProcessor as defined in the previous steps
along with the inputs and outputs objects that are defined in the below
steps.

Create `ProcessingInputs <https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.ProcessingInput>`__ and `ProcessingOutputs <https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.ProcessingOutput>`__ objects for Inputs and Outputs respectively for the SageMaker Processing Job.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    inputs = [
        ProcessingInput(
            source=input_data, destination="/opt/ml/processing/input", input_name="input-1"
        ),
        ProcessingInput(
            source=input_code,
            destination="/opt/ml/processing/input/code",
            input_name="code",
        ),
    ]
    
    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/train",
            destination="{}/{}".format(output_data,"train_data"),
            output_name="train_data",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/test",
            destination="{}/{}".format(output_data, "test_data"),
            output_name="test_data",
        ),
    ]

Create the ``ProcessingStep``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # preprocessing_job_name = generate_job_name()
    processing_step = ProcessingStep(
        "SageMaker pre-processing step",
        processor=sklearn_processor,
        job_name=execution_input["PreprocessingJobName"],
        inputs=inputs,
        outputs=outputs,
        container_arguments=["--train-test-split-ratio", "0.2"],
        container_entrypoint=["python3", "/opt/ml/processing/input/code/preprocessing.py"],
    )

Training using the pre-processed data
-------------------------------------

We create a ``SKLearn`` instance, which we will use to run a training
job using the training script ``train.py``. This will be used to create
a ``TrainingStep`` for the workflow.

.. code:: ipython3

    from sagemaker.sklearn.estimator import SKLearn
    
    sklearn = SKLearn(entry_point="train.py", train_instance_type="ml.m5.xlarge", role=role)

The training script ``train.py`` trains a logistic regression model on
the training data, and saves the model to the ``/opt/ml/model``
directory, which Amazon SageMaker tars and uploads into a
``model.tar.gz`` file into S3 at the end of the training job.

.. code:: ipython3

    %%writefile train.py
    
    import os
    
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.externals import joblib
    
    if __name__ == "__main__":
        training_data_directory = "/opt/ml/input/data/train"
        train_features_data = os.path.join(training_data_directory, "train_features.csv")
        train_labels_data = os.path.join(training_data_directory, "train_labels.csv")
        print("Reading input data")
        X_train = pd.read_csv(train_features_data, header=None)
        y_train = pd.read_csv(train_labels_data, header=None)
    
        model = LogisticRegression(class_weight="balanced", solver="lbfgs")
        print("Training LR model")
        model.fit(X_train, y_train)
        model_output_directory = os.path.join("/opt/ml/model", "model.joblib")
        print("Saving model to {}".format(model_output_directory))
        joblib.dump(model, model_output_directory)

Create the ``TrainingStep`` for the Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    training_step = steps.TrainingStep(
        "SageMaker Training Step",
        estimator=sklearn,
        data={"train": sagemaker.s3_input(preprocessed_training_data, content_type="csv")},
        job_name=execution_input["TrainingJobName"],
        wait_for_completion=True,
    )

Model Evaluation
----------------

``evaluation.py`` is the model evaluation script. Since the script also
runs using scikit-learn as a dependency, run this using the
``SKLearnProcessor`` you created previously. This script takes the
trained model and the test dataset as input, and produces a JSON file
containing classification evaluation metrics, including precision,
recall, and F1 score for each label, and accuracy and ROC AUC for the
model.

.. code:: ipython3

    %%writefile evaluation.py
    
    import json
    import os
    import tarfile
    
    import pandas as pd
    
    from sklearn.externals import joblib
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
    
    if __name__ == "__main__":
        model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
        print("Extracting model from path: {}".format(model_path))
        with tarfile.open(model_path) as tar:
            tar.extractall(path=".")
        print("Loading model")
        model = joblib.load("model.joblib")
    
        print("Loading test input data")
        test_features_data = os.path.join("/opt/ml/processing/test", "test_features.csv")
        test_labels_data = os.path.join("/opt/ml/processing/test", "test_labels.csv")
    
        X_test = pd.read_csv(test_features_data, header=None)
        y_test = pd.read_csv(test_labels_data, header=None)
        predictions = model.predict(X_test)
    
        print("Creating classification evaluation report")
        report_dict = classification_report(y_test, predictions, output_dict=True)
        report_dict["accuracy"] = accuracy_score(y_test, predictions)
        report_dict["roc_auc"] = roc_auc_score(y_test, predictions)
    
        print("Classification report:\n{}".format(report_dict))
    
        evaluation_output_path = os.path.join(
            "/opt/ml/processing/evaluation", "evaluation.json"
        )
        print("Saving classification report to {}".format(evaluation_output_path))
    
        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))

.. code:: ipython3

    MODELEVALUATION_SCRIPT_LOCATION = "evaluation.py"
    
    input_evaluation_code = sagemaker_session.upload_data(
        MODELEVALUATION_SCRIPT_LOCATION,
        bucket=sagemaker_session.default_bucket(),
        key_prefix="data/sklearn_processing/code",
    )

Create input and output objects for Model Evaluation ProcessingStep.

.. code:: ipython3

    preprocessed_testing_data = "{}/{}".format(output_data, "test_data")
    model_data_s3_uri = "{}/{}/{}".format(
        s3_bucket_base_uri, training_job_name, "output/model.tar.gz"
    )
    output_model_evaluation_s3_uri = "{}/{}/{}".format(
        s3_bucket_base_uri, training_job_name, "evaluation"
    )
    inputs_evaluation = [
        ProcessingInput(
            source=preprocessed_testing_data,
            destination="/opt/ml/processing/test",
            input_name="input-1",
        ),
        ProcessingInput(
            source=model_data_s3_uri,
            destination="/opt/ml/processing/model",
            input_name="input-2",
        ),
        ProcessingInput(
            source=input_evaluation_code,
            destination="/opt/ml/processing/input/code",
            input_name="code",
        ),
    ]
    
    outputs_evaluation = [
        ProcessingOutput(
            source="/opt/ml/processing/evaluation",
            destination=output_model_evaluation_s3_uri,
            output_name="evaluation",
        ),
    ]

.. code:: ipython3

    model_evaluation_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        max_runtime_in_seconds=1200,
    )

.. code:: ipython3

    processing_evaluation_step = ProcessingStep(
        "SageMaker Processing Model Evaluation step",
        processor=model_evaluation_processor,
        job_name=execution_input["EvaluationProcessingJobName"],
        inputs=inputs_evaluation,
        outputs=outputs_evaluation,
        container_entrypoint=["python3", "/opt/ml/processing/input/code/evaluation.py"],
    )

Create ``Fail`` state to mark the workflow failed in case any of the
steps fail.

.. code:: ipython3

    failed_state_sagemaker_processing_failure = stepfunctions.steps.states.Fail(
        "ML Workflow failed", cause="SageMakerProcessingJobFailed"
    )

Add the Error handling in the workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will use the `Catch
Block <https://aws-step-functions-data-science-sdk.readthedocs.io/en/stable/states.html#stepfunctions.steps.states.Catch>`__
to perform error handling. If the Processing Job Step or Training Step
fails, the flow will go into failure state.

.. code:: ipython3

    catch_state_processing = stepfunctions.steps.states.Catch(
        error_equals=["States.TaskFailed"],
        next_step=failed_state_sagemaker_processing_failure,
    )
    
    processing_step.add_catch(catch_state_processing)
    processing_evaluation_step.add_catch(catch_state_processing)
    training_step.add_catch(catch_state_processing)

Create and execute the ``Workflow``
-----------------------------------

.. code:: ipython3

    workflow_graph = Chain([processing_step, training_step, processing_evaluation_step])
    branching_workflow = Workflow(
        name="SageMakerProcessingWorkflow",
        definition=workflow_graph,
        role=workflow_execution_role,
    )
    
    branching_workflow.create()
    
    # Execute workflow
    execution = branching_workflow.execute(
        inputs={
            "PreprocessingJobName": preprocessing_job_name,  # Each pre processing job (SageMaker processing job) requires a unique name,
            "TrainingJobName": training_job_name,  # Each Sagemaker Training job requires a unique name,
            "EvaluationProcessingJobName": evaluation_job_name,  # Each SageMaker processing job requires a unique name,
        }
    )
    execution_output = execution.get_output(wait=True)

.. code:: ipython3

    execution.render_progress()

Inspect the output of the Workflow execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now retrieve the file ``evaluation.json`` from Amazon S3, which contains
the evaluation report.

.. code:: ipython3

    workflow_execution_output_json = execution.get_output(wait=True)

.. code:: ipython3

    from sagemaker.s3 import S3Downloader
    import json
    
    evaluation_output_config = workflow_execution_output_json["ProcessingOutputConfig"]
    for output in evaluation_output_config["Outputs"]:
        if output["OutputName"] == "evaluation":
            evaluation_s3_uri = "{}/{}".format(output["S3Output"]["S3Uri"],"evaluation.json")
            break
    
    evaluation_output = S3Downloader.read_file(evaluation_s3_uri)
    evaluation_output_dict = json.loads(evaluation_output)
    print(json.dumps(evaluation_output_dict, sort_keys=True, indent=4))

Clean Up
--------

When you are done, make sure to clean up your AWS account by deleting
resources you won’t be reusing. Uncomment the code below and run the
cell to delete the Step Function.

.. code:: ipython3

    #branching_workflow.delete()
