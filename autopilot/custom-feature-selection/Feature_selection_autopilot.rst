Bringing your own data processing code to SageMaker Autopilot
=============================================================

In a typical machine learning model building process, data scientists
are required to manually prepare the features, select the algorithm, and
optimize model parameters. It takes lots of effort and expertise.
SageMaker Autopilot (https://aws.amazon.com/sagemaker/) removes the
heavy lifting. It inspects your data set, and runs a number of
candidates to figure out the optimal combination of data preprocessing
steps, machine learning algorithms and hyperparameters. You can easily
deploy either on a real-time endpoint or for batch processing.

In some cases, customer wants to have the flexibility to bring custom
data processing code to SageMaker Autopilot. For example, customer might
have datasets with large number of independent variables. Customer would
like to have a custom feature selection step to remove irrelevant
variables first. The resulted smaller dataset is then used to launch
SageMaker Autopilot job. Customer would also like to include both the
custom processing code and models from SageMaker Autopilot for easy
deployment—either on a real-time endpoint or for batch processing. We
will demonstrate how to achieve this in this notebook.

Table of contents
~~~~~~~~~~~~~~~~~

-  `Setup <#setup>`__
-  `Generate dataset <#data_gene>`__
-  `Upload data to S3 <#upload>`__
-  `Feature Selection <#feature_selection>`__
-  `Prepare Feature Selection Script <#feature_script>`__
-  `Create SageMaker Scikit Estimator <#create_sklearn_estimator>`__
-  `Batch transform our training data <#preprocess_train_data>`__
-  `Launch SageMaker Autopilot job with the preprocessed
   data <#autopilot>`__
-  `Serial Inference Pipeline that combines feature selection and
   autopilot <#inference_pipeline>`__
-  `Set up the inference pipeline <#pipeline_setup>`__
-  `Make a request to our pipeline
   endpoint <#pipeline_inference_request>`__
-  `Delete Endpoint <#delete_endpoint>`__

Setup 
======

Let’s first create our Sagemaker session and role, and create a S3
prefix to use for the notebook example.

.. code:: ipython3

    # S3 prefix
    bucket = 'qqnl-autopilot'
    prefix = 'reuse-autopilot-blog'
    
    import sagemaker
    import os
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    
    # Get a SageMaker-compatible role used by this Notebook Instance.
    role = get_execution_role()

Generate dataset 
-----------------

We use ``sklearn.datasets.make_regression`` to generate data with 100
features. 5 of these features are informative.

.. code:: ipython3

    from sklearn.datasets import make_regression
    import pandas as pd
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_features = 100, n_samples = 1500, n_informative = 5, random_state=0)
    df_X = pd.DataFrame(X).rename(columns=lambda x: 'x_'+ str(x))
    df_y = pd.DataFrame(y).rename(columns=lambda x: 'y')
    df = pd.concat([df_X, df_y], axis=1)
    pd.options.display.max_columns = 14
    df.head()

Upload the data for training 
-----------------------------

When training large models with huge amounts of data, you’ll typically
use big data tools, like Amazon Athena, AWS Glue, or Amazon EMR, to
create your data in S3. In this notebook, we use the tools provided by
the SageMaker Python SDK to upload the data to ``S3``.

We first create a folder ``data`` to store our dataset locally. Then we
save our data as ``train.csv`` and upload it to the ``S3`` bucket
specified earlier.

.. code:: sh

    %%sh
    
    if [ ! -d ./data ]
    then
        mkdir data
    fi

.. code:: ipython3

    df.to_csv('./data/train.csv', index=False)
    
    WORK_DIRECTORY = 'data'
    
    train_input = sagemaker_session.upload_data(
        path='{}/{}'.format(WORK_DIRECTORY, 'train.csv'), 
        bucket=bucket,
        key_prefix='{}/{}'.format(prefix, 'training_data'))

Feature Selection 
==================

We use Scikit-learn on Sagemaker ``SKLearn`` Estimator with a feature
selection script as an entry point. The script is very similar to a
training script you might run outside of SageMaker, but you can access
useful properties about the training environment through various
environment variables, such as:

-  SM_MODEL_DIR: A string representing the path to the directory to
   write model artifacts to. These artifacts are uploaded to S3 for
   model hosting.
-  SM_OUTPUT_DIR: A string representing the filesystem path to write
   output artifacts to. Output artifacts may include checkpoints,
   graphs, and other files to save, not including model artifacts. These
   artifacts are compressed and uploaded to S3 to the same S3 prefix as
   the model artifacts.

A typical training script loads data from the input channels, trains a
model, and saves a model to model_dir so that it can be hosted later.

Prepare Feature Selection Script 
---------------------------------

Inside ``SKLearn`` container, ``sklearn.feature_selection`` module
contains several feature selection algorithms. We choose the following
feature selection algorithms in our training script.

-  Recursive feature elimination using
   ``sklearn.feature_selection.RFE``: the goal of recursive feature
   elimination (RFE) is to select features by recursively considering
   smaller and smaller sets of features. First, the estimator is trained
   on the initial set of features and the importance of each feature is
   obtained. Then, the least important features are pruned from current
   set of features. That procedure is recursively repeated on the pruned
   set until the desired number of features to select is eventually
   reached. We use Epsilon-Support Vector Regression
   (``sklearn.svm.SVR``) as our learning estimator for RFE.
   \* Univariate linear regression test using
   ``sklearn.feature_selection.f_regression``: Linear model for testing
   the individual effect of each of many regressors. This is done in 2
   steps. First the correlation between each regressor and the target is
   computed. Then the correction is converted to an F score then to a
   p-value. Features with low p-values are selected.
-  Select features according to the k highest scores using
   ``sklearn.feature_selection.SelectKBest``. We use mutual information
   as the score function. Mutual information between two random
   variables is a non-negative value, which measures the dependency
   between the variables. It is equal to zero if and only if two random
   variables are independent, and higher values mean higher dependency.

We stack the three feature selection algorithms into one
``sklearn.pipeline.Pipeline``. After training is done, we save model
artifacts to ``SM_MODEL_DIR``. We also saved the selected column names
for later use. The complete Python script is shown below:

**Note that the feature selection algorithms used here are for
demonstration purposes. You can update the script based on the feature
selection algorithm of your choice.**

.. code:: python

   from __future__ import print_function

   import time
   import sys
   from io import StringIO
   import os
   import shutil

   import argparse
   import csv
   import json
   import numpy as np
   import pandas as pd

   from sklearn.externals import joblib
   from sklearn.impute import SimpleImputer
   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVR
   from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFE

   from sagemaker_containers.beta.framework import (
       content_types, encoders, env, modules, transformer, worker)

   label_column = 'y'
   INPUT_FEATURES_SIZE = 100

   if __name__ == '__main__':

       parser = argparse.ArgumentParser()

       # Sagemaker specific arguments. Defaults are set in the environment variables.
       parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
       parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
       parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

       args = parser.parse_args()

       # Take the set of files and read them all into a single pandas dataframe
       input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
       if len(input_files) == 0:
           raise ValueError(('There are no files in {}.\n' +
                             'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                             'the data specification in S3 was incorrectly specified or the role specified\n' +
                             'does not have permission to access the data.').format(args.train, "train"))
       
       raw_data = [ pd.read_csv(file) for file in input_files ]
       concat_data = pd.concat(raw_data)
       
       number_of_columns_x = concat_data.shape[1]
       y_train = concat_data.iloc[:,number_of_columns_x-1].values
       X_train = concat_data.iloc[:,:number_of_columns_x-1].values
       
       '''Feature selection pipeline'''
       feature_selection_pipe = Pipeline([
                    ('svr', RFE(SVR(kernel="linear"))),# default: eliminate 50%
                    ('f_reg',SelectKBest(f_regression, k=30)),
                   ('mut_info',SelectKBest(mutual_info_regression, k=10))
                   ])
       
       
       feature_selection_pipe.fit(X_train,y_train)

       joblib.dump(feature_selection_pipe, os.path.join(args.model_dir, "model.joblib"))

       print("saved model!")
       
       
       '''Save selected feature names'''
       feature_names = concat_data.columns[:-1]
       feature_names = feature_names[feature_selection_pipe.named_steps['svr'].get_support()]
       feature_names = feature_names[feature_selection_pipe.named_steps['f_reg'].get_support()]
       feature_names = feature_names[feature_selection_pipe.named_steps['mut_info'].get_support()]
       joblib.dump(feature_names, os.path.join(args.model_dir, "selected_feature_names.joblib"))
       
       print("Selected features are: {}".format(feature_names))
       
       
   def input_fn(input_data, content_type):
       """Parse input data payload
       
       We currently only take csv input. Since we need to process both labelled
       and unlabelled data we first determine whether the label column is present
       by looking at how many columns were provided.
       """
       
       if content_type == 'text/csv':
           # Read the raw input data as CSV.
           df = pd.read_csv(StringIO(input_data))      
           return df
       else:
           raise ValueError("{} not supported by script!".format(content_type))
           

   def output_fn(prediction, accept):
       """Format prediction output
       
       The default accept/content-type between containers for serial inference is JSON.
       We also want to set the ContentType or mimetype as the same value as accept so the next
       container can read the response payload correctly.
       """
       if accept == "application/json":
           instances = []
           for row in prediction.tolist():
               instances.append({"features": row})

           json_output = {"instances": instances}

           return worker.Response(json.dumps(json_output), mimetype=accept)
       elif accept == 'text/csv':
           return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
       else:
           raise RuntimeException("{} accept type is not supported by this script.".format(accept))


   def predict_fn(input_data, model):
       """Preprocess input data
       
       We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
       so we want to use .transform().

       The output is returned in the following order:
       
           rest of features either one hot encoded or standardized
       """
       print("Input data shape at predict_fn: {}".format(input_data.shape))
       if input_data.shape[1] == INPUT_FEATURES_SIZE:
       # This is a unlabelled example, return only the features
           features = model.transform(input_data)
           return features
       
       elif input_data.shape[1] == INPUT_FEATURES_SIZE + 1:
       # Labeled data. Return label and features
           features = model.transform(input_data.iloc[:,:INPUT_FEATURES_SIZE])
           return np.insert(features, 0, input_data[label_column], axis=1)

   def model_fn(model_dir):
       """Deserialize fitted model
       """
       preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
       return preprocessor

Create SageMaker Scikit Estimator for feature selection 
--------------------------------------------------------

To run our Scikit-learn training script on SageMaker, we construct a
``sagemaker.sklearn.estimator.sklearn`` estimator, which accepts several
constructor arguments:

-  **entry_point**: The path to the Python script SageMaker runs for
   training and prediction.
-  **role**: Role ARN
-  **train_instance_type** *(optional)*: The type of SageMaker instances
   for training. **Note**: Because Scikit-learn does not natively
   support GPU training, Sagemaker Scikit-learn does not currently
   support training on GPU instance types.
-  **sagemaker_session** *(optional)*: The session used to train on
   Sagemaker.

To see the code for the SKLearn Estimator, see here:
https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/sklearn

.. code:: ipython3

    from sagemaker.sklearn.estimator import SKLearn
    
    script_path = 'sklearn_feature_selection.py'
    model_output_path = os.path.join('s3://',bucket, prefix, 'Feature_selection_model/')
    
    sklearn_preprocessor = SKLearn(
        entry_point=script_path,
        role=role,
        output_path = model_output_path,
        train_instance_type="ml.c4.xlarge",
        sagemaker_session= None)
    
    sklearn_preprocessor.fit({'train': train_input})

The trained model contains model.joblib, which is our feature selection
pipeline. In additon to that, we also saved selected features. It can be
retrived from ``model_output_path`` as show below. We retrive the
selected feature names for later use.

.. code:: ipython3

    key_prefix = os.path.join(prefix, 'Feature_selection_model', sklearn_preprocessor.latest_training_job.job_name ,'output','model.tar.gz')
    sagemaker_session.download_data(path='./', bucket=bucket, key_prefix = key_prefix)

.. code:: ipython3

    !tar xvzf model.tar.gz

.. code:: ipython3

    from sklearn.externals import joblib
    feature_list = list(joblib.load('selected_feature_names.joblib'))
    print(feature_list)

Batch transform our training data 
----------------------------------

Now that our feature selection model is properly fitted, let’s go ahead
and transform our training data. Let’s use batch transform to directly
process the raw data and store right back into s3.

.. code:: ipython3

    # Define a SKLearn Transformer from the trained SKLearn Estimator
    transformer_output = os.path.join('s3://',bucket, prefix, 'Feature_selection_output/')
    transformer=sklearn_preprocessor.transformer(
        instance_count=1, 
        instance_type='ml.m4.xlarge',
        output_path=transformer_output,
        assemble_with='Line',
        accept='text/csv')

.. code:: ipython3

    # Preprocess training input
    transformer.transform(train_input, content_type='text/csv')
    print('Waiting for transform job: ' + transformer.latest_transform_job.job_name)
    transformer.wait()
    preprocessed_train = transformer.output_path

Autopilot 
==========

First we add column names to transferred data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``selected_feature_names.joblib`` downloaded from previous step contains
the list of variables selected. For this demonstration, we download the
batch transform output file from S3 and add column name on this notebook
instance. When dealing with big dataset, you can use SageMaker
Processing or Amazon Glue to add column names.

.. code:: ipython3

    transformer_output_path =  os.path.join(transformer.output_path)
    
    key_prefix = transformer_output_path[transformer_output_path.find(bucket) + len(bucket)+1:]+'train.csv.out'
    print(transformer_output_path) 
    
    sagemaker_session.download_data(path='./', bucket=bucket, 
                                    key_prefix = key_prefix)
    df_new = pd.read_csv('train.csv.out', header=None)
    
    #first column is the target variable 
    df_new.columns= ['y'] + feature_list 

.. code:: ipython3

    df_new.to_csv('./data/train_new.csv', index=False)
    
    WORK_DIRECTORY = 'data'
    
    train_new_input = sagemaker_session.upload_data(
        path='{}/{}'.format(WORK_DIRECTORY, 'train_new.csv'), 
        bucket=bucket,
        key_prefix='{}/{}'.format(prefix, 'training_data_new'))
    
    df_new.head()

Set up and kick off autopilot job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    input_data_config = [{
          'DataSource': {
            'S3DataSource': {
              'S3DataType': 'S3Prefix',
              'S3Uri': 's3://{}/{}/training_data_new'.format(bucket,prefix)
            }
          },
          'TargetAttributeName': 'y'
        }
      ]
    
    output_data_config = {
        'S3OutputPath': 's3://{}/{}/autopilot_job_output'.format(bucket,prefix)
      }
    
    AutoML_Job_Config = {
        'CompletionCriteria': {
            #we set MaxCandidate to 50 to have shorter run time. Please adjust this for your use case. 
                'MaxCandidates': 50, 
                'MaxAutoMLJobRuntimeInSeconds': 1800
            }
      }

You can now launch the Autopilot job by calling the create_auto_ml_job
API.

.. code:: ipython3

    from time import gmtime, strftime, sleep
    import boto3
    region = boto3.Session().region_name
    
    sm = boto3.Session().client(service_name='sagemaker',region_name=region)
    timestamp_suffix = strftime('%d-%H-%M-%S', gmtime())
    
    auto_ml_job_name = 'automl-blog' + timestamp_suffix
    print('AutoMLJobName: ' + auto_ml_job_name)
    
    sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                          InputDataConfig=input_data_config,
                          OutputDataConfig=output_data_config,
                          AutoMLJobConfig = AutoML_Job_Config,
                          RoleArn=role)

Tracking SageMaker Autopilot job progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Autopilot job consists of the following high-level steps :

-  Analyzing Data, where the dataset is analyzed and Autopilot comes up
   with a list of ML pipelines that should be tried out on the dataset.
   The dataset is also split into train and validation sets.
-  Feature Engineering, where Autopilot performs feature transformation
   on individual features of the dataset as well as at an aggregate
   level.
-  Model Tuning, where the top performing pipeline is selected along
   with the optimal hyperparameters for the training algorithm (the last
   stage of the pipeline).

.. code:: ipython3

    print ('JobStatus - Secondary Status')
    print('------------------------------')
    
    
    describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
    print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
    job_run_status = describe_response['AutoMLJobStatus']
        
    while job_run_status not in ('Failed', 'Completed', 'Stopped'):
        describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
        job_run_status = describe_response['AutoMLJobStatus']
        
        print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
        sleep(30)

Results
~~~~~~~

Now use the describe_auto_ml_job API to look up the best candidate
selected by the SageMaker Autopilot job.

.. code:: ipython3

    from IPython.display import JSON
    
    best_candidate = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)['BestCandidate']
    best_candidate_name = best_candidate['CandidateName']
    
    print('\n')
    print("CandidateName: " + best_candidate_name)
    print("CandidateName Steps: " + best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName'])
    print("FinalAutoMLJobObjectiveMetricName: " + best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName'])
    print("FinalAutoMLJobObjectiveMetricValue: " + str(best_candidate['FinalAutoMLJobObjectiveMetric']['Value']))

Autopilot generates 2 containers, one for data processing, and the other
for machine learning model.

.. code:: ipython3

    best_candidate['InferenceContainers']

Serial Inference Pipeline that combines feature selection and autopilot 
========================================================================

Set up the inference pipeline 
------------------------------

Setting up a Machine Learning pipeline can be done with the Pipeline
Model. This sets up a list of models in a single endpoint; in this
example, we configure our pipeline model with the fitted Scikit-learn
inference model and Autopilot models. Deploying the model follows the
same ``deploy`` pattern in the SDK.

.. code:: ipython3

    sklearn_preprocessor.latest_training_job.describe()['HyperParameters']['sagemaker_submit_directory'][1:-1]

.. code:: ipython3

    from botocore.exceptions import ClientError
    sagemaker = boto3.client('sagemaker')
    import time
    from datetime import datetime
    time_stamp = datetime.now().strftime("%m-%d-%Y-%I-%M-%S-%p")
    
    pipeline_name = 'pipeline-blog-' + time_stamp
    pipeline_endpoint_config_name = 'pipeline-blog-endpoint-config-' + time_stamp
    pipeline_endpoint_name = 'pipeline-blog-endpoint-' + time_stamp
    
    sklearn_image = sklearn_preprocessor.image_name
    container_1_source = sklearn_preprocessor.latest_training_job.describe()['HyperParameters']['sagemaker_submit_directory'][1:-1]
    inference_containers = [
            {
                'Image': sklearn_image,
                'ModelDataUrl': sklearn_preprocessor.model_data,
                'Environment': {
                    'SAGEMAKER_SUBMIT_DIRECTORY':container_1_source,
                    'SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT': "text/csv",
                    'SAGEMAKER_PROGRAM':'sklearn_feature_selection.py'
                }
            }]
    
    inference_containers.extend(best_candidate['InferenceContainers'])
    
    response = sagemaker.create_model(
            ModelName=pipeline_name,
            Containers=inference_containers,
            ExecutionRoleArn=role)

Now that we’ve created our pipeline and let us deploy it to a hosted
endpoint.

.. code:: ipython3

    try:
        response = sagemaker.create_endpoint_config(
            EndpointConfigName=pipeline_endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'DefaultVariant',
                    'ModelName': pipeline_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m4.xlarge',
                },
            ],
        )
        print('{}\n'.format(response))
    
    except ClientError:
        print('Endpoint config already exists, continuing...')
    
    
    
    try:
        response = sagemaker.create_endpoint(
            EndpointName=pipeline_endpoint_name,
            EndpointConfigName=pipeline_endpoint_config_name,
        )
        print('{}\n'.format(response))
    
    except ClientError:
        print("Endpoint already exists, continuing...")
    
    
    # Monitor the status until completed
    endpoint_status = sagemaker.describe_endpoint(EndpointName=pipeline_endpoint_name)['EndpointStatus']
    while endpoint_status not in ('OutOfService','InService','Failed'):
        endpoint_status = sagemaker.describe_endpoint(EndpointName=pipeline_endpoint_name)['EndpointStatus']
        print(endpoint_status)
        time.sleep(30)

Make a request to our pipeline endpoint
---------------------------------------

Here we just grab the first line from the training data for
demonstration purpose. The ``ContentType`` field configures the first
container, while the ``Accept`` field configures the last container. You
can also specify each container’s ``Accept`` and ``ContentType`` values
using environment variables.

We make our request with the payload in ``'text/csv'`` format, since
that is what our script currently supports. If other formats need to be
supported, this would have to be added to the ``input_fn()`` method in
our entry point.

.. code:: ipython3

    test_data = df.iloc[0:5,:-1]
    print(test_data)

.. code:: ipython3

    from sagemaker.predictor import RealTimePredictor, csv_serializer
    from sagemaker.content_types import CONTENT_TYPE_CSV
    predictor = RealTimePredictor(
        endpoint=pipeline_endpoint_name,
        serializer=csv_serializer,
        sagemaker_session=sagemaker_session,
        content_type=CONTENT_TYPE_CSV,
        accept=CONTENT_TYPE_CSV)
    
    predictor.content_type = 'text/csv'
    predictor.predict(test_data.to_csv(sep=',', header=True, index=False)).decode('utf-8')

Delete Endpoint 
----------------

Once we are finished with the endpoint, we clean up the resources!

.. code:: ipython3

    sm_client = sagemaker_session.boto_session.client('sagemaker')
    sm_client.delete_endpoint(EndpointName=pipeline_endpoint_name)

