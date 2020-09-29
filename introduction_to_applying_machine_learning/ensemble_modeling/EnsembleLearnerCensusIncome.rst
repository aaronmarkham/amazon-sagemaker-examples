Ensemble Predictions From Multiple Models
=========================================

**Combining a Linear-Learner with XGBoost for superior predictive
peformance**

--------------

--------------

Contents
--------

1.  `Background <#Background>`__
2.  `Prepration <#Preparation>`__
3.  `Data <#Data>`__

    1. `Exploration and Transformation <#Exploration>`__

4.  `Training Xgboost model using SageMaker <#Training>`__
5.  `Hosting the model <#Hosting>`__
6.  `Evaluating the model on test samples <#Evaluation>`__
7.  `Training a second Logistic Regression model using
    SageMaker <#Linear-Model>`__
8.  `Hosting the Second model <#Hosting:Linear-Learner>`__
9.  `Evaluating the model on test
    samples <#Prediction:Linear-Learner>`__
10. `Combining the model results <#Ensemble>`__
11. `Evaluating the combined model on test
    samples <#Evaluate-Ensemble>`__
12. `Exentsions <#Extensions>`__

--------------

Background
----------

Quite often, in pratical applications of Machine-Learning on predictive
tasks, one model doesn’t suffice. Most of the prediction competitions
typically require combining forecasts from multiple sources to get an
improved forecast. By combining or averaging predictions from multiple
sources/models we typically get an improved forecast. This happens as
there is considerable uncertainty in the choice of the model and there
is no one true model in many practical applications. It is therefore
beneficial to combine predictions from different models. In the Bayesian
literature, this idea is referred as Bayesian Model Averaging
http://www.stat.colostate.edu/~jah/papers/statsci.pdf and has been shown
to work much better than just picking one model.

This notebook presents an illustrative example to predict if a person
makes over 50K a year based on information about their education,
work-experience, geneder etc.

-  Preparing your *SageMaker* notebook
-  Loading a dataset from S3 using SageMaker
-  Investigating and transforming the data so that it can be fed to
   *SageMaker* algorithms
-  Estimating a model using SageMaker’s XGBoost (eXtreme Gradient
   Boosting) algorithm
-  Hosting the model on SageMaker to make on-going predictions
-  Estimating a second model using SageMaker’s Linear Learner method
-  Combining the predictions from both the models and evluating the
   combined prediction
-  Generating final predictions on the test data set

--------------

Setup
-----

Let’s start by specifying:

-  The SageMaker role arn used to give learning and hosting access to
   your data. See the documentation for how to create these. Note, if
   more than one role is required for notebook instances, training,
   and/or hosting, please replace the boto call with a the appropriate
   full SageMaker role arn string.
-  The S3 bucket that you want to use for training and storing model
   objects.

.. code:: ipython3

    import os
    import boto3
    import time
    import re
    import sagemaker 
    role = get_execution_role()
    
    # Now let's define the S3 bucket we'll used for the remainder of this example.
    
    sess = sagemaker.Session()
    bucket=sess.default_bucket() #  enter your s3 bucket where you will copy data and model artificats
    prefix = 'sagemaker/DEMO-xgboost'  # place to upload training files within the bucket
    print(f'output data will be stored in: {bucket}')

Now let’s bring in the Python libraries that we’ll use throughout the
analysis

.. code:: ipython3

    import numpy as np                                # For matrix operations and numerical processing
    import pandas as pd                               # For munging tabular data
    import sklearn as sk                              # For access to a variety of machine learning models
    import matplotlib.pyplot as plt                   # For charts and visualizations
    from IPython.display import Image                 # For displaying images in the notebook
    from IPython.display import display               # For displaying outputs in the notebook
    from sklearn.datasets import dump_svmlight_file   # For outputting data to libsvm format for xgboost
    from time import gmtime, strftime                 # For labeling SageMaker models, endpoints, etc.
    import sys                                        # For writing outputs to notebook
    import math                                       # For ceiling function
    import json                                       # For parsing hosting output
    import io                                         # For working with stream data
    import sagemaker.amazon.common as smac            # For protobuf data format

--------------

Data
----

Let’s start by downloading publicly available Census Income dataset
avaiable at https://archive.ics.uci.edu/ml/datasets/Adult. In this
dataset we have different attributes such as age, workclass, education,
country, race etc for each person. We also have an indicator of person’s
income being more than $50K an year. The prediction task is to determine
whether a person makes over 50K a year.

-  Data comes in two separate files: adult.data and adult.test
-  The field names as well as additional information is available in the
   file adult.names

Now lets read this into a Pandas data frame and take a look.

.. code:: ipython3

    ## read the data
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header = None)
    
    ## read test data
    data_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header = None, skiprows=1)
    
    ## set column names
    data.columns = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 
    'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'IncomeGroup']
    
    data_test.columns = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 
    'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'IncomeGroup']
    
    


--------------

Exploration
-----------

Data exploration and transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In what follows we will do a basic exploration of the dataset to
understand the size of data, various fields it has, the values different
features take, distribution of target values etc.

.. code:: ipython3

    # set display options
    pd.set_option('display.max_columns', 100)     # Make sure we can see all of the columns
    pd.set_option('display.max_rows', 6)         # Keep the output on one page
    
    # disply data
    display(data)
    display(data_test)
    
    # display positive and negative counts
    display(data.iloc[:,14].value_counts())
    display(data_test.iloc[:,14].value_counts())


.. code:: ipython3

    ## Combine the two datasets to convert the categorical values to binary indicators
    data_combined = pd.concat([data, data_test])
    
    ## convert the categorical variables to binary indicators
    data_combined_bin = pd.get_dummies(data_combined, prefix=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 
                                                              'race', 'sex', 'native-country', 'IncomeGroup'], drop_first=True)
    
    # combine the income >50k indicators
    Income_50k = ((data_combined_bin.iloc[:,101]==1) | (data_combined_bin.iloc[:,102] ==1))+0;
    
    # make the income indicator as first column
    data_combined_bin = pd.concat([Income_50k, data_combined_bin.iloc[:, 0:100]], axis=1)
    
    # Post conversion to binary split the data sets separately
    data_bin = data_combined_bin.iloc[0:data.shape[0], :]
    data_test_bin = data_combined_bin.iloc[data.shape[0]:, :]
    
    # display the data sets post conversion to binary indicators
    display(data_bin)
    display(data_test_bin)
    
    # count number of positives and negatives
    display(data_bin.iloc[:,0].value_counts())
    display(data_test_bin.iloc[:,0].value_counts())


Data Description
~~~~~~~~~~~~~~~~

Let’s talk about the data. At a high level, we can see:

-  There are 15 columns and around 32K rows in the training data
-  There are 15 columns and around 16 K rows in the test data
-  IncomeGroup is the target field

**Specifics on the features:** \* 9 of the 14 features are categorical
and remaining 5 are numeric \* When we convert the catgorical features
to binary we find there are altogether 103-1 =102 features

**Target variable:** \* ``IncomeGroup_>50K``: Whether or not annual
income was more than 50K

--------------

Training
--------

As our first training algorithm we pick ``xgboost`` algorithm.
``xgboost`` is an extremely popular, open-source package for gradient
boosted trees. It is computationally powerful, fully featured, and has
been successfully used in many machine learning competitions. Let’s
start with a simple ``xgboost`` model, trained using ``SageMaker's``
managed, distributed training framework.

First we’ll need to specify training parameters. This includes: 1. The
role to use 1. Our training job name 1. The ``xgboost`` algorithm
container 1. Training instance type and count 1. S3 location for
training data 1. S3 location for output data 1. Algorithm
hyperparameters 1. Stopping conditions

Supported Training Input Format: csv, libsvm. For csv input, right now
we assume the input is separated by delimiter(automatically detect the
separator by Python’s builtin sniffer tool), without a header line and
also label is in the first column. Scoring Output Format: csv.

-  Since our data is in CSV format, we will convert our dataset to the
   way SageMaker’s XGboost supports.
-  We will keep the target field in first column and remaining features
   in the next few columns
-  We will remove the header line
-  We will also split the data into a separate training and validation
   sets
-  Store the data into our s3 bucket

Split the data into 80% training and 20% validation and save it before
calling XGboost

.. code:: ipython3

    # Split the data randomly as 80% for training and remaining 20% and save them locally
    train_list = np.random.rand(len(data_bin)) < 0.8
    data_train = data_bin[train_list]
    data_val = data_bin[~train_list]
    data_train.to_csv("formatted_train.csv", sep=',', header=False, index=False) # save training data 
    data_val.to_csv("formatted_val.csv", sep=',', header=False, index=False) # save validation data
    data_test_bin.to_csv("formatted_test.csv", sep=',', header=False,  index=False) # save test data


Upload training and validation data sets in the s3 bucket and prefix provided
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    
    train_file = 'formatted_train.csv'
    val_file = 'formatted_val.csv'
    
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/', train_file)).upload_file(train_file)
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'val/', val_file)).upload_file(val_file)


Specify images used for training and hosting SageMaker’s Xgboost algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'xgboost')

.. code:: ipython3

    import boto3
    from time import gmtime, strftime
    
    job_name = 'DEMO-xgboost-single-censusincome-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print("Training job", job_name)
    
    create_training_params = \
    {
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File"
        },
        "RoleArn": role,
        "OutputDataConfig": {
            "S3OutputPath": "s3://{}/{}/single-xgboost/".format(bucket, prefix),
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m4.4xlarge",
            "VolumeSizeInGB": 20
        },
        "TrainingJobName": job_name,
        "HyperParameters": {
            "max_depth":"5",
            "eta":"0.1",
            "gamma":"1",
            "min_child_weight":"1",
            "silent":"0",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "num_round": "20"
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 60 * 60
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri":  "s3://{}/{}/train/".format(bucket, prefix),
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "csv",
                "CompressionType": "None"
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/val/".format(bucket, prefix),
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "csv",
                "CompressionType": "None"
            }
        ]
    }
    


Now let’s kick off our training job in SageMaker’s distributed, managed
training, using the parameters we just created. Because training is
managed, we don’t have to wait for our job to finish to continue, but
for this case, let’s setup a while loop so we can monitor the status of
our training.

.. code:: ipython3

    %%time
    
    region = boto3.Session().region_name
    sm = boto3.client('sagemaker')
    
    sm.create_training_job(**create_training_params)
    
    status = sm.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print(status)
    sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    if status == 'Failed':
        message = sm.describe_training_job(TrainingJobName=job_name)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')  

We can read the training and evluation metrics from AWS cloudwatch.
train-auc: 0.916177 and validation-auc:0.906567.

--------------

Hosting
-------

Now that we’ve trained the ``xgboost`` algorithm on our data, let’s
setup a model which can later be hosted. We will: 1. Point to the
scoring container 1. Point to the model.tar.gz that came from training
1. Create the hosting model

.. code:: ipython3

    model_name=job_name + '-mdl'
    xgboost_hosting_container = {
        'Image': container,
        'ModelDataUrl': sm.describe_training_job(TrainingJobName=job_name)['ModelArtifacts']['S3ModelArtifacts'],
        'Environment': {'this': 'is'}
    }
    
    create_model_response = sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer=xgboost_hosting_container)

.. code:: ipython3

    print(create_model_response['ModelArn'])
    print(sm.describe_training_job(TrainingJobName=job_name)['ModelArtifacts']['S3ModelArtifacts'])

Once we’ve setup a model, we can configure what our hosting endpoints
should be. Here we specify: 1. EC2 instance type to use for hosting 1.
Initial number of instances 1. Our hosting model name

.. code:: ipython3

    from time import gmtime, strftime
    
    endpoint_config_name = 'DEMO-XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(endpoint_config_name)
    create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[{
            'InstanceType':'ml.m4.xlarge',
            'InitialInstanceCount':1,
            'InitialVariantWeight':1,
            'ModelName':model_name,
            'VariantName':'AllTraffic'}])
    
    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

Create endpoint
~~~~~~~~~~~~~~~

Lastly, the customer creates the endpoint that serves up the model,
through specifying the name and configuration defined above. The end
result is an endpoint that can be validated and incorporated into
production applications. This takes 9-11 minutes to complete.

.. code:: ipython3

    %%time
    import time
    
    endpoint_name = 'DEMO-XGBoostEndpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(endpoint_name)
    create_endpoint_response = sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print(create_endpoint_response['EndpointArn'])
    
    resp = sm.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print("Status: " + status)
    
    while status=='Creating':
        time.sleep(60)
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp['EndpointStatus']
        print("Status: " + status)
    
    print("Arn: " + resp['EndpointArn'])
    print("Status: " + status)

--------------

Evaluation
----------

There are many ways to compare the performance of a machine learning
model. In this example, we will generate predictions and compare the
ranking metric AUC (Area Under the ROC Curve).

.. code:: ipython3

    runtime= boto3.client('runtime.sagemaker')


.. code:: ipython3

    # Simple function to create a csv from our numpy array
    
    def np2csv(arr):
        csv = io.BytesIO()
        np.savetxt(csv, arr, delimiter=',', fmt='%g')
        return csv.getvalue().decode().rstrip()
    


.. code:: ipython3

    # Function to generate prediction through sample data
    def do_predict(data, endpoint_name, content_type):
        
        payload = np2csv(data)
        response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType=content_type, 
                                       Body=payload)
        result = response['Body'].read()
        result = result.decode("utf-8")
        result = result.split(',')
        preds = [float((num)) for num in result]
        return preds
    
    # Function to iterate through a larger data set and generate batch predictions
    def batch_predict(data, batch_size, endpoint_name, content_type):
        items = len(data)
        arrs = []
        
        for offset in range(0, items, batch_size):
            if offset+batch_size < items:
                datav = data.iloc[offset:(offset+batch_size),:].as_matrix()
                results = do_predict(datav, endpoint_name, content_type)
                arrs.extend(results)
            else:
                datav = data.iloc[offset:items,:].as_matrix()
                arrs.extend(do_predict(datav, endpoint_name, content_type))
            sys.stdout.write('.')
        return(arrs)

.. code:: ipython3

    ### read the saved data for scoring
    data_train = pd.read_csv("formatted_train.csv", sep=',', header=None) 
    data_test = pd.read_csv("formatted_test.csv", sep=',', header=None) 
    data_val = pd.read_csv("formatted_val.csv", sep=',', header=None) 

Generate predictions on train, validation and test sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    
    preds_train_xgb = batch_predict(data_train.iloc[:, 1:], 1000, endpoint_name, 'text/csv')
    preds_val_xgb = batch_predict(data_val.iloc[:, 1:], 1000, endpoint_name, 'text/csv')
    preds_test_xgb = batch_predict(data_test.iloc[:, 1:], 1000, endpoint_name, 'text/csv')

Compute performance metrics on the training,validation, test data sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

compute auc/ginni
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from sklearn.metrics import roc_auc_score
    train_labels = data_train.iloc[:,0];
    val_labels = data_val.iloc[:,0];
    test_labels = data_test.iloc[:,0];
    
    print("Training AUC", roc_auc_score(train_labels, preds_train_xgb)) ##0.9161
    print("Validation AUC", roc_auc_score(val_labels, preds_val_xgb) )###0.9065
    print("Test AUC", roc_auc_score(test_labels, preds_test_xgb) )###0.9112


--------------

Linear-Model
------------

Train a second model using SageMaker’s Linear Learner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    prefix = 'sagemaker/DEMO-linear'  ##subfolder inside the data bucket to be used for Linear Learner
    
    data_train = pd.read_csv("formatted_train.csv", sep=',', header=None) 
    data_test = pd.read_csv("formatted_test.csv", sep=',', header=None) 
    data_val = pd.read_csv("formatted_val.csv", sep=',', header=None) 
    
    train_y = data_train.iloc[:,0].as_matrix();
    train_X = data_train.iloc[:,1:].as_matrix();
    
    val_y = data_val.iloc[:,0].as_matrix();
    val_X = data_val.iloc[:,1:].as_matrix();
    
    test_y = data_test.iloc[:,0].as_matrix();
    test_X = data_test.iloc[:,1:].as_matrix();


Now, we’ll convert the datasets to the recordIO wrapped protobuf format
used by the Amazon SageMaker algorithms and upload this data to S3.
We’ll start with training data.

Convert to protobuf format and upload the training and validation data to s3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    train_file = 'linear_train.data'
    
    f = io.BytesIO()
    smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
    f.seek(0)
    
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f)


.. code:: ipython3

    validation_file = 'linear_validation.data'
    
    f = io.BytesIO()
    smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
    f.seek(0)
    
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', train_file)).upload_fileobj(f)

--------------

Training Algorithm Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can begin to specify our linear model. Amazon SageMaker’s Linear
Learner actually fits many models in parallel, each with slightly
different hyperparameters, and then returns the one with the best fit.
This functionality is automatically enabled. We can influence this using
parameters like:

-  ``num_models`` to increase to total number of models run. The
   specified parameters will always be one of those models, but the
   algorithm also chooses models with nearby parameter values in order
   to find a solution nearby that may be more optimal. In this case,
   we’re going to use the max of 32.
-  ``loss`` which controls how we penalize mistakes in our model
   estimates. For this case, let’s use logistic loss as we are
   interested in estimating probabilities.
-  ``wd`` or ``l1`` which control regularization. Regularization can
   prevent model overfitting by preventing our estimates from becoming
   too finely tuned to the training data, which can actually hurt
   generalizability. In this case, we’ll leave these parameters as their
   default “auto” though.

Specify images used for training and hosting SageMaker’s linear-learner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'linear-learner')

.. code:: ipython3

    linear_job = 'DEMO-linear-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    
    print("Job name is:", linear_job)
    
    linear_training_params = {
        "RoleArn": role,
        "TrainingJobName": linear_job,
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File"
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.c4.2xlarge",
            "VolumeSizeInGB": 10
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                        "S3DataDistributionType": "ShardedByS3Key"
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None"
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None"
            }
    
        ],
        "OutputDataConfig": {
            "S3OutputPath": "s3://{}/{}/".format(bucket, prefix)
        },
        "HyperParameters": {
            "feature_dim": "100",
            "mini_batch_size": "100",
            "predictor_type": "binary_classifier",
            "epochs": "10",
            "num_models": "32",
            "loss": "logistic"
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 60 * 60
        }
    }

.. code:: ipython3

    print(linear_job)

Now let’s kick off our training job in SageMaker’s distributed, managed
training, using the parameters we just created. Because training is
managed, we don’t have to wait for our job to finish to continue, but
for this case, let’s setup a while loop so we can monitor the status of
our training.

.. code:: ipython3

    %%time
    
    region = boto3.Session().region_name
    sm = boto3.client('sagemaker')
    
    sm.create_training_job(**linear_training_params)
    status = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']
    print(status)
    sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)
    if status == 'Failed':
        message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')

--------------

Hosting:Linear-Learner
----------------------

Now that we’ve trained the linear algorithm on our data, let’s setup a
model which can later be hosted. We will: 1. Point to the scoring
container 1. Point to the model.tar.gz that came from training 1. Create
the hosting model

.. code:: ipython3

    
    linear_hosting_container = {
       'Image': container,
       'ModelDataUrl': sm.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
    }
    
    #174872318107.dkr.ecr.us-west-2.amazonaws.com
    
    create_model_response = sm.create_model(
        ModelName=linear_job,
        ExecutionRoleArn=role,
        PrimaryContainer=linear_hosting_container)
    
    print(create_model_response['ModelArn'])

Once we’ve setup a model, we can configure what our hosting endpoints
should be. Here we specify: 1. EC2 instance type to use for hosting 1.
Initial number of instances 1. Our hosting model name

.. code:: ipython3

    linear_endpoint_config = 'DEMO-linear-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print(linear_endpoint_config)
    create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName=linear_endpoint_config,
        ProductionVariants=[{
            'InstanceType': 'ml.m4.xlarge',
            'InitialInstanceCount': 1,
            'ModelName': linear_job,
            'VariantName': 'AllTraffic'}])
    
    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

Now that we’ve specified how our endpoint should be configured, we can
create them. This can be done in the background, but for now let’s run a
loop that updates us on the status of the endpoints so that we know when
they are ready for use.

.. code:: ipython3

    %%time
    
    linear_endpoint = 'DEMO-linear-endpoint-' + time.strftime("%Y%m%d%H%M", time.gmtime())
    print(linear_endpoint)
    create_endpoint_response = sm.create_endpoint(
        EndpointName=linear_endpoint,
        EndpointConfigName=linear_endpoint_config)
    print(create_endpoint_response['EndpointArn'])
    
    resp = sm.describe_endpoint(EndpointName=linear_endpoint)
    status = resp['EndpointStatus']
    print("Status: " + status)
    
    sm.get_waiter('endpoint_in_service').wait(EndpointName=linear_endpoint)
    
    resp = sm.describe_endpoint(EndpointName=linear_endpoint)
    status = resp['EndpointStatus']
    print("Arn: " + resp['EndpointArn'])
    print("Status: " + status)
    
    if status != 'InService':
        raise Exception('Endpoint creation did not succeed')

Prediction:Linear-Learner
~~~~~~~~~~~~~~~~~~~~~~~~~

Predict using SageMaker’s Linear Learner and evaluate the performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have our hosted endpoint, we can generate statistical
predictions from it. Let’s predict on our test dataset to understand how
accurate our model is on unseen samples using AUC metric.

.. code:: ipython3

    def np2csv(arr):
        csv = io.BytesIO()
        np.savetxt(csv, arr, delimiter=',', fmt='%g')
        return csv.getvalue().decode().rstrip()

.. code:: ipython3

    # Function to generate prediction through sample data
    def do_predict_linear(data, endpoint_name, content_type):
        
        payload = np2csv(data)
        response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType=content_type, 
                                       Body=payload)
        result = json.loads(response['Body'].read().decode())
        preds =  [r['score'] for r in result['predictions']]
    
        return preds
    
    # Function to iterate through a larger data set and generate batch predictions
    def batch_predict_linear(data, batch_size, endpoint_name, content_type):
        items = len(data)
        arrs = []
        
        for offset in range(0, items, batch_size):
            if offset+batch_size < items:
                datav = data.iloc[offset:(offset+batch_size),:].as_matrix()
                results = do_predict_linear(datav, endpoint_name, content_type)
                arrs.extend(results)
            else:
                datav = data.iloc[offset:items,:].as_matrix()
                arrs.extend(do_predict_linear(datav, endpoint_name, content_type))
            sys.stdout.write('.')
        return(arrs)

.. code:: ipython3

    ### Predict on Training Data
    preds_train_lin = batch_predict_linear(data_train.iloc[:,1:], 100, linear_endpoint , 'text/csv')


.. code:: ipython3

    ### Predict on Validation Data
    preds_val_lin = batch_predict_linear(data_val.iloc[:,1:], 100, linear_endpoint , 'text/csv')


.. code:: ipython3

    ### Predict on Test Data
    preds_test_lin = batch_predict_linear(data_test.iloc[:,1:], 100, linear_endpoint , 'text/csv')

Evaluation of Linear Learner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    print("Training AUC", roc_auc_score(train_labels, preds_train_lin)) ##0.9091
    print("Validation AUC", roc_auc_score(val_labels, preds_val_lin) )###0.8998
    print("Test AUC", roc_auc_score(test_labels, preds_test_lin) )###0.9033


Ensemble
--------

Perform simple average of the two models and evaluate on training, validaion and test sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ens_train = 0.5*np.array(preds_train_xgb) + 0.5*np.array(preds_train_lin);
    ens_val = 0.5*np.array(preds_val_xgb) + 0.5*np.array(preds_val_lin);
    ens_test = 0.5*np.array(preds_test_xgb) + 0.5*np.array(preds_test_lin);
    


Evaluate-Ensemble
-----------------

Evaluate the combined ensemble model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Print AUC of the combined model
    print("Train AUC- Xgboost", round(roc_auc_score(train_labels, preds_train_xgb),5))
    print("Train AUC- Linear", round(roc_auc_score(train_labels, preds_train_lin),5))
    print("Train AUC- Ensemble", round(roc_auc_score(train_labels, ens_train),5))
    
    print("=======================================")
    print("Validation AUC- Xgboost", round(roc_auc_score(val_labels, preds_val_xgb),5))
    print("Validation AUC- Linear", round(roc_auc_score(val_labels, preds_val_lin),5))
    print("Validation AUC- Ensemble", round(roc_auc_score(val_labels, ens_val),5))
    
    print("======================================")
    print("Test AUC- Xgboost", round(roc_auc_score(test_labels, preds_test_xgb),5)) 
    print("Test AUC- Linear", round(roc_auc_score(test_labels, preds_test_lin),5)) 
    print("Test AUC- Ensemble", round(roc_auc_score(test_labels, ens_test),5))   
    


We observe that by doing a simple average of the two predictions we get improved AUC compared either of the two models on all training, validation and test data sets.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Save Final prediction on test-data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    final = pd.concat([data_test.iloc[:,0], pd.DataFrame(ens_test)], axis=1)
    final.to_csv("Xgboost-linear-ensemble-prediction.csv", sep=',', header=False, index=False)

Run below to delete endpoints once you are done.
''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: ipython3

    sm.delete_endpoint(EndpointName=endpoint_name)
    sm.delete_endpoint(EndpointName=linear_endpoint)

--------------

Extensions
----------

This example analyzed a relatively small dataset, but utilized SageMaker
features such as, \* managed single-machine training of XGboost model \*
managed training of Linear Learner \* highly available, real-time model
hosting, \* doing a batch prediction using the hosted model \* Doing an
ensemble of Xgboost and Linear Learner

This example can be extended in several ways using SageMaker features
such as, \* Distributed training of Xgboost/Linear model \* Picking a
different model for training \* Training a separate model for peforming
the ensemble instead of a taking a simple average.
