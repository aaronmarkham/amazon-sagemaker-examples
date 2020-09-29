SageMaker and AWS KMS–Managed Keys
==================================

**End-to-end encryption using SageMaker and KMS-Managed keys**

--------------

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Optionally, upload encrypted data files for
   training <#Optionally,-upload-encrypted-data-files-for-training>`__
4. `Training the XGBoost model <#Training-the-XGBoost-model>`__
5. `Set up hosting for the model <#Set-up-hosting-for-the-model>`__
6. `Validate the model for use <#Validate-the-model-for-use>`__
7. `Run batch prediction using batch
   transform <#Run-batch-prediction-using-batch-transform>`__

Setup
-----

Prerequisites
~~~~~~~~~~~~~

In order to successfully run this notebook, you must first:

1. Have an existing KMS key from AWS IAM console or create one (`learn
   more <http://docs.aws.amazon.com/kms/latest/developerguide/create-keys.html>`__).
2. Allow the IAM role used for SageMaker to encrypt and decrypt data
   with this key from within applications and when using AWS services
   integrated with KMS (`learn
   more <http://docs.aws.amazon.com/console/kms/key-users>`__).
3. Allow the IAM role for this notebook to create grants with this key
   (`learn
   more <https://docs.aws.amazon.com/sagemaker/latest/dg/api-permissions-reference.html>`__).

We use the ``key-id`` from the KMS key ARN
``arn:aws:kms:region:acct-id:key/key-id``.

General Setup
~~~~~~~~~~~~~

Let’s start by specifying: \* AWS region. \* The IAM role arn used to
give learning and hosting access to your data. See the documentation for
how to specify these. \* The KMS key arn that you want to use for
encryption. \* The S3 bucket that you want to use for training and model
data.

.. code:: ipython3

    %%time
    
    import os
    import io
    import boto3
    import pandas as pd
    import numpy as np
    import re
    from sagemaker import get_execution_role
    
    region = boto3.Session().region_name
    
    role = get_execution_role()
    
    kms_key_arn = '<your-kms-key-arn>'
    
    bucket='<s3-bucket>' # put your s3 bucket name here, and create s3 bucket
    prefix = 'sagemaker/DEMO-kms'
    # customize to your bucket where you have stored the data
    bucket_path = 's3://{}'.format(bucket)

Optionally, upload encrypted data files for training
----------------------------------------------------

To demonstrate SageMaker training with KMS encrypted data, we first
upload a toy dataset that has Server Side Encryption with customer
provided key.

Data ingestion
~~~~~~~~~~~~~~

We, first, read the dataset from an existing repository into memory.
This processing could be done *in situ* by Amazon Athena, Apache Spark
in Amazon EMR, Amazon Redshift, etc., assuming the dataset is present in
the appropriate location. Then, the next step would be to transfer the
data to S3 for use in training. For small datasets, such as the one used
below, reading into memory isn’t onerous, though it would be for larger
datasets.

.. code:: ipython3

    from sklearn.datasets import load_boston
    boston = load_boston()
    X = boston['data']
    y = boston['target']
    feature_names = boston['feature_names']
    data = pd.DataFrame(X, columns=feature_names)
    target = pd.DataFrame(y, columns={'MEDV'})
    data['MEDV'] = y
    local_file_name = 'boston.csv'
    data.to_csv(local_file_name, header=False, index=False)

Data preprocessing
~~~~~~~~~~~~~~~~~~

Now that we have the dataset, we need to split it into *train*,
*validation*, and *test* datasets which we can use to evaluate the
accuracy of the machine learning algorithm. We’ll also create a test
dataset file with the labels removed so it can be fed into a batch
transform job. We randomly split the dataset into 60% training, 20%
validation and 20% test. Note that SageMaker Xgboost, expects the label
column to be the first one in the datasets. So, we’ll move the median
value column (``MEDV``) from the last to the first position within the
``write_file`` method below.

.. code:: ipython3

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

.. code:: ipython3

    def write_file(X, y, fname, include_labels=True):
        feature_names = boston['feature_names']
        data = pd.DataFrame(X, columns=feature_names)
        if include_labels:
            data.insert(0, 'MEDV', y)
        data.to_csv(fname, header=False, index=False)

.. code:: ipython3

    train_file = 'train.csv'
    validation_file = 'val.csv'
    test_file = 'test.csv'
    test_no_labels_file = 'test_no_labels.csv'
    write_file(X_train, y_train, train_file)
    write_file(X_val, y_val, validation_file)
    write_file(X_test, y_test, test_file)
    write_file(X_test, y_test, test_no_labels_file, False)

Data upload to S3 with Server Side Encryption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    s3 = boto3.client('s3')
    
    data_train = open(train_file, 'rb')
    key_train = '{}/train/{}'.format(prefix,train_file)
    kms_key_id = kms_key_arn.split(':key/')[1]
    
    print("Put object...")
    s3.put_object(Bucket=bucket,
                  Key=key_train,
                  Body=data_train,
                  ServerSideEncryption='aws:kms',
                  SSEKMSKeyId=kms_key_id)
    print("Done uploading the training dataset")
    
    data_validation = open(validation_file, 'rb')
    key_validation = '{}/validation/{}'.format(prefix,validation_file)
    
    print("Put object...")
    s3.put_object(Bucket=bucket,
                  Key=key_validation,
                  Body=data_validation,
                  ServerSideEncryption='aws:kms',
                  SSEKMSKeyId=kms_key_id)
    
    print("Done uploading the validation dataset")
    
    data_test = open(test_no_labels_file, 'rb')
    key_test = '{}/test/{}'.format(prefix,test_no_labels_file)
    
    print("Put object...")
    s3.put_object(Bucket=bucket,
                  Key=key_test,
                  Body=data_test,
                  ServerSideEncryption='aws:kms',
                  SSEKMSKeyId=kms_key_id)
    
    print("Done uploading the test dataset")

Training the SageMaker XGBoost model
------------------------------------

Now that we have our data in S3, we can begin training. We’ll use Amazon
SageMaker XGboost algorithm as an example to demonstrate model training.
Note that nothing needs to be changed in the way you’d call the training
algorithm. The only requirement for training to succeed is that the IAM
role (``role``) used for S3 access has permissions to encrypt and
decrypt data with the KMS key (``kms_key_arn``). You can set these
permissions using the instructions
`here <http://docs.aws.amazon.com/kms/latest/developerguide/key-policies.html#key-policy-default-allow-users>`__.
If the permissions aren’t set, you’ll get the ``Data download failed``
error. Specify a ``VolumeKmsKeyId`` in the training job parameters to
have the volume attached to the ML compute instance encrypted using key
provided.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'xgboost')

.. code:: ipython3

    %%time
    from time import gmtime, strftime
    import time
    
    job_name = 'DEMO-xgboost-single-regression' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print("Training job", job_name)
    
    create_training_params = \
    {
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File"
        },
        "RoleArn": role,
        "OutputDataConfig": {
            "S3OutputPath": bucket_path + "/"+ prefix + "/output"
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m4.4xlarge",
            "VolumeSizeInGB": 5,
            "VolumeKmsKeyId": kms_key_arn
        },
        "TrainingJobName": job_name,
        "HyperParameters": {
            "max_depth":"5",
            "eta":"0.2",
            "gamma":"4",
            "min_child_weight":"6",
            "subsample":"0.7",
            "silent":"0",
            "objective":"reg:linear",
            "num_round":"5"
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 86400
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": bucket_path + "/"+ prefix + '/train',
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
                        "S3Uri": bucket_path + "/"+ prefix + '/validation',
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "csv",
                "CompressionType": "None"
            }
        ]
    }
    
    client = boto3.client('sagemaker')
    client.create_training_job(**create_training_params)
    
    try:
        # wait for the job to finish and report the ending status
        client.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
        training_info = client.describe_training_job(TrainingJobName=job_name)
        status = training_info['TrainingJobStatus']
        print("Training job ended with status: " + status)
    except:
        print('Training failed to start')
         # if exception is raised, that means it has failed
        message = client.describe_training_job(TrainingJobName=job_name)['FailureReason']
        print('Training failed with the following error: {}'.format(message))

Set up hosting for the model
----------------------------

In order to set up hosting, we have to import the model from training to
hosting.

Import model into hosting
~~~~~~~~~~~~~~~~~~~~~~~~~

Register the model with hosting. This allows the flexibility of
importing models trained elsewhere.

.. code:: ipython3

    %%time
    import boto3
    from time import gmtime, strftime
    
    model_name=job_name + '-model'
    print(model_name)
    
    info = client.describe_training_job(TrainingJobName=job_name)
    model_data = info['ModelArtifacts']['S3ModelArtifacts']
    print(model_data)
    
    primary_container = {
        'Image': container,
        'ModelDataUrl': model_data
    }
    
    create_model_response = client.create_model(
        ModelName = model_name,
        ExecutionRoleArn = role,
        PrimaryContainer = primary_container)
    
    print(create_model_response['ModelArn'])

Create endpoint configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker supports configuring REST endpoints in hosting with multiple
models, e.g. for A/B testing purposes. In order to support this,
customers create an endpoint configuration, that describes the
distribution of traffic across the models, whether split, shadowed, or
sampled in some way. In addition, the endpoint configuration describes
the instance type required for model deployment and the key used to
encrypt the volume attached to the endpoint instance.

.. code:: ipython3

    from time import gmtime, strftime
    
    endpoint_config_name = 'DEMO-XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(endpoint_config_name)
    create_endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        KmsKeyId = kms_key_arn,
        ProductionVariants=[{
            'InstanceType':'ml.m4.xlarge',
            'InitialVariantWeight':1,
            'InitialInstanceCount':1,
            'ModelName':model_name,
            'VariantName':'AllTraffic'}])
    
    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

Create endpoint
~~~~~~~~~~~~~~~

Create the endpoint that serves up the model, through specifying the
name and configuration defined above. The end result is an endpoint that
can be validated and incorporated into production applications. This
takes 9-11 minutes to complete.

.. code:: ipython3

    %%time
    import time
    
    endpoint_name = 'DEMO-XGBoostEndpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(endpoint_name)
    create_endpoint_response = client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print(create_endpoint_response['EndpointArn'])
    
    
    print('EndpointArn = {}'.format(create_endpoint_response['EndpointArn']))
    
    # get the status of the endpoint
    response = client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    print('EndpointStatus = {}'.format(status))
    
    
    # wait until the status has changed
    client.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
    
    
    # print the status of the endpoint
    endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
    status = endpoint_response['EndpointStatus']
    print('Endpoint creation ended with EndpointStatus = {}'.format(status))
    
    if status != 'InService':
        raise Exception('Endpoint creation failed.')

Validate the model for use
--------------------------

You can now validate the model for use. Obtain the endpoint from the
client library using the result from previous operations, and run a
single prediction on the trained model using that endpoint.

.. code:: ipython3

    runtime_client = boto3.client('runtime.sagemaker')

.. code:: ipython3

    import sys
    import math
    def do_predict(data, endpoint_name, content_type):
        response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType=content_type, 
                                       Body=data)
        result = response['Body'].read()
        result = result.decode("utf-8")
        return result
    
    # pull the first item from the test dataset
    with open('test.csv') as f:
        first_line = f.readline()
        features = first_line.split(',')[1:]
        feature_str = ','.join(features)
    
    prediction = do_predict(feature_str, endpoint_name, 'text/csv')
    print('Prediction: ' + prediction)

(Optional) Delete the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you’re ready to be done with this notebook, please run the
delete_endpoint line in the cell below. This will remove the hosted
endpoint you created and avoid any charges from a stray instance being
left on.

.. code:: ipython3

    client.delete_endpoint(EndpointName=endpoint_name)

Run batch prediction using batch transform
------------------------------------------

Create a transform job to do batch prediction using the trained model.
Similar to the training section above, the execution role assumed by
this notebook must have permissions to encrypt and decrypt data with the
KMS key (``kms_key_arn``) used for S3 server-side encryption. Similar to
training, specify a ``VolumeKmsKeyId`` so that the volume attached to
the transform instance is encrypted using the key provided.

.. code:: ipython3

    %%time
    transform_job_name = 'DEMO-xgboost-batch-prediction' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print("Transform job", transform_job_name)
    
    transform_params = \
    {
        "TransformJobName": transform_job_name,
        "ModelName": model_name,
        "TransformInput": {
            "ContentType": "text/csv",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/"+ prefix + '/test'
                }
            },
            "SplitType": "Line"
        },
        "TransformOutput": {
            "AssembleWith": "Line",
            "S3OutputPath": bucket_path + "/"+ prefix + '/predict'
        },
        "TransformResources": {
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
            "VolumeKmsKeyId": kms_key_arn
        }
    }
    
    client.create_transform_job(**transform_params)
    
    while True:
        response = client.describe_transform_job(TransformJobName=transform_job_name)
        status = response['TransformJobStatus']
        if status == 'InProgress':
            time.sleep(15)
        elif status == 'Completed':
            print("Transform job completed!")
            break
        else:
            print("Unexpected transform job status: " + status)
            break

Evaluate the batch predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following helps us calculate the Median Absolute Percent Error
(MdAPE) on the batch prediction output in S3. Note that the intent of
this example is not to produce the most accurate regressor but to
demonstrate how to handle KMS encrypted data with SageMaker.

.. code:: ipython3

    print("Downloading prediction object...")
    s3.download_file(Bucket=bucket,
                     Key=prefix + '/predict/' + test_no_labels_file + '.out',
                     Filename='./predictions.csv')
    
    preds = np.loadtxt('predictions.csv')
    print('\nMedian Absolute Percent Error (MdAPE) = ', np.median(np.abs(y_test - preds) / y_test))
