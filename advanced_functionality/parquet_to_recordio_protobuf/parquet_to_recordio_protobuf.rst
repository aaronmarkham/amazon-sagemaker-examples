Converting the Parquet data format to recordIO-wrapped protobuf
===============================================================

--------------

--------------

Contents
--------

1. `Introduction <#Introduction>`__
2. `Optional data ingestion <#Optional-data-ingestion>`__

   1. `Download the data <#Download-the-data>`__
   2. `Convert into Parquet format <#Convert-into-Parquet-format>`__

3. `Data conversion <#Data-conversion>`__

   1. `Convert to recordIO protobuf
      format <#Convert-to-recordIO-protobuf-format>`__
   2. `Upload to S3 <#Upload-to-S3>`__

4. `Training the linear model <#Training-the-linear-model>`__

Introduction
------------

In this notebook we illustrate how to convert a Parquet data format into
the recordIO-protobuf format that many SageMaker algorithms consume. For
the demonstration, first we’ll convert the publicly available MNIST
dataset into the Parquet format. Subsequently, it is converted into the
recordIO-protobuf format and uploaded to S3 for consumption by the
linear learner algorithm.

.. code:: ipython3

    import os
    import io
    import re
    import boto3
    import pandas as pd
    import numpy as np
    import time
    import sagemaker
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    
    sagemaker_session = sagemaker.Session()
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-parquet'

.. code:: ipython3

    !conda install -y -c conda-forge fastparquet scikit-learn

Optional data ingestion
-----------------------

Download the data
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    import pickle, gzip, numpy, urllib.request, json
    
    # Load the dataset
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

.. code:: ipython3

    from fastparquet import write
    from fastparquet import ParquetFile
    
    def save_as_parquet_file(dataset, filename, label_col):
        X = dataset[0]
        y = dataset[1]
        data = pd.DataFrame(X)
        data[label_col] = y
        data.columns = data.columns.astype(str) #Parquet expexts the column names to be strings
        write(filename, data)
        
    def read_parquet_file(filename):
        pf = ParquetFile(filename)
        return pf.to_pandas()
    
    def features_and_target(df, label_col):
        X = df.loc[:, df.columns != label_col].values
        y = df[label_col].values
        return [X, y]

Convert into Parquet format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    trainFile = 'train.parquet'
    validFile = 'valid.parquet'
    testFile = 'test.parquet'
    label_col = 'target'
    
    save_as_parquet_file(train_set, trainFile, label_col)
    save_as_parquet_file(valid_set, validFile, label_col)
    save_as_parquet_file(test_set, testFile, label_col)

Data conversion
---------------

Since algorithms have particular input and output requirements,
converting the dataset is also part of the process that a data scientist
goes through prior to initiating training. E.g., the Amazon SageMaker
implementation of Linear Learner takes recordIO-wrapped protobuf. Most
of the conversion effort is handled by the Amazon SageMaker Python SDK,
imported as ``sagemaker`` below.

.. code:: ipython3

    dfTrain = read_parquet_file(trainFile)
    dfValid = read_parquet_file(validFile)
    dfTest = read_parquet_file(testFile)
    
    train_X, train_y = features_and_target(dfTrain, label_col)
    valid_X, valid_y = features_and_target(dfValid, label_col)
    test_X, test_y = features_and_target(dfTest, label_col)

Convert to recordIO protobuf format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import io
    import numpy as np
    import sagemaker.amazon.common as smac
    
    trainVectors = np.array([t.tolist() for t in train_X]).astype('float32')
    trainLabels = np.where(np.array([t.tolist() for t in train_y]) == 0, 1, 0).astype('float32')
    
    bufTrain = io.BytesIO()
    smac.write_numpy_to_dense_tensor(bufTrain, trainVectors, trainLabels)
    bufTrain.seek(0)
    
    
    validVectors = np.array([t.tolist() for t in valid_X]).astype('float32')
    validLabels = np.where(np.array([t.tolist() for t in valid_y]) == 0, 1, 0).astype('float32')
    
    bufValid = io.BytesIO()
    smac.write_numpy_to_dense_tensor(bufValid, validVectors, validLabels)
    bufValid.seek(0)

Upload to S3
~~~~~~~~~~~~

.. code:: ipython3

    import boto3
    import os
    
    key = 'recordio-pb-data'
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(bufTrain)
    s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
    print('uploaded training data location: {}'.format(s3_train_data))
    
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', key)).upload_fileobj(bufValid)
    s3_validation_data = 's3://{}/{}/validation/{}'.format(bucket, prefix, key)
    print('uploaded validation data location: {}'.format(s3_validation_data))

Training the linear model
-------------------------

Once we have the data preprocessed and available in the correct format
for training, the next step is to actually train the model using the
data. Since this data is relatively small, it isn’t meant to show off
the performance of the Linear Learner training algorithm, although we
have tested it on multi-terabyte datasets.

This example takes four to six minutes to complete. Majority of the time
is spent provisioning hardware and loading the algorithm container since
the dataset is small.

First, let’s specify our containers. Since we want this notebook to run
in all 4 of Amazon SageMaker’s regions, we’ll create a small lookup.
More details on algorithm containers can be found in `AWS
documentation <https://docs-aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html>`__.

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
                        "S3DataDistributionType": "FullyReplicated"
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
            "feature_dim": "784",
            "mini_batch_size": "200",
            "predictor_type": "binary_classifier",
            "epochs": "10",
            "num_models": "32",
            "loss": "absolute_loss"
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 60 * 60
        }
    }

Now let’s kick off our training job in SageMaker’s distributed, managed
training, using the parameters we just created. Because training is
managed (AWS handles spinning up and spinning down hardware), we don’t
have to wait for our job to finish to continue, but for this case, let’s
setup a while loop so we can monitor the status of our training.

.. code:: ipython3

    %%time
    
    sm = boto3.Session().client('sagemaker')
    sm.create_training_job(**linear_training_params)
    
    status = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']
    print(status)
    sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)
    if status == 'Failed':
        message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')

.. code:: ipython3

    sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']

