Regression with Amazon SageMaker XGBoost algorithm
==================================================

**Single machine training for regression with Amazon SageMaker XGBoost
algorithm**

--------------

Introduction
------------

This notebook demonstrates the use of Amazon SageMaker’s implementation
of the XGBoost algorithm to train and host a regression model. We use
the `Abalone
data <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html>`__
originally from the `UCI data
repository <https://archive.ics.uci.edu/ml/datasets/abalone>`__. More
details about the original dataset can be found
`here <https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names>`__.
In the libsvm converted
`version <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html>`__,
the nominal feature (Male/Female/Infant) has been converted into a real
valued feature. Age of abalone is to be predicted from eight physical
measurements.

--------------

Setup
-----

This notebook was created and tested on an ml.m4.4xlarge notebook
instance. The kernel used was Python 3 (Data Science).

Let’s start by specifying: 1. The S3 bucket and prefix that you want to
use for training and model data. This should be within the same region
as the Notebook Instance, training, and hosting. 1. The IAM role arn
used to give training and hosting access to your data. See the
documentation for how to create these. Note, if more than one role is
required for notebook instances, training, and/or hosting, please
replace the boto regexp with a the appropriate full IAM role arn
string(s).

.. code:: ipython3

    %%time
    
    import os
    import boto3
    import re
    import sagemaker
    
    role = sagemaker.get_execution_role()
    region = boto3.Session().region_name
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket and prefix
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-xgboost-abalone-default'
    # customize to your bucket where you have stored the data
    bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region, bucket)

Fetching the dataset
~~~~~~~~~~~~~~~~~~~~

Following methods split the data into train/test/validation datasets and
upload files to S3.

.. code:: ipython3

    %%time
    
    import io
    import boto3
    import random
    
    def data_split(FILE_DATA, FILE_TRAIN, FILE_VALIDATION, FILE_TEST, PERCENT_TRAIN, PERCENT_VALIDATION, PERCENT_TEST):
        data = [l for l in open(FILE_DATA, 'r')]
        train_file = open(FILE_TRAIN, 'w')
        valid_file = open(FILE_VALIDATION, 'w')
        tests_file = open(FILE_TEST, 'w')
    
        num_of_data = len(data)
        num_train = int((PERCENT_TRAIN/100.0)*num_of_data)
        num_valid = int((PERCENT_VALIDATION/100.0)*num_of_data)
        num_tests = int((PERCENT_TEST/100.0)*num_of_data)
    
        data_fractions = [num_train, num_valid, num_tests]
        split_data = [[],[],[]]
    
        rand_data_ind = 0
    
        for split_ind, fraction in enumerate(data_fractions):
            for i in range(fraction):
                rand_data_ind = random.randint(0, len(data)-1)
                split_data[split_ind].append(data[rand_data_ind])
                data.pop(rand_data_ind)
    
        for l in split_data[0]:
            train_file.write(l)
    
        for l in split_data[1]:
            valid_file.write(l)
    
        for l in split_data[2]:
            tests_file.write(l)
    
        train_file.close()
        valid_file.close()
        tests_file.close()
    
    def write_to_s3(fobj, bucket, key):
        return boto3.Session(region_name=region).resource('s3').Bucket(bucket).Object(key).upload_fileobj(fobj)
    
    def upload_to_s3(bucket, channel, filename):
        fobj=open(filename, 'rb')
        key = prefix+'/'+channel
        url = 's3://{}/{}/{}'.format(bucket, key, filename)
        print('Writing to {}'.format(url))
        write_to_s3(fobj, bucket, key)

Data ingestion
~~~~~~~~~~~~~~

Next, we read the dataset from the existing repository into memory, for
preprocessing prior to training. This processing could be done *in situ*
by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc.,
assuming the dataset is present in the appropriate location. Then, the
next step would be to transfer the data to S3 for use in training. For
small datasets, such as this one, reading into memory isn’t onerous,
though it would be for larger datasets.

.. code:: ipython3

    %%time
    import urllib.request
    
    # Load the dataset
    FILE_DATA = 'abalone'
    urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone", FILE_DATA)
    
    #split the downloaded data into train/test/validation files
    FILE_TRAIN = 'abalone.train'
    FILE_VALIDATION = 'abalone.validation'
    FILE_TEST = 'abalone.test'
    PERCENT_TRAIN = 70
    PERCENT_VALIDATION = 15
    PERCENT_TEST = 15
    data_split(FILE_DATA, FILE_TRAIN, FILE_VALIDATION, FILE_TEST, PERCENT_TRAIN, PERCENT_VALIDATION, PERCENT_TEST)
    
    #upload the files to the S3 bucket
    upload_to_s3(bucket, 'train', FILE_TRAIN)
    upload_to_s3(bucket, 'validation', FILE_VALIDATION)
    upload_to_s3(bucket, 'test', FILE_TEST)

Training the XGBoost model
--------------------------

After setting training parameters, we kick off training, and poll for
status until training is completed, which in this example, takes between
5 and 6 minutes.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(region, 'xgboost', '0.90-1')

.. code:: ipython3

    %%time
    import boto3
    from time import gmtime, strftime
    
    job_name = 'DEMO-xgboost-regression-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print("Training job", job_name)
    
    #Ensure that the training and validation data folders generated above are reflected in the "InputDataConfig" parameter below.
    
    create_training_params = \
    {
        "AlgorithmSpecification": {
            "TrainingImage": container,
            "TrainingInputMode": "File"
        },
        "RoleArn": role,
        "OutputDataConfig": {
            "S3OutputPath": bucket_path + "/" + prefix + "/single-xgboost"
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.2xlarge",
            "VolumeSizeInGB": 5
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
            "num_round":"50"
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": bucket_path + "/" + prefix + '/train',
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "libsvm",
                "CompressionType": "None"
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": bucket_path + "/" + prefix + '/validation',
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "libsvm",
                "CompressionType": "None"
            }
        ]
    }
    
    
    client = boto3.client('sagemaker', region_name=region)
    client.create_training_job(**create_training_params)
    
    import time
    
    status = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print(status)
    while status !='Completed' and status!='Failed':
        time.sleep(60)
        status = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
        print(status)

Note that the “validation” channel has been initialized too. The
SageMaker XGBoost algorithm actually calculates RMSE and writes it to
the CloudWatch logs on the data passed to the “validation” channel.

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
the instance type required for model deployment.

.. code:: ipython3

    from time import gmtime, strftime
    
    endpoint_config_name = 'DEMO-XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(endpoint_config_name)
    create_endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[{
            'InstanceType':'ml.m5.xlarge',
            'InitialVariantWeight':1,
            'InitialInstanceCount':1,
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
    create_endpoint_response = client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print(create_endpoint_response['EndpointArn'])
    
    resp = client.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    while status=='Creating':
        print("Status: " + status)
        time.sleep(60)
        resp = client.describe_endpoint(EndpointName=endpoint_name)
        status = resp['EndpointStatus']
    
    print("Arn: " + resp['EndpointArn'])
    print("Status: " + status)

Validate the model for use
--------------------------

Finally, the customer can now validate the model for use. They can
obtain the endpoint from the client library using the result from
previous operations, and generate classifications from the trained model
using that endpoint.

.. code:: ipython3

    runtime_client = boto3.client('runtime.sagemaker', region_name=region)

Start with a single prediction.

.. code:: ipython3

    !head -1 abalone.test > abalone.single.test

.. code:: ipython3

    %%time
    import json
    from itertools import islice
    import math
    import struct
    
    file_name = 'abalone.single.test' #customize to your test file
    with open(file_name, 'r') as f:
        payload = f.read().strip()
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType='text/x-libsvm', 
                                       Body=payload)
    result = response['Body'].read()
    result = result.decode("utf-8")
    result = result.split(',')
    result = [math.ceil(float(i)) for i in result]
    label = payload.strip(' ').split()[0]
    print ('Label: ',label,'\nPrediction: ', result[0])

OK, a single prediction works. Let’s do a whole batch to see how good is
the predictions accuracy.

.. code:: ipython3

    import sys
    import math
    def do_predict(data, endpoint_name, content_type):
        payload = '\n'.join(data)
        response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType=content_type, 
                                       Body=payload)
        result = response['Body'].read()
        result = result.decode("utf-8")
        result = result.split(',')
        preds = [float((num)) for num in result]
        preds = [math.ceil(num) for num in preds]
        return preds
    
    def batch_predict(data, batch_size, endpoint_name, content_type):
        items = len(data)
        arrs = []
        
        for offset in range(0, items, batch_size):
            if offset+batch_size < items:
                results = do_predict(data[offset:(offset+batch_size)], endpoint_name, content_type)
                arrs.extend(results)
            else:
                arrs.extend(do_predict(data[offset:items], endpoint_name, content_type))
            sys.stdout.write('.')
        return(arrs)

The following helps us calculate the Median Absolute Percent Error
(MdAPE) on the batch dataset.

.. code:: ipython3

    %%time
    import json
    import numpy as np
    
    with open(FILE_TEST, 'r') as f:
        payload = f.read().strip()
    
    labels = [int(line.split(' ')[0]) for line in payload.split('\n')]
    test_data = [line for line in payload.split('\n')]
    preds = batch_predict(test_data, 100, endpoint_name, 'text/x-libsvm')
    
    print('\n Median Absolute Percent Error (MdAPE) = ', np.median(np.abs(np.array(labels) - np.array(preds)) / np.array(labels)))

Delete Endpoint
~~~~~~~~~~~~~~~

Once you are done using the endpoint, you can use the following to
delete it.

.. code:: ipython3

    client.delete_endpoint(EndpointName=endpoint_name)
