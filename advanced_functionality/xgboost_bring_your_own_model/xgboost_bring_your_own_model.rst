Amazon SageMaker XGBoost Bring Your Own Model
=============================================

**Hosting a Pre-Trained scikit-learn Model in Amazon SageMaker XGBoost
Algorithm Container**

--------------

--------------

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Optionally, train a scikit learn XGBoost
   model <#Optionally,-train-a-scikit-learn-XGBoost-model>`__
4. `Upload the pre-trained model to
   S3 <#Upload-the-pre-trained-model-to-S3>`__
5. `Set up hosting for the model <#Set-up-hosting-for-the-model>`__
6. `Validate the model for use <#Validate-the-model-for-use>`__

Setup
-----

Let’s start by specifying:

-  AWS region.
-  The IAM role arn used to give learning and hosting access to your
   data. See the documentation for how to specify these.
-  The S3 bucket that you want to use for training and model data.

.. code:: ipython3

    %%time
    
    import os
    import boto3
    import re
    import json
    import sagemaker
    from sagemaker import get_execution_role
    
    region = boto3.Session().region_name
    
    role = get_execution_role()
    
    bucket = sagemaker.Session().default_bucket()

.. code:: ipython3

    prefix = 'sagemaker/DEMO-xgboost-byo'
    bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region, bucket)
    # customize to your bucket where you have stored the data

Optionally, train a scikit learn XGBoost model
----------------------------------------------

These steps are optional and are needed to generate the scikit-learn
model that will eventually be hosted using the SageMaker Algorithm
contained.

Install XGboost
~~~~~~~~~~~~~~~

Note that for conda based installation, you’ll need to change the
Notebook kernel to the environment with conda and Python3.

.. code:: ipython3

    !conda install -y -c conda-forge xgboost==0.90

Fetch the dataset
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    import pickle, gzip, numpy, urllib.request, json
    
    # Load the dataset
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

Prepare the dataset for training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    
    import struct
    import io
    import boto3
    
    def get_dataset():
      import pickle
      import gzip
      with gzip.open('mnist.pkl.gz', 'rb') as f:
          u = pickle._Unpickler(f)
          u.encoding = 'latin1'
          return u.load()

.. code:: ipython3

    train_set, valid_set, test_set = get_dataset()
    
    train_X = train_set[0]
    train_y = train_set[1]
    
    valid_X = valid_set[0]
    valid_y = valid_set[1]
    
    test_X = test_set[0]
    test_y = test_set[1]

Train the XGBClassifier
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import xgboost as xgb
    import sklearn as sk 
    
    bt = xgb.XGBClassifier(max_depth=5,
                           learning_rate=0.2,
                           n_estimators=10,
                           objective='multi:softmax')   # Setup xgboost model
    bt.fit(train_X, train_y, # Train it to our data
           eval_set=[(valid_X, valid_y)], 
           verbose=False)

Save the trained model file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that the model file name must satisfy the regular expression
pattern: ``^[a-zA-Z0-9](-*[a-zA-Z0-9])*;``. The model file also need to
tar-zipped.

.. code:: ipython3

    model_file_name = "DEMO-local-xgboost-model"
    bt._Booster.save_model(model_file_name)

.. code:: ipython3

    !tar czvf model.tar.gz $model_file_name

Upload the pre-trained model to S3
----------------------------------

.. code:: ipython3

    fObj = open("model.tar.gz", 'rb')
    key= os.path.join(prefix, model_file_name, 'model.tar.gz')
    boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fObj)

Set up hosting for the model
----------------------------

Import model into hosting
~~~~~~~~~~~~~~~~~~~~~~~~~

This involves creating a SageMaker model from the model file previously
uploaded to S3.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'xgboost', '0.90-2')

.. code:: ipython3

    %%time
    from time import gmtime, strftime
    
    model_name = model_file_name + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    model_url = 'https://s3-{}.amazonaws.com/{}/{}'.format(region,bucket,key)
    sm_client = boto3.client('sagemaker')
    
    print (model_url)
    
    primary_container = {
        'Image': container,
        'ModelDataUrl': model_url,
    }
    
    create_model_response2 = sm_client.create_model(
        ModelName = model_name,
        ExecutionRoleArn = role,
        PrimaryContainer = primary_container)
    
    print(create_model_response2['ModelArn'])

Create endpoint configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker supports configuring REST endpoints in hosting with multiple
models, e.g. for A/B testing purposes. In order to support this, you can
create an endpoint configuration, that describes the distribution of
traffic across the models, whether split, shadowed, or sampled in some
way. In addition, the endpoint configuration describes the instance type
required for model deployment.

.. code:: ipython3

    from time import gmtime, strftime
    
    endpoint_config_name = 'DEMO-XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(endpoint_config_name)
    create_endpoint_config_response = sm_client.create_endpoint_config(
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

Lastly, you create the endpoint that serves up the model, through
specifying the name and configuration defined above. The end result is
an endpoint that can be validated and incorporated into production
applications. This takes 9-11 minutes to complete.

.. code:: ipython3

    %%time
    import time
    
    endpoint_name = 'DEMO-XGBoostEndpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print(endpoint_name)
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print(create_endpoint_response['EndpointArn'])
    
    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print("Status: " + status)
    
    while status=='Creating':
        time.sleep(60)
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp['EndpointStatus']
        print("Status: " + status)
    
    print("Arn: " + resp['EndpointArn'])
    print("Status: " + status)

Validate the model for use
--------------------------

Now you can obtain the endpoint from the client library using the result
from previous operations and generate classifications from the model
using that endpoint.

.. code:: ipython3

    runtime_client = boto3.client('runtime.sagemaker')

Lets generate the prediction for a single datapoint. We’ll pick one from
the test data generated earlier.

.. code:: ipython3

    import numpy as np
    point_X = test_X[0]
    point_X = np.expand_dims(point_X, axis=0)
    point_y = test_y[0]
    np.savetxt("test_point.csv", point_X, delimiter=",")

.. code:: ipython3

    %%time
    import json
    
    
    file_name = 'test_point.csv' #customize to your test file, will be 'mnist.single.test' if use data above
    
    with open(file_name, 'r') as f:
        payload = f.read().strip()
    
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType='text/csv', 
                                       Body=payload)
    result = response['Body'].read().decode('ascii')
    print('Predicted Class Probabilities: {}.'.format(result))

Post process the output
~~~~~~~~~~~~~~~~~~~~~~~

Since the result is a string, let’s process it to determine the the
output class label.

.. code:: ipython3

    floatArr = np.array(json.loads(result))
    predictedLabel = np.argmax(floatArr)
    print('Predicted Class Label: {}.'.format(predictedLabel))
    print('Actual Class Label: {}.'.format(point_y))

(Optional) Delete the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you’re ready to be done with this notebook, please run the
delete_endpoint line in the cell below. This will remove the hosted
endpoint you created and avoid any charges from a stray instance being
left on.

.. code:: ipython3

    sm_client.delete_endpoint(EndpointName=endpoint_name)

