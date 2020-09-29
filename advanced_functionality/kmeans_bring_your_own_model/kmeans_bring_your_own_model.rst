Bring Your Own Model (k-means)
==============================

**Hosting a Pre-Trained Model in Amazon SageMaker Algorithm Containers**

--------------

--------------

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `(Optional) <#Optional>`__
4. `Data <#Data>`__
5. `Train Locally <#Train%20Locally>`__
6. `Convert <#Convert>`__
7. `Host <#Host>`__
8. `Confirm <#Confirm>`__
9. `Extensions <#Extensions>`__

Setup
-----

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the boto regexp with a the appropriate full
   IAM role arn string(s).

.. code:: ipython3

    # Define IAM role
    import boto3
    import re
     
    import sagemaker
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-kmeans-byom'
     


.. code:: ipython3

    import numpy as np
    import sklearn.cluster
    import pickle
    import gzip
    import urllib.request
    import json
    import mxnet as mx
    import boto3
    import time
    import io
    import os

(Optional)
----------

*This section is only included for illustration purposes. In a real use
case, you’d be bringing your model from an existing process and not need
to complete these steps.*

Data
~~~~

For simplicity, we’ll utilize the MNIST dataset. This includes roughly
70K 28 x 28 pixel images of handwritten digits from 0 to 9. More detail
can be found `here <https://en.wikipedia.org/wiki/MNIST_database>`__.

.. code:: ipython3

    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

Train Locally
~~~~~~~~~~~~~

Again for simplicity, let’s stick with the k-means algorithm.

.. code:: ipython3

    kmeans = sklearn.cluster.KMeans(n_clusters=10).fit(train_set[0])

--------------

Convert
-------

The model format that Amazon SageMaker’s k-means container expects is an
MXNet NDArray with dimensions (num_clusters, feature_dim) that contains
the cluster centroids. For our current example, the 10 centroids for the
MNIST digits are stored in a (10, 784) dim NumPy array called
``kmeans.cluster_centers_``.

*Note: model formats will differ across algorithms, but this concept is
generalizable. Documentation, or just running a toy example and
interrogating the resulting model artifact is the best way to understand
the specific model format required for different algorithms.*

Let’s: - Convert to a MXNet NDArray - Save to a file ``model_algo-1``

.. code:: ipython3

    centroids = mx.ndarray.array(kmeans.cluster_centers_)
    mx.ndarray.save('model_algo-1', [centroids])

-  tar and gzip the model array

.. code:: ipython3

    !tar czvf model.tar.gz model_algo-1

-  Load to s3

.. code:: ipython3

    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'model.tar.gz')).upload_file('model.tar.gz')

--------------

Host
----

Stary by defining our model to hosting. Amazon SageMaker Algorithm
containers are published to accounts which are unique across region, so
we’ve accounted for that here.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    kmeans_model = 'DEMO-kmeans-byom-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    
    sm = boto3.client('sagemaker')
    container = get_image_uri(boto3.Session().region_name, 'kmeans')
    
    create_model_response = sm.create_model(
        ModelName=kmeans_model,
        ExecutionRoleArn=role,
        PrimaryContainer={
            'Image': container,
            'ModelDataUrl': 's3://{}/{}/model.tar.gz'.format(bucket, prefix)})
    
    print(create_model_response['ModelArn'])

Then setup our endpoint configuration.

.. code:: ipython3

    kmeans_endpoint_config = 'DEMO-kmeans-byom-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print(kmeans_endpoint_config)
    create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName=kmeans_endpoint_config,
        ProductionVariants=[{
            'InstanceType': 'ml.m4.xlarge',
            'InitialInstanceCount': 1,
            'ModelName': kmeans_model,
            'VariantName': 'AllTraffic'}])
    
    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

Finally, initiate our endpoints.

.. code:: ipython3

    %%time
    
    kmeans_endpoint = 'DEMO-kmeans-byom-endpoint-' + time.strftime("%Y%m%d%H%M", time.gmtime())
    print(kmeans_endpoint)
    create_endpoint_response = sm.create_endpoint(
        EndpointName=kmeans_endpoint,
        EndpointConfigName=kmeans_endpoint_config)
    print(create_endpoint_response['EndpointArn'])
    
    resp = sm.describe_endpoint(EndpointName=kmeans_endpoint)
    status = resp['EndpointStatus']
    print("Status: " + status)
    
    sm.get_waiter('endpoint_in_service').wait(EndpointName=kmeans_endpoint)
    
    resp = sm.describe_endpoint(EndpointName=kmeans_endpoint)
    status = resp['EndpointStatus']
    print("Arn: " + resp['EndpointArn'])
    print("Status: " + status)
    
    if status != 'InService':
        raise Exception('Endpoint creation did not succeed')

Confirm
~~~~~~~

Let’s confirm that our model is producing the same results. We’ll take
the first 100 records from our training dataset, score them in our
hosted endpoint…

.. code:: ipython3

    def np2csv(arr):
        csv = io.BytesIO()
        np.savetxt(csv, arr, delimiter=',', fmt='%g')
        return csv.getvalue().decode().rstrip()

.. code:: ipython3

    runtime = boto3.Session().client('runtime.sagemaker')
    
    payload = np2csv(train_set[0][0:100])
    response = runtime.invoke_endpoint(EndpointName=kmeans_endpoint,
                                       ContentType='text/csv',
                                       Body=payload)
    result = json.loads(response['Body'].read().decode())
    scored_labels = np.array([r['closest_cluster'] for r in result['predictions']])

… And then compare them to the model labels from our k-means example.

.. code:: ipython3

    scored_labels == kmeans.labels_[0:100]

--------------

Extensions
----------

This notebook showed how to seed a pre-existing model in an already
built container. This functionality could be replicated with other
Amazon SageMaker Algorithms, as well as the TensorFlow and MXNet
containers. Although this is certainly an easy method to bring your own
model, it is not likely to provide the flexibility of a bringing your
own scoring container. Please refer to other example notebooks which
show how to dockerize your own training and scoring container which
could be modified appropriately to your use case.

.. code:: ipython3

    # Remove endpoint to avoid stray charges
    sm.delete_endpoint(EndpointName=kmeans_endpoint)
