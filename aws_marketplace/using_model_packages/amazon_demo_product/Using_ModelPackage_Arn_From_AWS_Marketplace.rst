AWS Marketplace Product Usage Demonstration - Model Packages
============================================================

Using Model Package ARN with Amazon SageMaker APIs
--------------------------------------------------

This sample notebook demonstrates two new functionalities added to
Amazon SageMaker: 1. Using a Model Package ARN for inference via Batch
Transform jobs / Live Endpoints 2. Using a Marketplace Model Package ARN
- we will use `Scikit Decision Trees - Pretrained
Model <https://aws.amazon.com/marketplace/pp/prodview-7qop4x5ahrdhe?qid=1543169069960&sr=0-2&ref_=srh_res_product_title>`__

Overall flow diagram
--------------------

Compatibility
-------------

This notebook is compatible only with `Scikit Decision Trees -
Pretrained
Model <https://aws.amazon.com/marketplace/pp/prodview-7qop4x5ahrdhe?qid=1543169069960&sr=0-2&ref_=srh_res_product_title>`__
sample model that is published to AWS Marketplace

Set up the environment
----------------------

.. code:: ipython3

    import sagemaker as sage
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    
    # S3 prefixes
    common_prefix = "DEMO-scikit-byo-iris"
    batch_inference_input_prefix = common_prefix + "/batch-inference-input-data"

Create the session
~~~~~~~~~~~~~~~~~~

The session remembers our connection parameters to Amazon SageMaker.
We’ll use it to perform all of our Amazon SageMaker operations.

.. code:: ipython3

    sagemaker_session = sage.Session()

Create Model
------------

Now we use the above Model Package to create a model

.. code:: ipython3

    from src.scikit_product_arns import ScikitArnProvider
    
    modelpackage_arn = ScikitArnProvider.get_model_package_arn(sagemaker_session.boto_region_name)
    print("Using model package arn " + modelpackage_arn)

.. code:: ipython3

    from sagemaker import ModelPackage
    from sagemaker.predictor import csv_serializer
    
    def predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session, serializer=csv_serializer)
    
    model = ModelPackage(role=role,
                         model_package_arn=modelpackage_arn,
                         sagemaker_session=sagemaker_session,
                         predictor_cls=predict_wrapper)

Batch Transform Job
-------------------

Now let’s use the model built to run a batch inference job and verify it
works.

Batch Transform Input Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippet below is removing the “label” column (column indexed at 0)
and retaining the rest to be batch transform’s input.

NOTE: This is the same training data, which is a no-no from a
statistical/ML science perspective. But the aim of this notebook is to
demonstrate how things work end-to-end.

.. code:: ipython3

    import pandas as pd
    
    ## Remove first column that contains the label
    shape=pd.read_csv("data/training/iris.csv", header=None).drop([0], axis=1)
    
    TRANSFORM_WORKDIR = "data/transform"
    shape.to_csv(TRANSFORM_WORKDIR + "/batchtransform_test.csv", index=False, header=False)
    
    transform_input = sagemaker_session.upload_data(TRANSFORM_WORKDIR, key_prefix=batch_inference_input_prefix) + "/batchtransform_test.csv"
    print("Transform input uploaded to " + transform_input)

.. code:: ipython3

    import json 
    import uuid
    
    transformer = model.transformer(1, 'ml.m4.xlarge')
    transformer.transform(transform_input, content_type='text/csv')
    transformer.wait()
    
    print("Batch Transform output saved to " + transformer.output_path)

Inspect the Batch Transform Output in S3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from urllib.parse import urlparse
    
    parsed_url = urlparse(transformer.output_path)
    bucket_name = parsed_url.netloc
    file_key = '{}/{}.out'.format(parsed_url.path[1:], "batchtransform_test.csv")
    
    s3_client = sagemaker_session.boto_session.client('s3')
    
    response = s3_client.get_object(Bucket = sagemaker_session.default_bucket(), Key = file_key)
    response_bytes = response['Body'].read().decode('utf-8')
    print(response_bytes)

Live Inference Endpoint
-----------------------

Now we demonstrate the creation of an endpoint for live inference

.. code:: ipython3

    predictor = model.deploy(1, 'ml.m4.xlarge')

Choose some data and use it for a prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to do some predictions, we’ll extract some of the data we used
for training and do predictions against it. This is, of course, bad
statistical practice, but a good way to see how the mechanism works.

.. code:: ipython3

    TRAINING_WORKDIR = "data/training"
    shape=pd.read_csv(TRAINING_WORKDIR + "/iris.csv", header=None)
    
    import itertools
    
    a = [50*i for i in range(3)]
    b = [40+i for i in range(10)]
    indices = [i+j for i,j in itertools.product(a,b)]
    
    test_data=shape.iloc[indices[:-1]]
    test_X=test_data.iloc[:,1:]
    test_y=test_data.iloc[:,0]

.. code:: ipython3

    print(predictor.predict(test_X.values).decode('utf-8'))

Cleanup endpoint
~~~~~~~~~~~~~~~~

.. code:: ipython3

    predictor.delete_endpoint()
