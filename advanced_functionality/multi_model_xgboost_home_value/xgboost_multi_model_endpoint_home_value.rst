Amazon SageMaker Multi-Model Endpoints using XGBoost
====================================================

With `Amazon SageMaker multi-model
endpoints <https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html>`__,
customers can create an endpoint that seamlessly hosts up to thousands
of models. These endpoints are well suited to use cases where any one of
a large number of models, which can be served from a common inference
container to save inference costs, needs to be invokable on-demand and
where it is acceptable for infrequently invoked models to incur some
additional latency. For applications which require consistently low
inference latency, an endpoint deploying a single model is still the
best choice.

At a high level, Amazon SageMaker manages the loading and unloading of
models for a multi-model endpoint, as they are needed. When an
invocation request is made for a particular model, Amazon SageMaker
routes the request to an instance assigned to that model, downloads the
model artifacts from S3 onto that instance, and initiates loading of the
model into the memory of the container. As soon as the loading is
complete, Amazon SageMaker performs the requested invocation and returns
the result. If the model is already loaded in memory on the selected
instance, the downloading and loading steps are skipped and the
invocation is performed immediately.

To demonstrate how multi-model endpoints are created and used, this
notebook provides an example using a set of XGBoost models that each
predict housing prices for a single location. This domain is used as a
simple example to easily experiment with multi-model endpoints.

The Amazon SageMaker multi-model endpoint capability is designed to work
across with Mxnet, PyTorch and Scikit-Learn machine learning frameworks
(TensorFlow coming soon), SageMaker XGBoost, KNN, and Linear Learner
algorithms.

In addition, Amazon SageMaker multi-model endpoints are also designed to
work with cases where you bring your own container that integrates with
the multi-model server library. An example of this can be found
`here <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/multi_model_bring_your_own>`__
and documentation
`here. <https://docs.aws.amazon.com/sagemaker/latest/dg/build-multi-model-build-container.html>`__

Contents
~~~~~~~~

1. `Generate synthetic data for housing
   models <#Generate-synthetic-data-for-housing-models>`__
2. `Train multiple house value prediction
   models <#Train-multiple-house-value-prediction-models>`__
3. `Create the Amazon SageMaker MultiDataModel
   entity <#Create-the-Amazon-SageMaker-MultiDataModel-entity>`__
4. `Create the Multi-Model
   Endpoint <#Create-the-multi-model-endpoint>`__
5. `Deploy the Multi-Model
   Endpoint <#deploy-the-multi-model-endpoint>`__
6. `Get Predictions from the
   endpoint <#Get-predictions-from-the-endpoint>`__
7. `Additional Information <#Additional-information>`__
8. `Clean up <#Clean-up>`__

Generate synthetic data
=======================

The code below contains helper functions to generate synthetic data in
the form of a ``1x7`` numpy array representing the features of a house.

The first entry in the array is the randomly generated price of a house.
The remaining entries are the features (i.e. number of bedroom, square
feet, number of bathrooms, etc.).

These functions will be used to generate synthetic data for training,
validation, and testing. It will also allow us to submit synthetic
payloads for inference to test our multi-model endpoint.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import time

.. code:: ipython3

    NUM_HOUSES_PER_LOCATION = 1000
    LOCATIONS  = ['NewYork_NY',    'LosAngeles_CA',   'Chicago_IL',    'Houston_TX',   'Dallas_TX',
                  'Phoenix_AZ',    'Philadelphia_PA', 'SanAntonio_TX', 'SanDiego_CA',  'SanFrancisco_CA']
    PARALLEL_TRAINING_JOBS = 4 # len(LOCATIONS) if your account limits can handle it
    MAX_YEAR = 2019

.. code:: ipython3

    def gen_price(house):
        _base_price = int(house['SQUARE_FEET'] * 150)
        _price = int(_base_price + (10000 * house['NUM_BEDROOMS']) + \
                                   (15000 * house['NUM_BATHROOMS']) + \
                                   (15000 * house['LOT_ACRES']) + \
                                   (15000 * house['GARAGE_SPACES']) - \
                                   (5000 * (MAX_YEAR - house['YEAR_BUILT'])))
        return _price

.. code:: ipython3

    def gen_random_house():
        _house = {'SQUARE_FEET':   int(np.random.normal(3000, 750)),
                  'NUM_BEDROOMS':  np.random.randint(2, 7),
                  'NUM_BATHROOMS': np.random.randint(2, 7) / 2,
                  'LOT_ACRES':     round(np.random.normal(1.0, 0.25), 2),
                  'GARAGE_SPACES': np.random.randint(0, 4),
                  'YEAR_BUILT':    min(MAX_YEAR, int(np.random.normal(1995, 10)))}
        _price = gen_price(_house)
        return [_price, _house['YEAR_BUILT'],   _house['SQUARE_FEET'], 
                        _house['NUM_BEDROOMS'], _house['NUM_BATHROOMS'], 
                        _house['LOT_ACRES'],    _house['GARAGE_SPACES']]

.. code:: ipython3

    def gen_houses(num_houses):
        _house_list = []
        for i in range(num_houses):
            _house_list.append(gen_random_house())
        _df = pd.DataFrame(_house_list, 
                           columns=['PRICE',        'YEAR_BUILT',    'SQUARE_FEET',  'NUM_BEDROOMS',
                                    'NUM_BATHROOMS','LOT_ACRES',     'GARAGE_SPACES'])
        return _df

Train multiple house value prediction models
============================================

In the follow section, we are setting up the code to train a house price
prediction model for each of 4 different cities.

As such, we will launch multiple training jobs asynchronously, using the
XGBoost algorithm.

In this notebook, we will be using the AWS Managed XGBoost Image for
both training and inference - this image provides native support for
launching multi-model endpoints.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    import boto3
    
    from sklearn.model_selection import train_test_split
    
    s3 = boto3.resource('s3')
    
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
    
    BUCKET = sagemaker_session.default_bucket()
    
    # This is references the AWS managed XGBoost container
    XGBOOST_IMAGE = get_image_uri(boto3.Session().region_name, 'xgboost', repo_version='1.0-1')
    
    DATA_PREFIX = 'XGBOOST_BOSTON_HOUSING'
    MULTI_MODEL_ARTIFACTS = 'multi_model_artifacts'
    
    TRAIN_INSTANCE_TYPE = 'ml.m4.xlarge'
    ENDPOINT_INSTANCE_TYPE = 'ml.m4.xlarge'
    
    ENDPOINT_NAME = 'mme-xgboost-housing'
    
    MODEL_NAME = ENDPOINT_NAME

Split a given dataset into train, validation, and test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code below will generate 3 sets of data. 1 set to train, 1 set for
validation and 1 for testing.

.. code:: ipython3

    SEED = 7
    SPLIT_RATIOS = [0.6, 0.3, 0.1]
    
    def split_data(df):
        # split data into train and test sets
        seed      = SEED
        val_size  = SPLIT_RATIOS[1]
        test_size = SPLIT_RATIOS[2]
        
        num_samples = df.shape[0]
        X1 = df.values[:num_samples, 1:] # keep only the features, skip the target, all rows
        Y1 = df.values[:num_samples, :1] # keep only the target, all rows
    
        # Use split ratios to divide up into train/val/test
        X_train, X_val, y_train, y_val = \
            train_test_split(X1, Y1, test_size=(test_size + val_size), random_state=seed)
        # Of the remaining non-training samples, give proper ratio to validation and to test
        X_test, X_test, y_test, y_test = \
            train_test_split(X_val, y_val, test_size=(test_size / (test_size + val_size)), 
                             random_state=seed)
        # reassemble the datasets with target in first column and features after that
        _train = np.concatenate([y_train, X_train], axis=1)
        _val   = np.concatenate([y_val,   X_val],   axis=1)
        _test  = np.concatenate([y_test,  X_test],  axis=1)
    
        return _train, _val, _test

Launch a single training job for a given housing location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is nothing specific to multi-model endpoints in terms of the
models it will host. They are trained in the same way as all other
SageMaker models. Here we are using the XGBoost estimator and not
waiting for the job to complete.

.. code:: ipython3

    def launch_training_job(location):
        # clear out old versions of the data
        s3_bucket = s3.Bucket(BUCKET)
        full_input_prefix = f'{DATA_PREFIX}/model_prep/{location}'
        s3_bucket.objects.filter(Prefix=full_input_prefix + '/').delete()
    
        # upload the entire set of data for all three channels
        local_folder = f'data/{location}'
        inputs = sagemaker_session.upload_data(path=local_folder, key_prefix=full_input_prefix)
        print(f'Training data uploaded: {inputs}')
        
        _job = 'xgb-{}'.format(location.replace('_', '-'))
        full_output_prefix = f'{DATA_PREFIX}/model_artifacts/{location}'
        s3_output_path = f's3://{BUCKET}/{full_output_prefix}'
    
        
        xgb = sagemaker.estimator.Estimator(XGBOOST_IMAGE, role, 
                                            train_instance_count=1, train_instance_type=TRAIN_INSTANCE_TYPE,
                                            output_path=s3_output_path, base_job_name=_job,
                                            sagemaker_session=sagemaker_session)
        
        xgb.set_hyperparameters(max_depth=5, eta=0.2, gamma=4, min_child_weight=6, subsample=0.8, silent=0, 
                                early_stopping_rounds=5, objective='reg:linear', num_round=25) 
        
        DISTRIBUTION_MODE = 'FullyReplicated'
        
        train_input = sagemaker.s3_input(s3_data=inputs+'/train', 
                                         distribution=DISTRIBUTION_MODE, content_type='csv')
        
        val_input   = sagemaker.s3_input(s3_data=inputs+'/val', 
                                         distribution=DISTRIBUTION_MODE, content_type='csv')
        
        remote_inputs = {'train': train_input, 'validation': val_input}
    
        xgb.fit(remote_inputs, wait=False)
        
        # Return the estimator object
        return xgb

Kick off a model training job for each housing location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def save_data_locally(location, train, val, test):
        os.makedirs(f'data/{location}/train')
        np.savetxt( f'data/{location}/train/{location}_train.csv', train, delimiter=',', fmt='%.2f')
        
        os.makedirs(f'data/{location}/val')
        np.savetxt(f'data/{location}/val/{location}_val.csv', val, delimiter=',', fmt='%.2f')
        
        os.makedirs(f'data/{location}/test')
        np.savetxt(f'data/{location}/test/{location}_test.csv', test, delimiter=',', fmt='%.2f')

.. code:: ipython3

    import shutil
    import os
    
    estimators = []
    
    shutil.rmtree('data', ignore_errors=True)
    
    for loc in LOCATIONS[:PARALLEL_TRAINING_JOBS]:
        _houses = gen_houses(NUM_HOUSES_PER_LOCATION)
        _train, _val, _test = split_data(_houses)
        save_data_locally(loc, _train, _val, _test)
        estimator = launch_training_job(loc)
        estimators.append(estimator)
    
    print()
    print(f'{len(estimators)} training jobs launched: {[x.latest_training_job.job_name for x in estimators]}')

Wait for all model training to finish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def wait_for_training_job_to_complete(estimator):
        job = estimator.latest_training_job.job_name
        print(f'Waiting for job: {job}')
        status = estimator.latest_training_job.describe()['TrainingJobStatus']
        while status == 'InProgress':
            time.sleep(45)
            status = estimator.latest_training_job.describe()['TrainingJobStatus']
            if status == 'InProgress':
                print(f'{job} job status: {status}')
        print(f'DONE. Status for {job} is {status}\n')
            

.. code:: ipython3

    for est in estimators:
        wait_for_training_job_to_complete(est)

Create the multi-model endpoint with the SageMaker SDK
======================================================

Create a SageMaker Model from one of the Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    estimator = estimators[0]
    model = estimator.create_model(role=role, image=XGBOOST_IMAGE)

Create the Amazon SageMaker MultiDataModel entity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We create the multi-model endpoint using the
```MultiDataModel`` <https://sagemaker.readthedocs.io/en/stable/api/inference/multi_data_model.html>`__
class.

You can create a MultiDataModel by directly passing in a
``sagemaker.model.Model`` object - in which case, the Endpoint will
inherit information about the image to use, as well as any environmental
variables, network isolation, etc., once the MultiDataModel is deployed.

In addition, a MultiDataModel can also be created without explictly
passing a ``sagemaker.model.Model`` object. Please refer to the
documentation for additional details.

.. code:: ipython3

    from sagemaker.multidatamodel import MultiDataModel

.. code:: ipython3

    # This is where our MME will read models from on S3.
    model_data_prefix = f's3://{BUCKET}/{DATA_PREFIX}/{MULTI_MODEL_ARTIFACTS}/'

.. code:: ipython3

    mme = MultiDataModel(name=MODEL_NAME,
                         model_data_prefix=model_data_prefix,
                         model=model,# passing our model - passes container image needed for the endpoint
                         sagemaker_session=sagemaker_session)

Deploy the Multi Model Endpoint
===============================

You need to consider the appropriate instance type and number of
instances for the projected prediction workload across all the models
you plan to host behind your multi-model endpoint. The number and size
of the individual models will also drive memory requirements.

.. code:: ipython3

    predictor = mme.deploy(initial_instance_count=1,
                           instance_type=ENDPOINT_INSTANCE_TYPE,
                           endpoint_name=ENDPOINT_NAME)

Our endpoint has launched! Let’s look at what models are available to the endpoint!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By ‘available’, what we mean is, what model artfiacts are currently
stored under the S3 prefix we defined when setting up the
``MultiDataModel`` above i.e. ``model_data_prefix``.

Currently, since we have no artifacts (i.e. ``tar.gz`` files) stored
under our defined S3 prefix, our endpoint, will have no models
‘available’ to serve inference requests.

We will demonstrate how to make models ‘available’ to our endpoint
below.

.. code:: ipython3

    # No models visible!
    list(mme.list_models())

Lets deploy model artifacts to be found by the endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are now using the ``.add_model()`` method of the ``MultiDataModel``
to copy over our model artifacts from where they were initially stored,
during training, to where our endpoint will source model artifacts for
inference requests.

``model_data_source`` refers to the location of our model artifact
(i.e. where it was deposited on S3 after training completed)

``model_data_path`` is the **relative** path to the S3 prefix we
specified above (i.e. ``model_data_prefix``) where our endpoint will
source models for inference requests.

Since this is a **relative** path, we can simply pass the name of what
we wish to call the model artifact at inference time (i.e.
``Chicago_IL.tar.gz``)

Dynamically deploying additional models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also important to note, that we can always use the
``.add_model()`` method, as shown below, to dynamically deploy more
models to the endpoint, to serve up inference requests as needed.

.. code:: ipython3

    for est in estimators:
        artifact_path = est.latest_training_job.describe()['ModelArtifacts']['S3ModelArtifacts']
        model_name = artifact_path.split('/')[-4]+'.tar.gz'
        # This is copying over the model artifact to the S3 location for the MME.
        mme.add_model(model_data_source=artifact_path, model_data_path=model_name)

We have added the 4 model artifacts from our training jobs!
-----------------------------------------------------------

We can see that the S3 prefix we specified when setting up
``MultiDataModel`` now has 4 model artifacts. As such, the endpoint can
now serve up inference requests for these models.

.. code:: ipython3

    list(mme.list_models())

Get predictions from the endpoint
=================================

Recall that ``mme.deploy()`` returns a
`RealTimePredictor <https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/predictor.py#L35>`__
that we saved in a variable called ``predictor``.

We will use ``predictor`` to submit requests to the endpoint.

XGBoost supports ``text/csv`` for the content type and accept type. For
more information on XGBoost Input/Output Interface, please see
`here. <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html#InputOutput-XGBoost>`__

Since the default RealTimePredictor does not have a serializer or
deserializer set for requests, we will also set these.

This will allow us to submit a python list for inference, and get back a
float response.

.. code:: ipython3

    from sagemaker.predictor import csv_serializer, json_deserializer
    from sagemaker.content_types import CONTENT_TYPE_CSV

.. code:: ipython3

    predictor.serializer = csv_serializer
    predictor.deserializer = json_deserializer
    predictor.content_type = CONTENT_TYPE_CSV
    predictor.accept = CONTENT_TYPE_CSV

Invoking models on a multi-model endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice the higher latencies on the first invocation of any given model.
This is due to the time it takes SageMaker to download the model to the
Endpoint instance and then load the model into the inference container.
Subsequent invocations of the same model take advantage of the model
already being loaded into the inference container.

.. code:: ipython3

    start_time = time.time()
    
    predicted_value = predictor.predict(data=gen_random_house()[1:], target_model='Chicago_IL.tar.gz')
    
    duration = time.time() - start_time
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value, int(duration * 1000)))

.. code:: ipython3

    start_time = time.time()
    
    #Invoke endpoint
    predicted_value = predictor.predict(data=gen_random_house()[1:], target_model='Chicago_IL.tar.gz')
    
    duration = time.time() - start_time
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value, int(duration * 1000)))

.. code:: ipython3

    start_time = time.time()
    
    #Invoke endpoint
    predicted_value = predictor.predict(data=gen_random_house()[1:], target_model='Houston_TX.tar.gz')
    
    duration = time.time() - start_time
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value, int(duration * 1000)))

.. code:: ipython3

    start_time = time.time()
    
    #Invoke endpoint
    predicted_value = predictor.predict(data=gen_random_house()[1:], target_model='Houston_TX.tar.gz')
    
    duration = time.time() - start_time
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value, int(duration * 1000)))

Updating a model
~~~~~~~~~~~~~~~~

To update a model, you would follow the same approach as above and add
it as a new model. For example, if you have retrained the
``NewYork_NY.tar.gz`` model and wanted to start invoking it, you would
upload the updated model artifacts behind the S3 prefix with a new name
such as ``NewYork_NY_v2.tar.gz``, and then change the ``target_model``
field to invoke ``NewYork_NY_v2.tar.gz`` instead of
``NewYork_NY.tar.gz``. You do not want to overwrite the model artifacts
in Amazon S3, because the old version of the model might still be loaded
in the containers or on the storage volume of the instances on the
endpoint. Invocations to the new model could then invoke the old version
of the model.

Alternatively, you could stop the endpoint and re-deploy a fresh set of
models.

Using Boto APIs to invoke the endpoint
--------------------------------------

While developing interactively within a Jupyter notebook, since
``.deploy()`` returns a ``RealTimePredictor`` it is a more seamless
experience to start invoking your endpoint using the SageMaker SDK. You
have more fine grained control over the serialization and
deserialization protocols to shape your request and response payloads
to/from the endpoint.

This is great for iterative experimentation within a notebook.
Furthermore, should you have an application that has access to the
SageMaker SDK, you can always import ``RealTimePredictor`` and attach it
to an existing endpoint - this allows you to stick to using the high
level SDK if preferable.

Additional documentation on ``RealTimePredictor`` can be found
`here. <https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html?highlight=RealTimePredictor#sagemaker.predictor.RealTimePredictor>`__

The lower level Boto3 SDK may be preferable if you are attempting to
invoke the endpoint as a part of a broader architecture.

Imagine an API gateway frontend that uses a Lambda Proxy in order to
transform request payloads before hitting a SageMaker Endpoint - in this
example, Lambda does not have access to the SageMaker Python SDK, and as
such, Boto3 can still allow you to interact with your endpoint and serve
inference requests.

Boto3 allows for quick injection of ML intelligence via SageMaker
Endpoints into existing applications with minimal/no refactoring to
existing code.

Boto3 will submit your requests as a binary payload, while still
allowing you to supply your desired ``Content-Type`` and ``Accept``
headers with serialization being handled by the inference container in
the SageMaker Endpoint.

Additional documentation on ``.invoke_endpoint()`` can be found
`here. <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html>`__

.. code:: ipython3

    import boto3
    import json
    
    runtime_sm_client = boto3.client(service_name='sagemaker-runtime')
    
    def predict_one_house_value(features, model_name):
        print(f'Using model {model_name} to predict price of this house: {features}')
        
        # Notice how we alter the list into a string as the payload
        body = ','.join(map(str, features)) + '\n'
        
        start_time = time.time()
    
        response = runtime_sm_client.invoke_endpoint(
                            EndpointName=ENDPOINT_NAME,
                            ContentType='text/csv',
                            TargetModel=model_name,
                            Body=body)
        
        predicted_value = json.loads(response['Body'].read())[0]
    
        duration = time.time() - start_time
        
        print('${:,.2f}, took {:,d} ms\n'.format(predicted_value, int(duration * 1000)))

.. code:: ipython3

    predict_one_house_value(gen_random_house()[1:], 'Chicago_IL.tar.gz')

Clean up
--------

Here, to be sure we are not billed for endpoints we are no longer using,
we clean up.

.. code:: ipython3

    predictor.delete_endpoint()

.. code:: ipython3

    predictor.delete_model()
