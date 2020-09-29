Amazon SageMaker Multi-Model Endpoints using Scikit Learn
=========================================================

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
notebook provides an example using a set of Scikit Learn models that
each predict housing prices for a single location. This domain is used
as a simple example to easily experiment with multi-model endpoints.

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

Generate synthetic data for housing models
------------------------------------------

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
                           columns=['PRICE', 'YEAR_BUILT',
                                    'SQUARE_FEET', 'NUM_BEDROOMS',
                                    'NUM_BATHROOMS', 'LOT_ACRES',
                                    'GARAGE_SPACES'])
        return _df

Train multiple house value prediction models
============================================

In the follow section, we are setting up the code to train a house price
prediction model for each of 4 different cities.

As such, we will launch multiple training jobs asynchronously, using the
AWS Managed container for Scikit Learn via the Sagemaker SDK using the
``SKLearn`` estimator class.

In this notebook, we will be using the AWS Managed Scikit Learn image
for both training and inference - this image provides native support for
launching multi-model endpoints.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    import boto3
    
    s3 = boto3.resource('s3')
    
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
    
    BUCKET      = sagemaker_session.default_bucket()
    TRAINING_FILE     = 'training.py'
    INFERENCE_FILE = 'inference.py'
    SOURCE_DIR = 'source_dir'
    
    DATA_PREFIX            = 'DEMO_MME_SCIKIT_V1'
    MULTI_MODEL_ARTIFACTS  = 'multi_model_artifacts'
    
    TRAIN_INSTANCE_TYPE    = 'ml.m4.xlarge'
    ENDPOINT_INSTANCE_TYPE = 'ml.m4.xlarge'
    
    ENDPOINT_NAME = 'mme-sklearn-housing-V1'
    
    MODEL_NAME = ENDPOINT_NAME

Split a given dataset into train, validation, and test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code below will generate 3 sets of data. 1 set to train, 1 set for
validation and 1 for testing.

.. code:: ipython3

    from sklearn.model_selection import train_test_split
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

Prepare training and inference scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By using the Scikit Learn estimator via the Sagemaker SDK, we can host
and train models on Amazon Sagemaker.

For training, we do the following:

1. Prepare a training script - this script will execute the training
   logic within a SageMaker managed Scikit Learn container.

2. Create a ``sagemaker.sklearn.estimator.SKLearn`` estimator

3. Call the estimators ``.fit()`` method.

For more information on using scikit learn with the Sagemaker SDK, see
the docs
`here. <https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html>`__

Below, we will create the training script called ``training.py`` that
will be located at the root of a dicrectory called ``source_dir``.

In this example, we will be training a
`RandomForestRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`__
model that will later be used for inference in predicting house prices.

**NOTE:** You would modify the script below to implement your own
training logic.

.. code:: ipython3

    !mkdir $SOURCE_DIR

.. code:: ipython3

    %%writefile $SOURCE_DIR/$TRAINING_FILE
    
    import argparse
    import os
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    if __name__ =='__main__':
    
        print('extracting arguments')
        parser = argparse.ArgumentParser()
    
        # hyperparameters sent by the client are passed as command-line arguments to the script.
        # to simplify the demo we don't use all sklearn RandomForest hyperparameters
        parser.add_argument('--n-estimators', type=int, default=10)
        parser.add_argument('--min-samples-leaf', type=int, default=3)
    
        # Data, model, and output directories
        parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
        parser.add_argument('--model-name', type=str)
    
        args, _ = parser.parse_known_args()
    
        print('reading data')
        print('model_name: {}'.format(args.model_name))
    
        train_file = os.path.join(args.train, args.model_name + '_train.csv')    
        train_df = pd.read_csv(train_file) # read in the training data
    
        val_file = os.path.join(args.validation, args.model_name + '_val.csv')
        test_df = pd.read_csv(os.path.join(val_file)) # read in the test data
    
        # Matrix representation of the data
        print('building training and testing datasets')
        X_train = train_df[train_df.columns[1:train_df.shape[1]]] 
        X_test = test_df[test_df.columns[1:test_df.shape[1]]]
        y_train = train_df[train_df.columns[0]]
        y_test = test_df[test_df.columns[0]]
    
        # fitting the model
        print('training model')
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1)
        
        model.fit(X_train, y_train)
    
        # print abs error
        print('validating model')
        abs_err = np.abs(model.predict(X_test) - y_test)
    
        # print couple perf metrics
        for q in [10, 50, 90]:
            print('AE-at-' + str(q) + 'th-percentile: '
                  + str(np.percentile(a=abs_err, q=q)))
            
        # persist model
        path = os.path.join(args.model_dir, 'model.joblib')
        joblib.dump(model, path)
        print('model persisted at ' + path)

When using multi-model endpoints with the Sagemaker managed Scikit Learn
container, we need to provide an entry point script for inference that
will **at least** load the saved model.

We will now create this script and call it ``inference.py`` and store it
at the root of a directory called ``source_dir``. This is the same
directory which contains our ``training.py`` script.

**Note:** You could place the below ``model_fn`` function within the
``training.py`` script (above the main guard) if you prefer to have a
single script.

**Note:** You would modify the script below to implement your own
inferencing logic.

Additional information on model loading and model serving for Scikit
Learn on SageMaker can be found
`here. <https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html#deploy-a-scikit-learn-model>`__

.. code:: ipython3

    %%writefile $SOURCE_DIR/$INFERENCE_FILE
    
    import os
    import joblib
    
    
    def model_fn(model_dir):
        print('loading model.joblib from: {}'.format(model_dir))
        loaded_model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        return loaded_model

Launch a single training job for a given housing location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is nothing specific to multi-model endpoints in terms of the
models it will host. They are trained in the same way as all other
SageMaker models. Here we are using the Scikit Learn estimator and not
waiting for the job to complete.

.. code:: ipython3

    from sagemaker.sklearn.estimator import SKLearn
    
    def launch_training_job(location):
        # clear out old versions of the data
        s3_bucket = s3.Bucket(BUCKET)
        full_input_prefix = f'{DATA_PREFIX}/model_prep/{location}'
        s3_bucket.objects.filter(Prefix=full_input_prefix + '/').delete()
    
        # upload the entire set of data for all three channels
        local_folder = f'data/{location}'
        inputs = sagemaker_session.upload_data(path=local_folder, 
                                                key_prefix=full_input_prefix)
        
        print(f'Training data uploaded: {inputs}')
        
        _job = 'skl-{}'.format(location.replace('_', '-'))
        full_output_prefix = f'{DATA_PREFIX}/model_artifacts/{location}'
        s3_output_path = f's3://{BUCKET}/{full_output_prefix}'
        
        code_location = f's3://{BUCKET}/{full_input_prefix}/code'
        
    
        # Add code_location argument in order to ensure that code_artifacts are stored in the same place.
        estimator = SKLearn(
            entry_point=TRAINING_FILE, # script to use for training job
            role=role,
            source_dir=SOURCE_DIR, # Location of scripts
            train_instance_count=1,
            train_instance_type=TRAIN_INSTANCE_TYPE,
            framework_version='0.23-1',# 0.23-1 is the latest version
            output_path=s3_output_path,# Where to store model artifacts
            base_job_name=_job,
            code_location=code_location,# This is where the .tar.gz of the source_dir will be stored
            metric_definitions=[
                {'Name' : 'median-AE',
                 'Regex': 'AE-at-50th-percentile: ([0-9.]+).*$'}],
            hyperparameters = {'n-estimators'    : 100,
                                'min-samples-leaf': 3,
                                'model-name'      : location})
        
        DISTRIBUTION_MODE = 'FullyReplicated'
        
        train_input = sagemaker.s3_input(s3_data=inputs+'/train', 
                                          distribution=DISTRIBUTION_MODE, content_type='csv')
        
        val_input   = sagemaker.s3_input(s3_data=inputs+'/val', 
                                          distribution=DISTRIBUTION_MODE, content_type='csv')
        
        remote_inputs = {'train': train_input, 'validation': val_input}
    
        estimator.fit(remote_inputs, wait=False)
        
        # Return the estimator object
        return estimator
        

Kick off a model training job for each housing location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def save_data_locally(location, train, val, test):
    #     _header = ','.join(COLUMNS)
        
        os.makedirs(f'data/{location}/train')
        np.savetxt(f'data/{location}/train/{location}_train.csv', train, delimiter=',', fmt='%.2f')
        
        os.makedirs(f'data/{location}/val')
        np.savetxt(f'data/{location}/val/{location}_val.csv',     val,   delimiter=',', fmt='%.2f')
        
        os.makedirs(f'data/{location}/test')
        np.savetxt(f'data/{location}/test/{location}_test.csv',   test,  delimiter=',', fmt='%.2f')
        

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
        time.sleep(2) # to avoid throttling the CreateTrainingJob API
    
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

    # wait for the jobs to finish
    for est in estimators:
        wait_for_training_job_to_complete(est)

Create the multi-model endpoint with the SageMaker SDK
======================================================

Create a SageMaker Model from one of the Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    estimator = estimators[0]
    # inference.py is the entry_point for when we deploy the model
    # Note how we do NOT specify source_dir again, this information is inherited from the estimator
    model = estimator.create_model(role=role, entry_point='inference.py')


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
                         model=model,# passing our model
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
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value[0], int(duration * 1000)))

.. code:: ipython3

    start_time = time.time()
    
    predicted_value = predictor.predict(data=gen_random_house()[1:], target_model='Chicago_IL.tar.gz')
    
    duration = time.time() - start_time
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value[0], int(duration * 1000)))

.. code:: ipython3

    start_time = time.time()
    
    predicted_value = predictor.predict(data=gen_random_house()[1:], target_model='Houston_TX.tar.gz')
    
    duration = time.time() - start_time
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value[0], int(duration * 1000)))

.. code:: ipython3

    start_time = time.time()
    
    predicted_value = predictor.predict(data=gen_random_house()[1:], target_model='Houston_TX.tar.gz')
    
    duration = time.time() - start_time
    print('${:,.2f}, took {:,d} ms\n'.format(predicted_value[0], int(duration * 1000)))

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
to/from the endpoint. This is the approach we demonstrated above where
the ``RealTimePredictor`` was stored in the variable ``predictor``.

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
    
        float_features = [float(i) for i in features]
        body = ','.join(map(str, float_features)) + '\n'
        
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
