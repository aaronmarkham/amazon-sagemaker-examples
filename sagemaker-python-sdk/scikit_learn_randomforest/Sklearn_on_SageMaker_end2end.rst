Develop, Train, Optimize and Deploy Scikit-Learn Random Forest
--------------------------------------------------------------

-  Doc https://sagemaker.readthedocs.io/en/stable/using_sklearn.html
-  SDK https://sagemaker.readthedocs.io/en/stable/sagemaker.sklearn.html
-  boto3
   https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#client

In this notebook we show how to use Amazon SageMaker to develop, train,
tune and deploy a Scikit-Learn based ML model (Random Forest). More info
on Scikit-Learn can be found here
https://scikit-learn.org/stable/index.html. We use the Boston Housing
dataset, present in Scikit-Learn:
https://scikit-learn.org/stable/datasets/index.html#boston-dataset

More info on the dataset:

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. ‘Hedonic
prices and the demand for clean air’, J. Environ. Economics &
Management, vol.5, 81-102, 1978. Used in Belsley, Kuh & Welsch,
‘Regression diagnostics …’, Wiley, 1980. N.B. Various transformations
are used in the table on pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning
papers that address regression problems. References

-  Belsley, Kuh & Welsch, ‘Regression diagnostics: Identifying
   Influential Data and Sources of Collinearity’, Wiley, 1980. 244-261.
-  Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning.
   In Proceedings on the Tenth International Conference of Machine
   Learning, 236-243, University of Massachusetts, Amherst. Morgan
   Kaufmann.

**This sample is provided for demonstration purposes, make sure to
conduct appropriate testing if derivating this code for your own
use-cases!**

.. code:: ipython3

    import datetime
    import time
    import tarfile
    
    import boto3
    import pandas as pd
    import numpy as np
    from sagemaker import get_execution_role
    import sagemaker
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    
    
    sm_boto3 = boto3.client('sagemaker')
    
    sess = sagemaker.Session()
    
    region = sess.boto_session.region_name
    
    bucket = sess.default_bucket()  # this could also be a hard-coded bucket name
    
    print('Using bucket ' + bucket)

Prepare data
------------

We load a dataset from sklearn, split it and send it to S3

.. code:: ipython3

    # we use the Boston housing dataset 
    data = load_boston()

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42)
    
    trainX = pd.DataFrame(X_train, columns=data.feature_names)
    trainX['target'] = y_train
    
    testX = pd.DataFrame(X_test, columns=data.feature_names)
    testX['target'] = y_test

.. code:: ipython3

    trainX.head()

.. code:: ipython3

    trainX.to_csv('boston_train.csv')
    testX.to_csv('boston_test.csv')

.. code:: ipython3

    # send data to S3. SageMaker will take training data from s3
    trainpath = sess.upload_data(
        path='boston_train.csv', bucket=bucket,
        key_prefix='sagemaker/sklearncontainer')
    
    testpath = sess.upload_data(
        path='boston_test.csv', bucket=bucket,
        key_prefix='sagemaker/sklearncontainer')

Writing a *Script Mode* script
------------------------------

The below script contains both training and inference functionality and
can run both in SageMaker Training hardware or locally (desktop,
SageMaker notebook, on prem, etc). Detailed guidance here
https://sagemaker.readthedocs.io/en/stable/using_sklearn.html#preparing-the-scikit-learn-training-script

.. code:: ipython3

    %%writefile script.py
    
    import argparse
    import joblib
    import os
    
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    
    
    
    # inference functions ---------------
    def model_fn(model_dir):
        clf = joblib.load(os.path.join(model_dir, "model.joblib"))
        return clf
    
    
    
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
        parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
        parser.add_argument('--train-file', type=str, default='boston_train.csv')
        parser.add_argument('--test-file', type=str, default='boston_test.csv')
        parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
        parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target
    
        args, _ = parser.parse_known_args()
    
        print('reading data')
        train_df = pd.read_csv(os.path.join(args.train, args.train_file))
        test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
        print('building training and testing datasets')
        X_train = train_df[args.features.split()]
        X_test = test_df[args.features.split()]
        y_train = train_df[args.target]
        y_test = test_df[args.target]
    
        # train
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
        path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(model, path)
        print('model persisted at ' + path)
        print(args.min_samples_leaf)

Local training
--------------

Script arguments allows us to remove from the script any
SageMaker-specific configuration, and run locally

.. code:: ipython3

    ! python script.py --n-estimators 100 \
                       --min-samples-leaf 2 \
                       --model-dir ./ \
                       --train ./ \
                       --test ./ \
                       --features 'CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT' \
                       --target target

SageMaker Training
------------------

Launching a training job with the Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # We use the Estimator from the SageMaker Python SDK
    from sagemaker.sklearn.estimator import SKLearn
    
    FRAMEWORK_VERSION = '0.23-1'
    
    sklearn_estimator = SKLearn(
        entry_point='script.py',
        role = get_execution_role(),
        train_instance_count=1,
        train_instance_type='ml.c5.xlarge',
        framework_version=FRAMEWORK_VERSION,
        base_job_name='rf-scikit',
        metric_definitions=[
            {'Name': 'median-AE',
             'Regex': "AE-at-50th-percentile: ([0-9.]+).*$"}],
        hyperparameters = {'n-estimators': 100,
                           'min-samples-leaf': 3,
                           'features': 'CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT',
                           'target': 'target'})

.. code:: ipython3

    # launch training job, with asynchronous call
    sklearn_estimator.fit({'train':trainpath, 'test': testpath}, wait=False)

Alternative: launching a training with ``boto3``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``boto3`` is more verbose yet gives more visibility in the low-level
details of Amazon SageMaker

.. code:: ipython3

    # first compress the code and send to S3
    
    source = 'source.tar.gz'
    project = 'scikitlearn-train-from-boto3'
    
    tar = tarfile.open(source, 'w:gz')
    tar.add ('script.py')
    tar.close()
    
    s3 = boto3.client('s3')
    s3.upload_file(source, bucket, project+'/'+source)

When using ``boto3`` to launch a training job we must explicitly point
to a docker image.

.. code:: ipython3

    from sagemaker.fw_registry import default_framework_uri
    
    training_image = default_framework_uri(
        'scikit-learn', region, '{}-cpu-py3'.format(FRAMEWORK_VERSION))
    print(training_image)

.. code:: ipython3

    # launch training job
    
    response = sm_boto3.create_training_job(
        TrainingJobName='sklearn-boto3-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        HyperParameters={
            'n_estimators': '300',
            'min_samples_leaf': '3',
            'sagemaker_program': 'script.py',
            'features': 'CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT',
            'target': 'target',
            'sagemaker_submit_directory': 's3://' + bucket + '/' + project + '/' + source 
        },
        AlgorithmSpecification={
            'TrainingImage': training_image,
            'TrainingInputMode': 'File',
            'MetricDefinitions': [
                {'Name': 'median-AE', 'Regex': 'AE-at-50th-percentile: ([0-9.]+).*$'},
            ]
        },
        RoleArn=get_execution_role(),
        InputDataConfig=[
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': trainpath,
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                }},
            {
                'ChannelName': 'test',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': testpath,
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                }},
        ],
        OutputDataConfig={'S3OutputPath': 's3://'+ bucket + '/sagemaker-sklearn-artifact/'},
        ResourceConfig={
            'InstanceType': 'ml.c5.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={'MaxRuntimeInSeconds': 86400},
        EnableNetworkIsolation=False
    )
    
    print(response)

Launching a tuning job with the Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # we use the Hyperparameter Tuner
    from sagemaker.tuner import IntegerParameter
    
    # Define exploration boundaries
    hyperparameter_ranges = {
        'n-estimators': IntegerParameter(20, 100),
        'min-samples-leaf': IntegerParameter(2, 6)}
    
    # create Optimizer
    Optimizer = sagemaker.tuner.HyperparameterTuner(
        estimator=sklearn_estimator,
        hyperparameter_ranges=hyperparameter_ranges,
        base_tuning_job_name='RF-tuner',
        objective_type='Minimize',
        objective_metric_name='median-AE',
        metric_definitions=[
            {'Name': 'median-AE',
             'Regex': "AE-at-50th-percentile: ([0-9.]+).*$"}],  # extract tracked metric from logs with regexp 
        max_jobs=20,
        max_parallel_jobs=2)

.. code:: ipython3

    Optimizer.fit({'train': trainpath, 'test': testpath})

.. code:: ipython3

    # get tuner results in a df
    results = Optimizer.analytics().dataframe()
    while results.empty:
        time.sleep(1)
        results = Optimizer.analytics().dataframe()
    results.head()

Deploy to a real-time endpoint
------------------------------

Deploy with Python SDK
~~~~~~~~~~~~~~~~~~~~~~

An ``Estimator`` could be deployed directly after training, with an
``Estimator.deploy()`` but here we showcase the more extensive process
of creating a model from s3 artifacts, that could be used to deploy a
model that was trained in a different session or even out of SageMaker.

.. code:: ipython3

    sklearn_estimator.latest_training_job.wait(logs='None')
    artifact = sm_boto3.describe_training_job(
        TrainingJobName=sklearn_estimator.latest_training_job.name)['ModelArtifacts']['S3ModelArtifacts']
    
    print('Model artifact persisted at ' + artifact)

.. code:: ipython3

    from sagemaker.sklearn.model import SKLearnModel
    
    model = SKLearnModel(
        model_data=artifact,
        role=get_execution_role(),
        entry_point='script.py',
        framework_version=FRAMEWORK_VERSION)

.. code:: ipython3

    predictor = model.deploy(
        instance_type='ml.c5.large',
        initial_instance_count=1)

Invoke with the Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # the SKLearnPredictor does the serialization from pandas for us
    print(predictor.predict(testX[data.feature_names]))

Alternative: invoke with ``boto3``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    runtime = boto3.client('sagemaker-runtime')

Option 1: ``csv`` serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # csv serialization
    response = runtime.invoke_endpoint(
        EndpointName=predictor.endpoint,
        Body=testX[data.feature_names].to_csv(header=False, index=False).encode('utf-8'),
        ContentType='text/csv')
    
    print(response['Body'].read())

Option 2: ``npy`` serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # npy serialization
    from io import BytesIO
    
    
    #Serialise numpy ndarray as bytes
    buffer = BytesIO()
    # Assuming testX is a data frame
    np.save(buffer, testX[data.feature_names].values)
    
    response = runtime.invoke_endpoint(
        EndpointName=predictor.endpoint,
        Body=buffer.getvalue(),
        ContentType='application/x-npy')
    
    print(response['Body'].read())

Don’t forget to delete the endpoint !
-------------------------------------

.. code:: ipython3

    sm_boto3.delete_endpoint(EndpointName=predictor.endpoint)
