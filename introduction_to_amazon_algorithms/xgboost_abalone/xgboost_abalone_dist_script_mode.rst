Regression with Amazon SageMaker XGBoost algorithm
==================================================

**Distributed training for regression with Amazon SageMaker XGBoost
script mode**

--------------

Contents
--------

1. `Introduction <#Introduction>`__
2. `Setup <#Setup>`__
3. `Fetching the dataset <#Fetching-the-dataset>`__
4. `Data Ingestion <#Data-ingestion>`__
5. `Training the XGBoost model <#Training-the-XGBoost-model>`__
6. `Deploying the XGBoost model <#Deploying-the-XGBoost-model>`__

--------------

Introduction
------------

This notebook demonstrates the use of Amazon SageMaker XGBoost to train
and host a regression model. `XGBoost (eXtreme Gradient
Boosting) <https://xgboost.readthedocs.io>`__ is a popular and efficient
machine learning algorithm used for regression and classification tasks
on tabular datasets. It implements a technique know as gradient boosting
on trees, and performs remarkably well in machine learning competitions,
and gets a lot of attention from customers.

We use the `Abalone
data <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html>`__,
originally from the `UCI data
repository <https://archive.ics.uci.edu/ml/datasets/abalone>`__. More
details about the original dataset can be found
`here <https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names>`__.
In this libsvm converted version, the nominal feature
(Male/Female/Infant) has been converted into a real valued feature as
required by XGBoost. Age of abalone is to be predicted from eight
physical measurements.

--------------

Setup
-----

This notebook was created and tested on an ml.m5.2xlarge notebook
instance.

Let’s start by specifying: 1. The S3 bucket and prefix that you want to
use for training and model data. This should be within the same region
as the Notebook Instance, training, and hosting. 1. The IAM role arn
used to give training and hosting access to your data. See the
documentation for how to create these. Note, if more than one role is
required for notebook instances, training, and/or hosting, please
replace the boto regexp with a the appropriate full IAM role arn
string(s).

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install -qU awscli boto3 "sagemaker>=1.71.0,<2.0.0"

.. code:: ipython3

    %%time
    
    import os
    import boto3
    import re
    import sagemaker
    
    # Get a SageMaker-compatible role used by this Notebook Instance.
    role = sagemaker.get_execution_role()
    region = boto3.Session().region_name
    
    ### update below values appropriately ###
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-xgboost-dist-script'
    #### 
    
    print(region)

Fetching the dataset
~~~~~~~~~~~~~~~~~~~~

Following methods split the data into train/test/validation datasets and
upload files to S3.

.. code:: ipython3

    %%time
    
    import io
    import boto3
    import random
    
    def data_split(FILE_DATA, DATA_DIR, FILE_TRAIN_BASE, FILE_TRAIN_1, FILE_VALIDATION, FILE_TEST, 
                   PERCENT_TRAIN_0, PERCENT_TRAIN_1, PERCENT_VALIDATION, PERCENT_TEST):
        data = [l for l in open(FILE_DATA, 'r')]
        train_file_0 = open(DATA_DIR + "/" + FILE_TRAIN_0, 'w')
        train_file_1 = open(DATA_DIR + "/" + FILE_TRAIN_1, 'w')
        valid_file = open(DATA_DIR + "/" + FILE_VALIDATION, 'w')
        tests_file = open(DATA_DIR + "/" + FILE_TEST, 'w')
    
        num_of_data = len(data)
        num_train_0 = int((PERCENT_TRAIN_0/100.0)*num_of_data)
        num_train_1 = int((PERCENT_TRAIN_1/100.0)*num_of_data)
        num_valid = int((PERCENT_VALIDATION/100.0)*num_of_data)
        num_tests = int((PERCENT_TEST/100.0)*num_of_data)
    
        data_fractions = [num_train_0, num_train_1, num_valid, num_tests]
        split_data = [[],[],[],[]]
    
        rand_data_ind = 0
    
        for split_ind, fraction in enumerate(data_fractions):
            for i in range(fraction):
                rand_data_ind = random.randint(0, len(data)-1)
                split_data[split_ind].append(data[rand_data_ind])
                data.pop(rand_data_ind)
    
        for l in split_data[0]:
            train_file_0.write(l)
    
        for l in split_data[1]:
            train_file_1.write(l)
            
        for l in split_data[2]:
            valid_file.write(l)
    
        for l in split_data[3]:
            tests_file.write(l)
    
        train_file_0.close()
        train_file_1.close()
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
    FILE_TRAIN_0 = 'abalone.train_0'
    FILE_TRAIN_1 = 'abalone.train_1'
    FILE_VALIDATION = 'abalone.validation'
    FILE_TEST = 'abalone.test'
    PERCENT_TRAIN_0 = 35
    PERCENT_TRAIN_1 = 35
    PERCENT_VALIDATION = 15
    PERCENT_TEST = 15
    
    DATA_DIR = 'data'
    
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    
    data_split(FILE_DATA, DATA_DIR, FILE_TRAIN_0, FILE_TRAIN_1, FILE_VALIDATION, FILE_TEST, 
               PERCENT_TRAIN_0, PERCENT_TRAIN_1, PERCENT_VALIDATION, PERCENT_TEST)


.. code:: ipython3

    #upload the files to the S3 bucket
    upload_to_s3(bucket, 'train/train_0.libsvm', DATA_DIR + "/" + FILE_TRAIN_0)
    upload_to_s3(bucket, 'train/train_1.libsvm', DATA_DIR + "/" + FILE_TRAIN_1)
    upload_to_s3(bucket, 'validation/validation.libsvm', DATA_DIR + "/" + FILE_VALIDATION)
    upload_to_s3(bucket, 'test/test.libsvm', DATA_DIR + "/" + FILE_TEST)

Create a XGBoost script to train with
-------------------------------------

SageMaker can now run an XGboost script using the XGBoost estimator.
When executed on SageMaker a number of helpful environment variables are
available to access properties of the training environment, such as:

-  ``SM_MODEL_DIR``: A string representing the path to the directory to
   write model artifacts to. Any artifacts saved in this folder are
   uploaded to S3 for model hosting after the training job completes.
-  ``SM_OUTPUT_DIR``: A string representing the filesystem path to write
   output artifacts to. Output artifacts may include checkpoints,
   graphs, and other files to save, not including model artifacts. These
   artifacts are compressed and uploaded to S3 to the same S3 prefix as
   the model artifacts.

Supposing two input channels, ‘train’ and ‘validation’, were used in the
call to the XGBoost estimator’s fit() method, the following environment
variables will be set, following the format
``SM_CHANNEL_[channel_name]``:

``SM_CHANNEL_TRAIN``: A string representing the path to the directory
containing data in the ‘train’ channel ``SM_CHANNEL_VALIDATION``: Same
as above, but for the ‘validation’ channel.

A typical training script loads data from the input channels, configures
training with hyperparameters, trains a model, and saves a model to
model_dir so that it can be hosted later. Hyperparameters are passed to
your script as arguments and can be retrieved with an
argparse.ArgumentParser instance. For example, the script that we will
run in this notebook is provided as the accompanying file
(``abalone.py``) and also shown below:

.. code:: python


   import argparse
   import json
   import logging
   import os
   import pandas as pd
   import pickle as pkl

   from sagemaker_containers import entry_point
   from sagemaker_xgboost_container.data_utils import get_dmatrix
   from sagemaker_xgboost_container import distributed

   import xgboost as xgb


   def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master):
       """Run xgb train on arguments given with rabit initialized.

       This is our rabit execution function.

       :param args_dict: Argument dictionary used to run xgb.train().
       :param is_master: True if current node is master host in distributed training, or is running single node training job. Note that rabit_run will include this argument.
       """
       booster = xgb.train(params=params, dtrain=dtrain, evals=evals, num_boost_round=num_boost_round)

       if is_master:
           model_location = model_dir + '/xgboost-model'
           pkl.dump(booster, open(model_location, 'wb'))
           logging.info("Stored trained model at {}".format(model_location))


   if __name__ == '__main__':
       parser = argparse.ArgumentParser()

       # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
       parser.add_argument('--max_depth', type=int,)
       parser.add_argument('--eta', type=float)
       parser.add_argument('--gamma', type=int)
       parser.add_argument('--min_child_weight', type=int)
       parser.add_argument('--subsample', type=float)
       parser.add_argument('--verbose', type=int)
       parser.add_argument('--objective', type=str)
       parser.add_argument('--num_round', type=int)

       # Sagemaker specific arguments. Defaults are set in the environment variables.
       parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
       parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
       parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
       parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
       parser.add_argument('--sm_hosts', type=str, default=os.environ['SM_HOSTS'])
       parser.add_argument('--sm_current_host', type=str, default=os.environ['SM_CURRENT_HOST'])

       args, _ = parser.parse_known_args()

       # Get SageMaker host information from runtime environment variables
       sm_hosts = json.loads(os.environ['SM_HOSTS'])
       sm_current_host = args.sm_current_host

       dtrain = get_dmatrix(args.train, 'libsvm')
       dval = get_dmatrix(args.validation, 'libsvm')
       watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

       train_hp = {
           'max_depth': args.max_depth,
           'eta': args.eta,
           'gamma': args.gamma,
           'min_child_weight': args.min_child_weight,
           'subsample': args.subsample,
           'verbose': args.verbose,
           'objective': args.objective}

       xgb_train_args = dict(
           params=train_hp,
           dtrain=dtrain,
           evals=watchlist,
           num_boost_round=args.num_round,
           model_dir=args.model_dir)

       if len(sm_hosts) > 1:
           # Wait until all hosts are able to find each other
           entry_point._wait_hostname_resolution()

           # Execute training function after initializing rabit.
           distributed.rabit_run(
               exec_fun=_xgb_train,
               args=xgb_train_args,
               include_in_training=(dtrain is not None),
               hosts=sm_hosts,
               current_host=sm_current_host,
               update_rabit_args=True
           )
       else:
           # If single node training, call training method directly.
           if dtrain:
               xgb_train_args['is_master'] = True
               _xgb_train(**xgb_train_args)
           else:
               raise ValueError("Training channel must have data to train model.")


   def model_fn(model_dir):
       """Deserialized and return fitted model.

       Note that this should have the same name as the serialized model in the _xgb_train method
       """
       model_file = 'xgboost-model'
       booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
       return booster

Because the container imports your training script, always put your
training code in a main guard ``(if __name__=='__main__':)`` so that the
container does not inadvertently run your training code at the wrong
point in execution.

For more information about training environment variables, please visit
https://github.com/aws/sagemaker-containers.

Training the XGBoost model
--------------------------

After setting training parameters, we kick off training, and poll for
status until training is completed, which in this example, takes between
few minutes.

To run our training script on SageMaker, we construct a
sagemaker.xgboost.estimator.XGBoost estimator, which accepts several
constructor arguments:

-  **entry_point**: The path to the Python script SageMaker runs for
   training and prediction.
-  **role**: Role ARN
-  **train_instance_type** *(optional)*: The type of SageMaker instances
   for training. **Note**: Because Scikit-learn does not natively
   support GPU training, Sagemaker Scikit-learn does not currently
   support training on GPU instance types.
-  **sagemaker_session** *(optional)*: The session used to train on
   Sagemaker.
-  **hyperparameters** *(optional)*: A dictionary passed to the train
   function as hyperparameters.

.. code:: ipython3

    hyperparams = {
            "max_depth":"5",
            "eta":"0.2",
            "gamma":"4",
            "min_child_weight":"6",
            "subsample":"0.7",
            "verbose":"1",
            "objective":"reg:linear",
            "num_round":"50"}
    
    instance_type = "ml.m5.2xlarge"
    output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-dist-xgb')
    content_type = "libsvm"

.. code:: ipython3

    # Open Source distributed script mode
    from sagemaker.session import s3_input, Session
    from sagemaker.xgboost.estimator import XGBoost
    
    boto_session = boto3.Session(region_name=region)
    session = Session(boto_session=boto_session)
    script_path = 'abalone.py'
    
    xgb_script_mode_estimator = XGBoost(
        entry_point=script_path,
        framework_version='1.0-1', # Note: framework_version is mandatory
        hyperparameters=hyperparams,
        role=role,
        train_instance_count=2, 
        train_instance_type=instance_type,
        output_path=output_path)
    
    train_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)
    validation_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'validation'), content_type=content_type)

Train XGBoost Estimator on abalone data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training is as simple as calling ``fit`` on the Estimator. This will
start a SageMaker Training job that will download the data, invoke the
entry point code (in the provided script file), and save any model
artifacts that the script creates.

.. code:: ipython3

    xgb_script_mode_estimator.fit({'train': train_input, 'validation': validation_input})

Deploying the XGBoost model
---------------------------

After training, we can use the estimator to create an Amazon SageMaker
endpoint – a hosted and managed prediction service that we can use to
perform inference.

You can also optionally specify other functions to customize the
behavior of deserialization of the input request (``input_fn()``),
serialization of the predictions (``output_fn()``), and how predictions
are made (``predict_fn()``). The defaults work for our current use-case
so we don’t need to define them.

.. code:: ipython3

    predictor = xgb_script_mode_estimator.deploy(initial_instance_count=1, 
                                                 instance_type="ml.m5.2xlarge")
    predictor.serializer = str

.. code:: ipython3

    test_file = DATA_DIR + "/" + FILE_TEST
    with open(test_file, 'r') as f:
        payload = f.read()

.. code:: ipython3

    runtime_client = boto3.client('runtime.sagemaker', region_name=region)
    response = runtime_client.invoke_endpoint(EndpointName=predictor.endpoint, 
                                              ContentType='text/x-libsvm', 
                                              Body=payload)
    result = response['Body'].read().decode('ascii')
    print('Predicted values are {}.'.format(result))

(Optional) Delete the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you’re done with this exercise, please run the delete_endpoint line
in the cell below. This will remove the hosted endpoint and avoid any
charges from a stray instance being left on.

.. code:: ipython3

    xgb_script_mode_estimator.delete_endpoint()
