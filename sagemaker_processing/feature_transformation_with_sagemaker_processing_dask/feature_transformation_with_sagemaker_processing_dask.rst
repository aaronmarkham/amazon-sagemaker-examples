Feature transformation with Amazon SageMaker Processing and Dask
================================================================

Typically a machine learning (ML) process consists of few steps. First,
gathering data with various ETL jobs, then pre-processing the data,
featurizing the dataset by incorporating standard techniques or prior
knowledge, and finally training an ML model using an algorithm.

Often, distributed data processing frameworks such as Dask are used to
pre-process data sets in order to prepare them for training. In this
notebook we’ll use Amazon SageMaker Processing, and leverage the power
of Dask in a managed SageMaker environment to run our preprocessing
workload.

What is Dask Distributed?
~~~~~~~~~~~~~~~~~~~~~~~~~

Dask.distributed: is a lightweight and open source library for
distributed computing in Python. It is also a centrally managed,
distributed, dynamic task scheduler. It is also a centrally managed,
distributed, dynamic task scheduler. Dask has three main components:

**dask-scheduler process:** coordinates the actions of several workers.
The scheduler is asynchronous and event-driven, simultaneously
responding to requests for computation from multiple clients and
tracking the progress of multiple workers.

**dask-worker processes:** Which are spread across multiple machines and
the concurrent requests of several clients.

**dask-client process:** which is is the primary entry point for users
of dask.distributed

source: https://docs.dask.org/en/latest/

Contents
--------

1. `Objective <#Objective:-predict-the-age-of-an-Abalone-from-its-physical-measurement>`__
2. `Setup <#Setup>`__
3. `Using Amazon SageMaker Processing to execute a Dask
   Job <#Using-Amazon-SageMaker-Processing-to-execute-a-Dask-Job>`__
4. `Downloading dataset and uploading to
   S3 <#Downloading-dataset-and-uploading-to-S3>`__
5. `Build a Dask container for running the preprocessing
   job <#Build-a-Dask-container-for-running-the-preprocessing-job>`__
6. `Run the preprocessing job using Amazon SageMaker
   Processing <#Run-the-preprocessing-job-using-Amazon-SageMaker-Processing>`__
   1. `Inspect the preprocessed
   dataset <#Inspect-the-preprocessed-dataset>`__

Setup
-----

Let’s start by specifying: \* The S3 bucket and prefixes that you use
for training and model data. Use the default bucket specified by the
Amazon SageMaker session. \* The IAM role ARN used to give processing
and training access to the dataset.

.. code:: ipython3

    from time import gmtime, strftime
    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sagemaker_session.default_bucket()
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    
    prefix = "sagemaker/dask-preprocess-demo"
    input_prefix = prefix + "/input/raw/census"
    input_preprocessed_prefix = prefix + "/input/preprocessed/census"
    model_prefix = prefix + "/model"

Using Amazon SageMaker Processing to execute a Dask job
-------------------------------------------------------

Downloading dataset and uploading to Amazon Simple Storage Service (Amazon S3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset used here is the Census-Income KDD Dataset. The first step
are to select features, clean the data, and turn the data into features
that the training algorithm can use to train a binary classification
model which can then be used to predict whether rows representing census
responders have an income greater or less than $50,000. In this example,
we will use Dask distributed to preprocess and transform the data to
make it ready for the training process. In the next section, you
download from the bucket below then upload to your own bucket so that
Amazon SageMaker can access the dataset.

.. code:: ipython3

    import boto3
    import pandas as pd
    
    s3 = boto3.client('s3')
    region = sagemaker_session.boto_region_name
    input_data = 's3://sagemaker-sample-data-{}/processing/census/census-income.csv'.format(region)
    !aws s3 cp $input_data .
    
    # Uploading the training data to S3
    sagemaker_session.upload_data(path='census-income.csv', bucket=bucket, key_prefix=input_prefix)

Build a dask container for running the preprocessing job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example Dask container is included in the ``./container`` directory
of this example. The container handles the bootstrapping of Dask
Scheduler and mapping each instance to a Dask Worke. At a high level the
container provides:

-  A set of default worker/scheduler configurations
-  A bootstrapping script for configuring and starting up
   scheduler/worker nodes
-  Starting dask cluster from all the workers including the scheduler
   node

After the container build and push process is complete, use the Amazon
SageMaker Python SDK to submit a managed, distributed dask application
that performs our dataset preprocessing.

Build the example Dask container.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %cd container
    !docker build -t sagemaker-dask-example .
    %cd ../

Create an Amazon Elastic Container Registry (Amazon ECR) repository for the Dask container and push the image.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import boto3
    
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name
    
    ecr_repository = 'sagemaker-dask-example'
    tag = ':latest'
    uri_suffix = 'amazonaws.com'
    if region in ['cn-north-1', 'cn-northwest-1']:
        uri_suffix = 'amazonaws.com.cn'
    dask_repository_uri = '{}.dkr.ecr.{}.{}/{}'.format(account_id, region, uri_suffix, ecr_repository + tag)
    
    # Create ECR repository and push docker image
    !$(aws ecr get-login --region $region --registry-ids $account_id --no-include-email)
    !aws ecr create-repository --repository-name $ecr_repository
    !docker tag {ecr_repository + tag} $dask_repository_uri
    !docker push $dask_repository_uri

Run the preprocessing job using Amazon SageMaker Processing on Dask Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, use the Amazon SageMaker Python SDK to submit a processing job.
Use the the custom Dask container that was just built, and a Scikit
Learn script for preprocessing in the job configuration.

Create the Dask preprocessing script.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%writefile preprocess.py
    from __future__ import print_function, unicode_literals
    import argparse
    import json
    import logging
    import os
    import sys
    import time
    import warnings
    import boto3
    import numpy as np
    import pandas as pd
    from tornado import gen
    import dask.dataframe as dd
    import joblib
    from dask.distributed import Client
    from sklearn.compose import make_column_transformer
    from sklearn.exceptions import DataConversionWarning
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import (
        KBinsDiscretizer,
        LabelBinarizer,
        OneHotEncoder,
        PolynomialFeatures,
        StandardScaler,
    )
    
    warnings.filterwarnings(action="ignore", category=DataConversionWarning)
    attempts_counter = 3
    attempts = 0
    
    
    def upload_objects(bucket, prefix, local_path):
        try:
            bucket_name = bucket  # s3 bucket name
            root_path = local_path  # local folder for upload
    
            s3_bucket = s3_client.Bucket(bucket_name)
    
            for path, subdirs, files in os.walk(root_path):
                for file in files:
                    s3_bucket.upload_file(
                        os.path.join(path, file), "{}/output/{}".format(prefix, file)
                    )
        except Exception as err:
            logging.exception(err)
    
    
    def print_shape(df):
        negative_examples, positive_examples = np.bincount(df["income"])
        print(
            "Data shape: {}, {} positive examples, {} negative examples".format(
                df.shape, positive_examples, negative_examples
            )
        )
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
        args, _ = parser.parse_known_args()
        
        # Get processor scrip arguments
        args_iter = iter(sys.argv[1:])
        script_args = dict(zip(args_iter, args_iter))
        scheduler_ip = sys.argv[-1]
    
        # S3 client
        s3_region = script_args["s3_region"]
        s3_client = boto3.resource("s3", s3_region)
        print(f'Using the {s3_region} region')
        
        # Start the Dask cluster client
        try:
            client = Client("tcp://{ip}:8786".format(ip=scheduler_ip))
            logging.info("Printing cluster information: {}".format(client))
        except Exception as err:
            logging.exception(err)
    
        columns = [
            "age",
            "education",
            "major industry code",
            "class of worker",
            "num persons worked for employer",
            "capital gains",
            "capital losses",
            "dividends from stocks",
            "income",
        ]
        class_labels = [" - 50000.", " 50000+."]
        input_data_path = "s3://{}".format(os.path.join(
            script_args["s3_input_bucket"],
            script_args["s3_input_key_prefix"],
            "census-income.csv",
        ))
        
        # Creating the necessary paths to save the output files
        if not os.path.exists("/opt/ml/processing/train"):
            os.makedirs("/opt/ml/processing/train")
    
        if not os.path.exists("/opt/ml/processing/test"):
            os.makedirs("/opt/ml/processing/test")
    
        print("Reading input data from {}".format(input_data_path))
        df = pd.read_csv(input_data_path)
        df = pd.DataFrame(data=df, columns=columns)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.replace(class_labels, [0, 1], inplace=True)
    
        negative_examples, positive_examples = np.bincount(df["income"])
        print(
            "Data after cleaning: {}, {} positive examples, {} negative examples".format(
                df.shape, positive_examples, negative_examples
            )
        )
    
        split_ratio = args.train_test_split_ratio
        print("Splitting data into train and test sets with ratio {}".format(split_ratio))
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0
        )
    
        preprocess = make_column_transformer(
            (
                KBinsDiscretizer(encode="onehot-dense", n_bins=2),
                ["age", "num persons worked for employer"],
            ),
            (
                StandardScaler(),
                ["capital gains", "capital losses", "dividends from stocks"],
            ),
            (
                OneHotEncoder(sparse=False),
                ["education", "major industry code", "class of worker"],
            ),
        )
    
        print("Running preprocessing and feature engineering transformations in Dask")
        with joblib.parallel_backend("dask"):
            train_features = preprocess.fit_transform(X_train)
            test_features = preprocess.transform(X_test)
    
        print("Train data shape after preprocessing: {}".format(train_features.shape))
        print("Test data shape after preprocessing: {}".format(test_features.shape))
    
        train_features_output_path = os.path.join(
            "/opt/ml/processing/train", "train_features.csv"
        )
        train_labels_output_path = os.path.join(
            "/opt/ml/processing/train", "train_labels.csv"
        )
    
        test_features_output_path = os.path.join(
            "/opt/ml/processing/test", "test_features.csv"
        )
        test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")
    
        print("Saving training features to {}".format(train_features_output_path))
        pd.DataFrame(train_features).to_csv(
            train_features_output_path, header=False, index=False
        )
    
        print("Saving test features to {}".format(test_features_output_path))
        pd.DataFrame(test_features).to_csv(
            test_features_output_path, header=False, index=False
        )
    
        print("Saving training labels to {}".format(train_labels_output_path))
        y_train.to_csv(train_labels_output_path, header=False, index=False)
    
        print("Saving test labels to {}".format(test_labels_output_path))
        y_test.to_csv(test_labels_output_path, header=False, index=False)
        upload_objects(
            script_args["s3_output_bucket"],
            script_args["s3_output_key_prefix"],
            "/opt/ml/processing/train/",
        )
        upload_objects(
            script_args["s3_output_bucket"],
            script_args["s3_output_key_prefix"],
            "/opt/ml/processing/test/",
        )
    
        # wait for the file creation
        while attempts < attempts_counter:
            if os.path.exists(train_features_output_path) and os.path.isfile(
                train_features_output_path
            ):
                try:
                    # Calculate the processed dataset baseline statistics on the Dask cluster
                    dask_df = dd.read_csv(train_features_output_path)
                    dask_df = client.persist(dask_df)
                    baseline = dask_df.describe().compute()
                    print(baseline)
                    break
    
                except:
                    time.sleep(2)
        if attempts == attempts_counter:
            raise Exception(
                "Output file {} couldn't be found".format(train_features_output_path)
            )
    
        print(client)
        sys.exit(os.EX_OK)

Run a processing job using the Docker image and preprocessing script you
just created. When invoking the ``dask_processor.run()`` function, pass
the Amazon S3 input and output paths as arguments that are required by
our preprocessing script to determine input and output location in
Amazon S3. Here, you also specify the number of instances and instance
type that will be used for the distributed Spark job.

.. code:: ipython3

    from sagemaker.processing import ProcessingInput, ScriptProcessor
    
    dask_processor = ScriptProcessor(
        base_job_name="dask-preprocessor",
        image_uri=dask_repository_uri,
        command=["/opt/program/bootstrap.py"],
        role=role,
        instance_count=2,
        instance_type="ml.m5.large",
        max_runtime_in_seconds=1200,
    )
    
    dask_processor.run(
        code="preprocess.py",
        arguments=[
            "s3_input_bucket",
            bucket,
            "s3_input_key_prefix",
            input_prefix,
            "s3_output_bucket",
            bucket,
            "s3_output_key_prefix",
            input_preprocessed_prefix,
            "s3_region",
            region
        ],
        logs=True
    )

Inspect the preprocessed dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take a look at a few rows of the transformed dataset to make sure the
preprocessing was successful.

.. code:: ipython3

    print('Top 5 rows from s3://{}/{}/train/'.format(bucket, input_preprocessed_prefix))
    !aws s3 cp --quiet s3://$bucket/$input_preprocessed_prefix/output/train_features.csv - | head -n5

Now, you can use the output files of the transformation process as input
to a training job and train a regression model.
