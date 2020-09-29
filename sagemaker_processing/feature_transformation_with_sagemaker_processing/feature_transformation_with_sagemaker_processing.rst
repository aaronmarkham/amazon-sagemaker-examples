Feature transformation with Amazon SageMaker Processing and SparkML
===================================================================

Typically a machine learning (ML) process consists of few steps. First,
gathering data with various ETL jobs, then pre-processing the data,
featurizing the dataset by incorporating standard techniques or prior
knowledge, and finally training an ML model using an algorithm.

Often, distributed data processing frameworks such as Spark are used to
pre-process data sets in order to prepare them for training. In this
notebook we’ll use Amazon SageMaker Processing, and leverage the power
of Spark in a managed SageMaker environment to run our preprocessing
workload. Then, we’ll take our preprocessed dataset and train a
regression model using XGBoost.

Contents
--------

1.  `Objective <#Objective:-predict-the-age-of-an-Abalone-from-its-physical-measurement>`__
2.  `Setup <#Setup>`__
3.  `Using Amazon SageMaker Processing to execute a SparkML
    Job <#Using-Amazon-SageMaker-Processing-to-execute-a-SparkML-Job>`__
4.  `Downloading dataset and uploading to
    S3 <#Downloading-dataset-and-uploading-to-S3>`__
5.  `Build a Spark container for running the preprocessing
    job <#Build-a-Spark-container-for-running-the-preprocessing-job>`__
6.  `Run the preprocessing job using Amazon SageMaker
    Processing <#Run-the-preprocessing-job-using-Amazon-SageMaker-Processing>`__
    1. `Inspect the preprocessed
    dataset <#Inspect-the-preprocessed-dataset>`__
7.  `Train a regression model using the Amazon SageMaker XGBoost
    algorithm <#Train-a-regression-model-using-the-SageMaker-XGBoost-algorithm>`__
8.  `Retrieve the XGBoost algorithm
    image <#Retrieve-the-XGBoost-algorithm-image>`__
9.  `Set XGBoost model parameters and dataset
    details <#Set-XGBoost-model-parameters-and-dataset-details>`__
10. `Train the XGBoost model <#Train-the-XGBoost-model>`__

Objective: predict the age of an Abalone from its physical measurement
----------------------------------------------------------------------

The dataset is available from `UCI Machine
Learning <https://archive.ics.uci.edu/ml/datasets/abalone>`__. The aim
for this task is to determine age of an Abalone (a kind of shellfish)
from its physical measurements. At the core, it’s a regression problem.
The dataset contains several features - ``sex`` (categorical),
``length`` (continuous), ``diameter`` (continuous), ``height``
(continuous), ``whole_weight`` (continuous), ``shucked_weight``
(continuous), ``viscera_weight`` (continuous), ``shell_weight``
(continuous) and ``rings`` (integer).Our goal is to predict the variable
``rings`` which is a good approximation for age (age is ``rings`` +
1.5).

Use SparkML to process the dataset (apply one or many feature
transformers) and upload the transformed dataset to Amazon S3 so that it
can be used for training with XGBoost.

Setup
-----

Let’s start by specifying: \* The S3 bucket and prefixes that you use
for training and model data. Use the default bucket specified by the
Amazon SageMaker session. \* The IAM role ARN used to give processing
and training access to the dataset.

.. code:: ipython3

    import sagemaker
    from time import gmtime, strftime
    
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sagemaker_session.default_bucket()
    
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    
    prefix = 'sagemaker/spark-preprocess-demo/' + timestamp_prefix
    input_prefix = prefix + '/input/raw/abalone'
    input_preprocessed_prefix = prefix + '/input/preprocessed/abalone'
    model_prefix = prefix + '/model'

Using Amazon SageMaker Processing to execute a SparkML job
----------------------------------------------------------

Downloading dataset and uploading to Amazon Simple Storage Service (Amazon S3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Amazon SageMaker team downloaded the abalone dataset from the
University of California, Irvine repository and uploaded it to an S3
buckets. In this notebook, you download from that bucket and upload to
your own bucket so that Amazon SageMaker can access the dataset.

.. code:: ipython3

    # Fetch the dataset from the SageMaker bucket
    !wget https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv
    
    # Uploading the training data to S3
    sagemaker_session.upload_data(path='abalone.csv', bucket=bucket, key_prefix=input_prefix)

Build a Spark container for running the preprocessing job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example Spark container is included in the ``./container`` directory
of this example. The container handles the bootstrapping of all Spark
configuration, and serves as a wrapper around the ``spark-submit`` CLI.
At a high level the container provides: \* A set of default
Spark/YARN/Hadoop configurations \* A bootstrapping script for
configuring and starting up Spark master/worker nodes \* A wrapper
around the ``spark-submit`` CLI to submit a Spark application

After the container build and push process is complete, use the Amazon
SageMaker Python SDK to submit a managed, distributed Spark application
that performs our dataset preprocessing.

Build the example Spark container.

.. code:: ipython3

    %cd container
    !docker build -t sagemaker-spark-example .
    %cd ../

Create an Amazon Elastic Container Registry (Amazon ECR) repository for
the Spark container and push the image.

.. code:: ipython3

    import boto3
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name
    
    ecr_repository = 'sagemaker-spark-example'
    tag = ':latest'
    uri_suffix = 'amazonaws.com'
    if region in ['cn-north-1', 'cn-northwest-1']:
        uri_suffix = 'amazonaws.com.cn'
    spark_repository_uri = '{}.dkr.ecr.{}.{}/{}'.format(account_id, region, uri_suffix, ecr_repository + tag)
    
    # Create ECR repository and push docker image
    !$(aws ecr get-login --region $region --registry-ids $account_id --no-include-email)
    !aws ecr create-repository --repository-name $ecr_repository
    !docker tag {ecr_repository + tag} $spark_repository_uri
    !docker push $spark_repository_uri

Run the preprocessing job using Amazon SageMaker Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, use the Amazon SageMaker Python SDK to submit a processing job.
Use the Spark container that was just built, and a SparkML script for
preprocessing in the job configuration.

Create the SparkML preprocessing script.

.. code:: ipython3

    %%writefile preprocess.py
    from __future__ import print_function
    from __future__ import unicode_literals
    
    import time
    import sys
    import os
    import shutil
    import csv
    
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.sql.types import StructField, StructType, StringType, DoubleType
    from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import *
    
    
    def csv_line(data):
        r = ','.join(str(d) for d in data[1])
        return str(data[0]) + "," + r
    
    
    def main():
        spark = SparkSession.builder.appName("PySparkAbalone").getOrCreate()
        
        # Convert command line args into a map of args
        args_iter = iter(sys.argv[1:])
        args = dict(zip(args_iter, args_iter))
        
        # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
        spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
                                                          "org.apache.hadoop.mapred.FileOutputCommitter")
        
        # Defining the schema corresponding to the input data. The input data does not contain the headers
        schema = StructType([StructField("sex", StringType(), True), 
                             StructField("length", DoubleType(), True),
                             StructField("diameter", DoubleType(), True),
                             StructField("height", DoubleType(), True),
                             StructField("whole_weight", DoubleType(), True),
                             StructField("shucked_weight", DoubleType(), True),
                             StructField("viscera_weight", DoubleType(), True), 
                             StructField("shell_weight", DoubleType(), True), 
                             StructField("rings", DoubleType(), True)])
    
        # Downloading the data from S3 into a Dataframe
        total_df = spark.read.csv(('s3a://' + os.path.join(args['s3_input_bucket'], args['s3_input_key_prefix'],
                                                       'abalone.csv')), header=False, schema=schema)
    
        #StringIndexer on the sex column which has categorical value
        sex_indexer = StringIndexer(inputCol="sex", outputCol="indexed_sex")
        
        #one-hot-encoding is being performed on the string-indexed sex column (indexed_sex)
        sex_encoder = OneHotEncoder(inputCol="indexed_sex", outputCol="sex_vec")
    
        #vector-assembler will bring all the features to a 1D vector for us to save easily into CSV format
        assembler = VectorAssembler(inputCols=["sex_vec", 
                                               "length", 
                                               "diameter", 
                                               "height", 
                                               "whole_weight", 
                                               "shucked_weight", 
                                               "viscera_weight", 
                                               "shell_weight"], 
                                    outputCol="features")
        
        # The pipeline comprises of the steps added above
        pipeline = Pipeline(stages=[sex_indexer, sex_encoder, assembler])
        
        # This step trains the feature transformers
        model = pipeline.fit(total_df)
        
        # This step transforms the dataset with information obtained from the previous fit
        transformed_total_df = model.transform(total_df)
        
        # Split the overall dataset into 80-20 training and validation
        (train_df, validation_df) = transformed_total_df.randomSplit([0.8, 0.2])
        
        # Convert the train dataframe to RDD to save in CSV format and upload to S3
        train_rdd = train_df.rdd.map(lambda x: (x.rings, x.features))
        train_lines = train_rdd.map(csv_line)
        train_lines.saveAsTextFile('s3a://' + os.path.join(args['s3_output_bucket'], args['s3_output_key_prefix'], 'train'))
        
        # Convert the validation dataframe to RDD to save in CSV format and upload to S3
        validation_rdd = validation_df.rdd.map(lambda x: (x.rings, x.features))
        validation_lines = validation_rdd.map(csv_line)
        validation_lines.saveAsTextFile('s3a://' + os.path.join(args['s3_output_bucket'], args['s3_output_key_prefix'], 'validation'))
    
    
    if __name__ == "__main__":
        main()

Run a processing job using the Docker image and preprocessing script you
just created. When invoking the ``spark_processor.run()`` function, pass
the Amazon S3 input and output paths as arguments that are required by
our preprocessing script to determine input and output location in
Amazon S3. Here, you also specify the number of instances and instance
type that will be used for the distributed Spark job.

.. code:: ipython3

    from sagemaker.processing import ScriptProcessor, ProcessingInput
    spark_processor = ScriptProcessor(base_job_name='spark-preprocessor',
                                      image_uri=spark_repository_uri,
                                      command=['/opt/program/submit'],
                                      role=role,
                                      instance_count=2,
                                      instance_type='ml.r5.xlarge',
                                      max_runtime_in_seconds=1200,
                                      env={'mode': 'python'})
    
    spark_processor.run(code='preprocess.py',
                        arguments=['s3_input_bucket', bucket,
                                   's3_input_key_prefix', input_prefix,
                                   's3_output_bucket', bucket,
                                   's3_output_key_prefix', input_preprocessed_prefix],
                        logs=False)

Inspect the preprocessed dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take a look at a few rows of the transformed dataset to make sure the
preprocessing was successful.

.. code:: ipython3

    print('Top 5 rows from s3://{}/{}/train/'.format(bucket, input_preprocessed_prefix))
    !aws s3 cp --quiet s3://$bucket/$input_preprocessed_prefix/train/part-00000 - | head -n5

Train a regression model using the SageMaker XGBoost algorithm
--------------------------------------------------------------

Use Amazon SageMaker XGBoost algorithm to train on this dataset. You
already know the Amazon S3 location where the preprocessed training data
was uploaded as part of the processing job output.

Retrieve the XGBoost algorithm image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieve the XGBoost built-in algorithm image so that you can use it in
the training job.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    training_image = get_image_uri(sagemaker_session.boto_region_name, 'xgboost', repo_version="0.90-1")
    print(training_image)

Set XGBoost model parameters and dataset details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, configure an Estimator for the XGBoost algorithm and the input
dataset. The notebook is parameterized so that the same data location
used in the SparkML script can now be passed to XGBoost Estimator as
well.

.. code:: ipython3

    s3_train_data = 's3://{}/{}/{}'.format(bucket, input_preprocessed_prefix, 'train/part')
    s3_validation_data = 's3://{}/{}/{}'.format(bucket, input_preprocessed_prefix, 'validation/part')
    s3_output_location = 's3://{}/{}/{}'.format(bucket, prefix, 'xgboost_model')
    
    xgb_model = sagemaker.estimator.Estimator(training_image,
                                              role, 
                                              train_instance_count=1, 
                                              train_instance_type='ml.m4.xlarge',
                                              train_volume_size = 20,
                                              train_max_run = 3600,
                                              input_mode= 'File',
                                              output_path=s3_output_location,
                                              sagemaker_session=sagemaker_session)
    
    xgb_model.set_hyperparameters(objective = "reg:linear",
                                  eta = .2,
                                  gamma = 4,
                                  max_depth = 5,
                                  num_round = 10,
                                  subsample = 0.7,
                                  silent = 0,
                                  min_child_weight = 6)
    
    train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                            content_type='text/csv', s3_data_type='S3Prefix')
    validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                                 content_type='text/csv', s3_data_type='S3Prefix')
    
    data_channels = {'train': train_data, 'validation': validation_data}

Train the XGBoost model
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    xgb_model.fit(inputs=data_channels, logs=True)

Summary
~~~~~~~

Voila! You completed the first portion of the machine learning pipeline
using Amazon SageMaker Processing for feature transformation and Amazon
SageMaker XGBoost for training a regression model.
