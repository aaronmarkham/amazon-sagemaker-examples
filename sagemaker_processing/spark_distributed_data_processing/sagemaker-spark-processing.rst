Distributed Data Processing using Apache Spark and SageMaker Processing
=======================================================================

Apache Spark is a unified analytics engine for large-scale data
processing. The Spark framework is often used within the context of
machine learning workflows to run data transformation or feature
engineering workloads at scale. Amazon SageMaker provides a set of
prebuilt Docker images that include Apache Spark and other dependencies
needed to run distributed data processing jobs on Amazon SageMaker. This
example notebook demonstrates how to use the prebuilt Spark images on
SageMaker Processing using the SageMaker Python SDK.

This notebook walks through the following scenarios to illustrate the
functionality of the SageMaker Spark Container:

-  Running a basic PySpark application using the SageMaker Python SDK’s
   ``PySparkProcessor`` class
-  Viewing the Spark UI via the ``start_history_server()`` function of a
   ``PySparkProcessor`` object
-  Adding additional python and jar file dependencies to jobs
-  Running a basic Java/Scala-based Spark job using the SageMaker Python
   SDK’s ``SparkJarProcessor`` class
-  Specifying additional Spark configuration

Setup
-----

Install the latest SageMaker Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This notebook requires the latest v2.x version of the SageMaker Python
SDK. First, ensure that the latest version is installed.

.. code:: ipython3

    !pip install -U "sagemaker>2.0"

*Restart your notebook kernel after upgrading the SDK*

Example 1: Running a basic PySpark application
----------------------------------------------

The first example is a basic Spark MLlib data processing script. This
script will take a raw data set and do some transformations on it such
as string indexing and one hot encoding.

Setup S3 bucket locations and roles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, setup some locations in the default SageMaker bucket to store the
raw input datasets and the Spark job output. Here, you’ll also define
the role that will be used to execute all SageMaker Processing jobs.

.. code:: ipython3

    import logging
    import sagemaker
    from time import gmtime, strftime
    
    sagemaker_logger = logging.getLogger("sagemaker")
    sagemaker_logger.setLevel(logging.INFO)
    sagemaker_logger.addHandler(logging.StreamHandler())
    
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    role = sagemaker.get_execution_role()

Next, you’ll download the example dataset from a SageMaker staging
bucket.

.. code:: ipython3

    # Fetch the dataset from the SageMaker bucket
    !wget https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv -O ./data/abalone.csv

Write the PySpark script
~~~~~~~~~~~~~~~~~~~~~~~~

The source for a preprocessing script is in the cell below. The cell
uses the ``%%writefile`` directive to save this file locally. This
script does some basic feature engineering on a raw input dataset. In
this example, the dataset is the `Abalone Data
Set <https://archive.ics.uci.edu/ml/datasets/abalone>`__ and the code
below performs string indexing, one hot encoding, vector assembly, and
combines them into a pipeline to perform these transformations in order.
The script then does an 80-20 split to produce training and validation
datasets as output.

.. code:: ipython3

    %%writefile ./code/preprocess.py
    from __future__ import print_function
    from __future__ import unicode_literals
    
    import argparse
    import csv
    import os
    import shutil
    import sys
    import time
    
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        OneHotEncoder,
        StringIndexer,
        VectorAssembler,
        VectorIndexer,
    )
    from pyspark.sql.functions import *
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
    )
    
    
    def csv_line(data):
        r = ','.join(str(d) for d in data[1])
        return str(data[0]) + "," + r
    
    
    def main():
        parser = argparse.ArgumentParser(description="app inputs and outputs")
        parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
        parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
        parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
        parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
        args = parser.parse_args()
    
        spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    
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
        total_df = spark.read.csv(('s3://' + os.path.join(args.s3_input_bucket, args.s3_input_key_prefix,
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
        train_lines.saveAsTextFile('s3://' + os.path.join(args.s3_output_bucket, args.s3_output_key_prefix, 'train'))
        
        # Convert the validation dataframe to RDD to save in CSV format and upload to S3
        validation_rdd = validation_df.rdd.map(lambda x: (x.rings, x.features))
        validation_lines = validation_rdd.map(csv_line)
        validation_lines.saveAsTextFile('s3://' + os.path.join(args.s3_output_bucket, args.s3_output_key_prefix, 'validation'))
    
    
    if __name__ == "__main__":
        main()

Run the SageMaker Processing Job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, you’ll use the ``PySparkProcessor`` class to define a Spark job
and run it using SageMaker Processing. A few things to note in the
definition of the ``PySparkProcessor``:

-  This is a multi-node job with 2x m5.xlarge instances (which is
   specified via the ``instance_count`` and ``instance_type``
   parameters)
-  Spark framework version 2.4 is specified via the
   ``framework_version`` parameter
-  The PySpark script defined above is passed via via the ``submit_app``
   parameter
-  Command-line arguments to the PySpark script (such as the s3 input
   and output locations) are passed via the ``arguments`` parameter
-  Spark event logs will be offloaded to the s3 location specified in
   ``spark_event_logs_s3_uri`` and can be used to view the Spark UI
   while the job is in progress or after it completes

.. code:: ipython3

    from sagemaker.spark.processing import PySparkProcessor
    
    # Upload the raw input dataset to a unique S3 location
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    prefix = "sagemaker/spark-preprocess-demo/{}".format(timestamp_prefix)
    input_prefix_abalone = "{}/input/raw/abalone".format(prefix)
    input_preprocessed_prefix_abalone = "{}/input/preprocessed/abalone".format(prefix)
    
    sagemaker_session.upload_data(path='./data/abalone.csv', bucket=bucket, key_prefix=input_prefix_abalone)
    
    # Run the processing job
    spark_processor = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version="2.4",
        role=role,
        instance_count=2,
        instance_type="ml.m5.xlarge",
        max_runtime_in_seconds=1200,
    )
    
    spark_processor.run(
        submit_app="./code/preprocess.py",
        arguments=["--s3_input_bucket", bucket,
                   "--s3_input_key_prefix", input_prefix_abalone,
                   "--s3_output_bucket", bucket,
                   "--s3_output_key_prefix", input_preprocessed_prefix_abalone],
        spark_event_logs_s3_uri="s3://{}/{}/spark_event_logs".format(bucket, prefix),
        logs=False
    )

Validate Data Processing Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, validate the output of our data preprocessing job by looking at
the first 5 rows of the output dataset.

.. code:: ipython3

    print("Top 5 rows from s3://{}/{}/train/".format(bucket, input_preprocessed_prefix_abalone))
    !aws s3 cp --quiet s3://$bucket/$input_preprocessed_prefix_abalone/train/part-00000 - | head -n5

View the Spark UI
~~~~~~~~~~~~~~~~~

Next, you can view the Spark UI by running the history server locally in
this notebook. (**Note:** this feature will only work in a local
development environment with docker installed or on a Sagemaker Notebook
Instance. This feature does not currently work in SageMaker Studio
Notebooks.)

.. code:: ipython3

    spark_processor.start_history_server()

After viewing the Spark UI, you can terminate the history server before
proceeding.

.. code:: ipython3

    spark_processor.terminate_history_server()     

Example 2: Specify additional python and jar file dependencies
--------------------------------------------------------------

The next example demonstrates a scenario where additional python file
dependencies are required by the PySpark script. You’ll use a sample
PySpark script that requires additional user-defined functions (UDFs)
defined in a local module.

.. code:: ipython3

    %%writefile ./code/hello_py_spark_app.py
    import argparse
    import time
    
    # Import local module to test spark-submit--py-files dependencies
    import hello_py_spark_udfs as udfs
    from pyspark.sql import SparkSession, SQLContext
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType
    import time
    
    if __name__ == "__main__":
        print("Hello World, this is PySpark!")
    
        parser = argparse.ArgumentParser(description="inputs and outputs")
        parser.add_argument("--input", type=str, help="path to input data")
        parser.add_argument("--output", required=False, type=str, help="path to output data")
        args = parser.parse_args()
        spark = SparkSession.builder.appName("SparkTestApp").getOrCreate()
        sqlContext = SQLContext(spark.sparkContext)
    
        # Load test data set
        inputPath = args.input
        outputPath = args.output
        salesDF = spark.read.json(inputPath)
        salesDF.printSchema()
    
        salesDF.createOrReplaceTempView("sales")
    
        # Define a UDF that doubles an integer column
        # The UDF function is imported from local module to test spark-submit--py-files dependencies
        double_udf_int = udf(udfs.double_x, IntegerType())
    
        # Save transformed data set to disk
        salesDF.select("date", "sale", double_udf_int("sale").alias("sale_double")).write.json(outputPath)

.. code:: ipython3

    %%writefile ./code/hello_py_spark_udfs.py
    def double_x(x):
        return x + x

Create a processing job with python file dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, you’ll create a processing job where the additional python file
dependencies are specified via the ``submit_py_files`` argument in the
``run()`` function. If your Spark application requires additional jar
file dependencies, these can be specified via the ``submit_jars``
argument of the ``run()`` function.

.. code:: ipython3

    # Define job input/output URIs
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    prefix = "sagemaker/spark-preprocess-demo/{}".format(timestamp_prefix)
    input_prefix_sales = "{}/input/sales".format(prefix)
    output_prefix_sales = "{}/output/sales".format(prefix)
    input_s3_uri = "s3://{}/{}".format(bucket, input_prefix_sales)
    output_s3_uri = "s3://{}/{}".format(bucket, output_prefix_sales)
    
    sagemaker_session.upload_data(path="./data/data.jsonl", bucket=bucket, key_prefix=input_prefix_sales)
    
    spark_processor = PySparkProcessor(
        base_job_name="sm-spark-udfs",
        framework_version="2.4",
        role=role,
        instance_count=2,
        instance_type="ml.m5.xlarge",
        max_runtime_in_seconds=1200,
    )
    
    spark_processor.run(
        submit_app="./code/hello_py_spark_app.py",
        submit_py_files=["./code/hello_py_spark_udfs.py"],
        arguments=["--input", input_s3_uri, "--output", output_s3_uri],
        logs=False
    )

Validate Data Processing Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, validate the output of the Spark job by ensuring that the output
URI contains the Spark ``_SUCCESS`` file along with the output json
lines file.

.. code:: ipython3

    print('Output files in {}'.format(output_s3_uri))
    !aws s3 ls $output_s3_uri/

Example 3: Run a Java/Scala Spark application
---------------------------------------------

In the next example, you’ll take a Spark application jar (located in
``./code/spark-test-app.jar``) that is already built and run it using
SageMaker Processing. Here, you’ll use the ``SparkJarProcessor`` class
to define the job parameters.

In the ``run()`` function you’ll specify:

-  The location of the Spark application jar file in the ``submit_app``
   argument
-  The main class for the Spark application in the ``submit_class``
   argument
-  Input/output arguments for the Spark application

.. code:: ipython3

    from sagemaker.spark.processing import SparkJarProcessor
    
    # Upload the raw input dataset to S3
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    prefix = "sagemaker/spark-preprocess-demo/{}".format(timestamp_prefix)
    input_prefix_sales = "{}/input/sales".format(prefix)
    output_prefix_sales = "{}/output/sales".format(prefix)
    input_s3_uri = "s3://{}/{}".format(bucket, input_prefix_sales)
    output_s3_uri = "s3://{}/{}".format(bucket, output_prefix_sales)
    
    sagemaker_session.upload_data(path="./data/data.jsonl", bucket=bucket, key_prefix=input_prefix_sales)
    
    spark_processor = SparkJarProcessor(
        base_job_name="sm-spark-java",
        framework_version="2.4",
        role=role,
        instance_count=2,
        instance_type="ml.m5.xlarge",
        max_runtime_in_seconds=1200,
    )
    
    spark_processor.run(
        submit_app="./code/spark-test-app.jar",
        submit_class="com.amazonaws.sagemaker.spark.test.HelloJavaSparkApp",
        arguments=["--input", input_s3_uri, "--output", output_s3_uri],
        logs=False
    )

Example 4: Specifying additional Spark configuration
----------------------------------------------------

Overriding Spark configuration is crucial for a number of tasks such as
tuning your Spark application or configuring the hive metastore. Using
the SageMaker Python SDK, you can easily override Spark/Hive/Hadoop
configuration.

An example usage would be overriding Spark executor memory/cores as
demonstrated in the next example.

For more information on configuring your Spark application, see the EMR
documentation on `Configuring
Applications <https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html>`__

.. code:: ipython3

    # Upload the raw input dataset to a unique S3 location
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    prefix = "sagemaker/spark-preprocess-demo/{}".format(timestamp_prefix)
    input_prefix_abalone = "{}/input/raw/abalone".format(prefix)
    input_preprocessed_prefix_abalone = "{}/input/preprocessed/abalone".format(prefix)
    
    sagemaker_session.upload_data(path="./data/abalone.csv", bucket=bucket, key_prefix=input_prefix_abalone)
    
    spark_processor = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version="2.4",
        role=role,
        instance_count=2,
        instance_type="ml.m5.xlarge",
        max_runtime_in_seconds=1200,
    )
    
    configuration = [{
        "Classification": "spark-defaults",
        "Properties": {"spark.executor.memory": "2g", "spark.executor.cores": "1"},
    }]
    
    spark_processor.run(
        submit_app="./code/preprocess.py",
        arguments=["--s3_input_bucket", bucket,
                   "--s3_input_key_prefix", input_prefix_abalone,
                   "--s3_output_bucket", bucket,
                   "--s3_output_key_prefix", input_preprocessed_prefix_abalone],
        configuration=configuration,
        logs=False
    )
