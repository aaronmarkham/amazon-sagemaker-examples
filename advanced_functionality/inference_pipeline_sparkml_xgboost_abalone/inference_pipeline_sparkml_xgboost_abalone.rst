Feature processing with Spark, training with XGBoost and deploying as Inference Pipeline
========================================================================================

Typically a Machine Learning (ML) process consists of few steps:
gathering data with various ETL jobs, pre-processing the data,
featurizing the dataset by incorporating standard techniques or prior
knowledge, and finally training an ML model using an algorithm.

In many cases, when the trained model is used for processing real time
or batch prediction requests, the model receives data in a format which
needs to pre-processed (e.g. featurized) before it can be passed to the
algorithm. In the following notebook, we will demonstrate how you can
build your ML Pipeline leveraging Spark Feature Transformers and
SageMaker XGBoost algorithm & after the model is trained, deploy the
Pipeline (Feature Transformer and XGBoost) as an Inference Pipeline
behind a single Endpoint for real-time inference and for batch
inferences using Amazon SageMaker Batch Transform.

In this notebook, we use Amazon Glue to run serverless Spark. Though the
notebook demonstrates the end-to-end flow on a small dataset, the setup
can be seamlessly used to scale to larger datasets.

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

We’ll use SparkML to process the dataset (apply one or many feature
transformers) and upload the transformed dataset to S3 so that it can be
used for training with XGBoost.

Methodologies
-------------

The Notebook consists of a few high-level steps:

-  Using AWS Glue for executing the SparkML feature processing job.
-  Using SageMaker XGBoost to train on the processed dataset produced by
   SparkML job.
-  Building an Inference Pipeline consisting of SparkML & XGBoost models
   for a realtime inference endpoint.
-  Building an Inference Pipeline consisting of SparkML & XGBoost models
   for a single Batch Transform job.

Using AWS Glue for executing the SparkML job
============================================

We’ll be running the SparkML job using `AWS
Glue <https://aws.amazon.com/glue>`__. AWS Glue is a serverless ETL
service which can be used to execute standard Spark/PySpark jobs. Glue
currently only supports ``Python 2.7``, hence we’ll write the script in
``Python 2.7``.

Permission setup for invoking AWS Glue from this Notebook
---------------------------------------------------------

In order to enable this Notebook to run AWS Glue jobs, we need to add
one additional permission to the default execution role of this
notebook. We will be using SageMaker Python SDK to retrieve the default
execution role and then you have to go to `IAM
Dashboard <https://console.aws.amazon.com/iam/home>`__ to edit the Role
to add AWS Glue specific permission.

Finding out the current execution role of the Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are using SageMaker Python SDK to retrieve the current role for this
Notebook which needs to be enhanced.

.. code:: ipython2

    # Import SageMaker Python SDK to get the Session and execution_role
    import sagemaker
    from sagemaker import get_execution_role
    sess = sagemaker.Session()
    role = get_execution_role()
    print(role[role.rfind('/') + 1:])

Adding AWS Glue as an additional trusted entity to this role
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is needed if you want to pass the execution role of this
Notebook while calling Glue APIs as well without creating an additional
**Role**. If you have not used AWS Glue before, then this step is
mandatory.

If you have used AWS Glue previously, then you should have an already
existing role that can be used to invoke Glue APIs. In that case, you
can pass that role while calling Glue (later in this notebook) and skip
this next step.

On the IAM dashboard, please click on **Roles** on the left sidenav and
search for this Role. Once the Role appears, click on the Role to go to
its **Summary** page. Click on the **Trust relationships** tab on the
**Summary** page to add AWS Glue as an additional trusted entity.

Click on **Edit trust relationship** and replace the JSON with this
JSON.

::

   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "Service": [
             "sagemaker.amazonaws.com",
             "glue.amazonaws.com"
           ]
         },
         "Action": "sts:AssumeRole"
       }
     ]
   }

Once this is complete, click on **Update Trust Policy** and you are
done.

Downloading dataset and uploading to S3
---------------------------------------

SageMaker team has downloaded the dataset from UCI and uploaded to one
of the S3 buckets in our account. In this Notebook, we will download
from that bucket and upload to your bucket so that AWS Glue can access
the data. The default AWS Glue permissions we just added expects the
data to be present in a bucket with the string ``aws-glue``. Hence,
after we download the dataset, we will create an S3 bucket in your
account with a valid name and then upload the data to S3.

.. code:: ipython2

    !wget https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv

Creating an S3 bucket and uploading this dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we will create an S3 bucket with the ``aws-glue`` string in the
name and upload this data to the S3 bucket. In case you want to use some
existing bucket to run your Spark job via AWS Glue, you can use that
bucket to upload your data provided the ``Role`` has access permission
to upload and download from that bucket.

Once the bucket is created, the following cell would also update the
``abalone.csv`` file downloaded locally to this bucket under the
``input/abalone`` prefix.

.. code:: ipython2

    import boto3
    import botocore
    from botocore.exceptions import ClientError
    
    boto_session = sess.boto_session
    s3 = boto_session.resource('s3')
    account = boto_session.client('sts').get_caller_identity()['Account']
    region = boto_session.region_name
    default_bucket = 'aws-glue-{}-{}'.format(account, region)
    
    try:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=default_bucket)
        else:
            s3.create_bucket(Bucket=default_bucket, CreateBucketConfiguration={'LocationConstraint': region})
    except ClientError as e:
        error_code = e.response['Error']['Code']
        message = e.response['Error']['Message']
        if error_code == 'BucketAlreadyOwnedByYou':
            print ('A bucket with the same name already exists in your account - using the same bucket.')
            pass        
    
    # Uploading the training data to S3
    sess.upload_data(path='abalone.csv', bucket=default_bucket, key_prefix='input/abalone')    

Writing the feature processing script using SparkML
---------------------------------------------------

The code for feature transformation using SparkML can be found in
``abalone_processing.py`` file written in the same directory. You can go
through the code itself to see how it is using standard SparkML
constructs to define the Pipeline for featurizing the data.

Once the Spark ML Pipeline ``fit`` and ``transform`` is done, we are
splitting our dataset into 80-20 train & validation as part of the
script and uploading to S3 so that it can be used with XGBoost for
training.

Serializing the trained Spark ML Model with `MLeap <https://github.com/combust/mleap>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apache Spark is best suited batch processing workloads. In order to use
the Spark ML model we trained for low latency inference, we need to use
the MLeap library to serialize it to an MLeap bundle and later use the
`SageMaker SparkML
Serving <https://github.com/aws/sagemaker-sparkml-serving-container>`__
to perform realtime and batch inference.

By using the ``SerializeToBundle()`` method from MLeap in the script, we
are serializing the ML Pipeline into an MLeap bundle and uploading to S3
in ``tar.gz`` format as SageMaker expects.

Uploading the code and other dependencies to S3 for AWS Glue
------------------------------------------------------------

Unlike SageMaker, in order to run your code in AWS Glue, we do not need
to prepare a Docker image. We can upload the code and dependencies
directly to S3 and pass those locations while invoking the Glue job.

Upload the SparkML script to S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will be uploading the ``abalone_processing.py`` script to S3 now so
that Glue can use it to run the PySpark job. You can replace it with
your own script if needed. If your code has multiple files, you need to
zip those files and upload to S3 instead of uploading a single file like
it’s being done here.

.. code:: ipython2

    script_location = sess.upload_data(path='abalone_processing.py', bucket=default_bucket, key_prefix='codes')

Upload MLeap dependencies to S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For our job, we will also have to pass MLeap dependencies to Glue. MLeap
is an additional library we are using which does not come bundled with
default Spark.

Similar to most of the packages in the Spark ecosystem, MLeap is also
implemented as a Scala package with a front-end wrapper written in
Python so that it can be used from PySpark. We need to make sure that
the MLeap Python library as well as the JAR is available within the Glue
job environment. In the following cell, we will download the MLeap
Python dependency & JAR from a SageMaker hosted bucket and upload to the
S3 bucket we created above in your account.

If you are using some other Python libraries like ``nltk`` in your code,
you need to download the wheel file from PyPI and upload to S3 in the
same way. At this point, Glue only supports passing pure Python
libraries in this way (e.g. you can not pass ``Pandas`` or ``OpenCV``).
However you can use ``NumPy`` & ``SciPy`` without having to pass these
as packages because these are pre-installed in the Glue environment.

.. code:: ipython2

    !wget https://s3-us-west-2.amazonaws.com/sparkml-mleap/0.9.6/python/python.zip
    !wget https://s3-us-west-2.amazonaws.com/sparkml-mleap/0.9.6/jar/mleap_spark_assembly.jar    

.. code:: ipython2

    python_dep_location = sess.upload_data(path='python.zip', bucket=default_bucket, key_prefix='dependencies/python')
    jar_dep_location = sess.upload_data(path='mleap_spark_assembly.jar', bucket=default_bucket, key_prefix='dependencies/jar')

Defining output locations for the data and model
------------------------------------------------

Next we define the output location where the transformed dataset should
be uploaded. We are also specifying a model location where the MLeap
serialized model would be updated. This locations should be consumed as
part of the Spark script using ``getResolvedOptions`` method of AWS Glue
library (see ``abalone_processing.py`` for details).

By designing our code in that way, we can re-use these variables as part
of other SageMaker operations from this Notebook (details below).

.. code:: ipython2

    from time import gmtime, strftime
    import time
    
    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    
    # Input location of the data, We uploaded our train.csv file to input key previously
    s3_input_bucket = default_bucket
    s3_input_key_prefix = 'input/abalone'
    
    # Output location of the data. The input data will be split, transformed, and 
    # uploaded to output/train and output/validation
    s3_output_bucket = default_bucket
    s3_output_key_prefix = timestamp_prefix + '/abalone'
    
    # the MLeap serialized SparkML model will be uploaded to output/mleap
    s3_model_bucket = default_bucket
    s3_model_key_prefix = s3_output_key_prefix + '/mleap'

Calling Glue APIs
~~~~~~~~~~~~~~~~~

Next we’ll be creating Glue client via Boto so that we can invoke the
``create_job`` API of Glue. ``create_job`` API will create a job
definition which can be used to execute your jobs in Glue. The job
definition created here is mutable. While creating the job, we are also
passing the code location as well as the dependencies location to Glue.

``AllocatedCapacity`` parameter controls the hardware resources that
Glue will use to execute this job. It is measures in units of ``DPU``.
For more information on ``DPU``, please see
`here <https://docs.aws.amazon.com/glue/latest/dg/add-job.html>`__.

.. code:: ipython2

    glue_client = boto_session.client('glue')
    job_name = 'sparkml-abalone-' + timestamp_prefix
    response = glue_client.create_job(
        Name=job_name,
        Description='PySpark job to featurize the Abalone dataset',
        Role=role, # you can pass your existing AWS Glue role here if you have used Glue before
        ExecutionProperty={
            'MaxConcurrentRuns': 1
        },
        Command={
            'Name': 'glueetl',
            'ScriptLocation': script_location
        },
        DefaultArguments={
            '--job-language': 'python',
            '--extra-jars' : jar_dep_location,
            '--extra-py-files': python_dep_location
        },
        AllocatedCapacity=5,
        Timeout=60,
    )
    glue_job_name = response['Name']
    print(glue_job_name)

The aforementioned job will be executed now by calling ``start_job_run``
API. This API creates an immutable run/execution corresponding to the
job definition created above. We will require the ``job_run_id`` for the
particular job execution to check for status. We’ll pass the data and
model locations as part of the job execution parameters.

.. code:: ipython2

    job_run_id = glue_client.start_job_run(JobName=job_name,
                                           Arguments = {
                                            '--S3_INPUT_BUCKET': s3_input_bucket,
                                            '--S3_INPUT_KEY_PREFIX': s3_input_key_prefix,
                                            '--S3_OUTPUT_BUCKET': s3_output_bucket,
                                            '--S3_OUTPUT_KEY_PREFIX': s3_output_key_prefix,
                                            '--S3_MODEL_BUCKET': s3_model_bucket,
                                            '--S3_MODEL_KEY_PREFIX': s3_model_key_prefix
                                           })['JobRunId']
    print(job_run_id)

Checking Glue job status
~~~~~~~~~~~~~~~~~~~~~~~~

Now we will check for the job status to see if it has ``succeeded``,
``failed`` or ``stopped``. Once the job is succeeded, we have the
transformed data into S3 in CSV format which we can use with XGBoost for
training. If the job fails, you can go to `AWS Glue
console <https://us-west-2.console.aws.amazon.com/glue/home>`__, click
on **Jobs** tab on the left, and from the page, click on this particular
job and you will be able to find the CloudWatch logs (the link under
**Logs**) link for these jobs which can help you to see what exactly
went wrong in the job execution.

.. code:: ipython2

    job_run_status = glue_client.get_job_run(JobName=job_name,RunId=job_run_id)['JobRun']['JobRunState']
    while job_run_status not in ('FAILED', 'SUCCEEDED', 'STOPPED'):
        job_run_status = glue_client.get_job_run(JobName=job_name,RunId=job_run_id)['JobRun']['JobRunState']
        print (job_run_status)
        time.sleep(30)

Using SageMaker XGBoost to train on the processed dataset produced by SparkML job
---------------------------------------------------------------------------------

Now we will use SageMaker XGBoost algorithm to train on this dataset. We
already know the S3 location where the preprocessed training data was
uploaded as part of the Glue job.

We need to retrieve the XGBoost algorithm image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will retrieve the XGBoost built-in algorithm image so that it can
leveraged for the training job.

.. code:: ipython2

    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    training_image = get_image_uri(sess.boto_region_name, 'xgboost', repo_version="latest")
    print (training_image)

Next XGBoost model parameters and dataset details will be set properly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have parameterized this Notebook so that the same data location which
was used in the PySpark script can now be passed to XGBoost Estimator as
well.

.. code:: ipython2

    s3_train_data = 's3://{}/{}/{}'.format(s3_output_bucket, s3_output_key_prefix, 'train')
    s3_validation_data = 's3://{}/{}/{}'.format(s3_output_bucket, s3_output_key_prefix, 'validation')
    s3_output_location = 's3://{}/{}/{}'.format(s3_output_bucket, s3_output_key_prefix, 'xgboost_model')
    
    xgb_model = sagemaker.estimator.Estimator(training_image,
                                             role, 
                                             train_instance_count=1, 
                                             train_instance_type='ml.m5.xlarge',
                                             train_volume_size = 20,
                                             train_max_run = 3600,
                                             input_mode= 'File',
                                             output_path=s3_output_location,
                                             sagemaker_session=sess)
    
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

Finally XGBoost training will be performed.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    xgb_model.fit(inputs=data_channels, logs=True)

Building an Inference Pipeline consisting of SparkML & XGBoost models for a realtime inference endpoint
=======================================================================================================

Next we will proceed with deploying the models in SageMaker to create an
Inference Pipeline. You can create an Inference Pipeline with upto five
containers.

Deploying a model in SageMaker requires two components:

-  Docker image residing in ECR.
-  Model artifacts residing in S3.

**SparkML**

For SparkML, Docker image for MLeap based SparkML serving is provided by
SageMaker team. For more information on this, please see `SageMaker
SparkML
Serving <https://github.com/aws/sagemaker-sparkml-serving-container>`__.
MLeap serialized SparkML model was uploaded to S3 as part of the SparkML
job we executed in AWS Glue.

**XGBoost**

For XGBoost, we will use the same Docker image we used for training. The
model artifacts for XGBoost was uploaded as part of the training job we
just ran.

Passing the schema of the payload via environment variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SparkML serving container needs to know the schema of the request
that’ll be passed to it while calling the ``predict`` method. In order
to alleviate the pain of not having to pass the schema with every
request, ``sagemaker-sparkml-serving`` allows you to pass it via an
environment variable while creating the model definitions. This schema
definition will be required in our next step for creating a model.

We will see later that you can overwrite this schema on a per request
basis by passing it as part of the individual request payload as well.

.. code:: ipython2

    import json
    schema = {
        "input": [
            {
                "name": "sex",
                "type": "string"
            }, 
            {
                "name": "length",
                "type": "double"
            }, 
            {
                "name": "diameter",
                "type": "double"
            }, 
            {
                "name": "height",
                "type": "double"
            }, 
            {
                "name": "whole_weight",
                "type": "double"
            }, 
            {
                "name": "shucked_weight",
                "type": "double"
            },
            {
                "name": "viscera_weight",
                "type": "double"
            }, 
            {
                "name": "shell_weight",
                "type": "double"
            }, 
        ],
        "output": 
            {
                "name": "features",
                "type": "double",
                "struct": "vector"
            }
    }
    schema_json = json.dumps(schema)
    print(schema_json)

Creating a ``PipelineModel`` which comprises of the SparkML and XGBoost model in the right order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we’ll create a SageMaker ``PipelineModel`` with SparkML and
XGBoost.The ``PipelineModel`` will ensure that both the containers get
deployed behind a single API endpoint in the correct order. The same
model would later be used for Batch Transform as well to ensure that a
single job is sufficient to do prediction against the Pipeline.

Here, during the ``Model`` creation for SparkML, we will pass the schema
definition that we built in the previous cell.

.. code:: ipython2

    from sagemaker.model import Model
    from sagemaker.pipeline import PipelineModel
    from sagemaker.sparkml.model import SparkMLModel
    
    sparkml_data = 's3://{}/{}/{}'.format(s3_model_bucket, s3_model_key_prefix, 'model.tar.gz')
    # passing the schema defined above by using an environment variable that sagemaker-sparkml-serving understands
    sparkml_model = SparkMLModel(model_data=sparkml_data, env={'SAGEMAKER_SPARKML_SCHEMA' : schema_json})
    xgb_model = Model(model_data=xgb_model.model_data, image=training_image)
    
    model_name = 'inference-pipeline-' + timestamp_prefix
    sm_model = PipelineModel(name=model_name, role=role, models=[sparkml_model, xgb_model])

Deploying the ``PipelineModel`` to an endpoint for realtime inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we will deploy the model we just created with the ``deploy()``
method to start an inference endpoint and we will send some requests to
the endpoint to verify that it works as expected.

.. code:: ipython2

    endpoint_name = 'inference-pipeline-ep-' + timestamp_prefix
    sm_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

Invoking the newly created inference endpoint with a payload to transform the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we will invoke the endpoint with a valid payload that SageMaker
SparkML Serving can recognize. There are three ways in which input
payload can be passed to the request:

-  Pass it as a valid CSV string. In this case, the schema passed via
   the environment variable will be used to determine the schema. For
   CSV format, every column in the input has to be a basic datatype
   (e.g. int, double, string) and it can not be a Spark ``Array`` or
   ``Vector``.

-  Pass it as a valid JSON string. In this case as well, the schema
   passed via the environment variable will be used to infer the schema.
   With JSON format, every column in the input can be a basic datatype
   or a Spark ``Vector`` or ``Array`` provided that the corresponding
   entry in the schema mentions the correct value.

-  Pass the request in JSON format along with the schema and the data.
   In this case, the schema passed in the payload will take precedence
   over the one passed via the environment variable (if any).

Passing the payload in CSV format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will first see how the payload can be passed to the endpoint in CSV
format.

.. code:: ipython2

    from sagemaker.predictor import json_serializer, csv_serializer, json_deserializer, RealTimePredictor
    from sagemaker.content_types import CONTENT_TYPE_CSV, CONTENT_TYPE_JSON
    payload = "F,0.515,0.425,0.14,0.766,0.304,0.1725,0.255"
    predictor = RealTimePredictor(endpoint=endpoint_name, sagemaker_session=sess, serializer=csv_serializer,
                                    content_type=CONTENT_TYPE_CSV, accept=CONTENT_TYPE_CSV)
    print(predictor.predict(payload))

Passing the payload in JSON format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will now pass a different payload in JSON format.

.. code:: ipython2

    payload = {"data": ["F",0.515,0.425,0.14,0.766,0.304,0.1725,0.255]}
    predictor = RealTimePredictor(endpoint=endpoint_name, sagemaker_session=sess, serializer=json_serializer,
                                    content_type=CONTENT_TYPE_JSON, accept=CONTENT_TYPE_CSV)
    
    print(predictor.predict(payload))

[Optional] Passing the payload with both schema and the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we will pass the input payload comprising of both the schema and
the data. If you notice carefully, this schema will be slightly
different than what we have passed via the environment variable. The
locations of ``length`` and ``sex`` column have been swapped and so the
data. The server now parses the payload with this schema and works
properly.

.. code:: ipython2

    payload = {
        "schema": {
            "input": [
            {
                "name": "length",
                "type": "double"
            }, 
            {
                "name": "sex",
                "type": "string"
            }, 
            {
                "name": "diameter",
                "type": "double"
            }, 
            {
                "name": "height",
                "type": "double"
            }, 
            {
                "name": "whole_weight",
                "type": "double"
            }, 
            {
                "name": "shucked_weight",
                "type": "double"
            },
            {
                "name": "viscera_weight",
                "type": "double"
            }, 
            {
                "name": "shell_weight",
                "type": "double"
            }, 
        ],
        "output": 
            {
                "name": "features",
                "type": "double",
                "struct": "vector"
            }
        },
        "data": [0.515,"F",0.425,0.14,0.766,0.304,0.1725,0.255]
    }
    
    predictor = RealTimePredictor(endpoint=endpoint_name, sagemaker_session=sess, serializer=json_serializer,
                                    content_type=CONTENT_TYPE_JSON, accept=CONTENT_TYPE_CSV)
    
    print(predictor.predict(payload))

[Optional] Deleting the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not plan to use this endpoint, then it is a good practice to
delete the endpoint so that you do not incur the cost of running it.

.. code:: ipython2

    sm_client = boto_session.client('sagemaker')
    sm_client.delete_endpoint(EndpointName=endpoint_name)

Building an Inference Pipeline consisting of SparkML & XGBoost models for a single Batch Transform job
======================================================================================================

SageMaker Batch Transform also supports chaining multiple containers
together when deploying an Inference Pipeline and performing a single
batch transform jobs to transform your data for a batch use-case similar
to the real-time use-case we have seen above.

Preparing data for Batch Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch Transform requires data in the same format described above, with
one CSV or JSON being per line. For this Notebook, SageMaker team has
created a sample input in CSV format which Batch Transform can process.
The input is basically a similar CSV file to the training file with only
difference is that it does not contain the label (``rings``) field.

Next we will download a sample of this data from one of the SageMaker
buckets (named ``batch_input_abalone.csv``) and upload to your S3
bucket. We will also inspect first five rows of the data post
downloading.

.. code:: ipython2

    !wget https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/batch_input_abalone.csv
    !printf "\n\nShowing first five lines\n\n"    
    !head -n 5 batch_input_abalone.csv 
    !printf "\n\nAs we can see, it is identical to the training file apart from the label being absent here.\n\n"  

.. code:: ipython2

    batch_input_loc = sess.upload_data(path='batch_input_abalone.csv', bucket=default_bucket, key_prefix='batch')

Invoking the Transform API to create a Batch Transform job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we will create a Batch Transform job using the ``Transformer``
class from Python SDK to create a Batch Transform job.

.. code:: ipython2

    input_data_path = 's3://{}/{}/{}'.format(default_bucket, 'batch', 'batch_input_abalone.csv')
    output_data_path = 's3://{}/{}/{}'.format(default_bucket, 'batch_output/abalone', timestamp_prefix)
    job_name = 'serial-inference-batch-' + timestamp_prefix
    transformer = sagemaker.transformer.Transformer(
        # This was the model created using PipelineModel and it contains feature processing and XGBoost
        model_name = model_name,
        instance_count = 1,
        instance_type = 'ml.m5.xlarge',
        strategy = 'SingleRecord',
        assemble_with = 'Line',
        output_path = output_data_path,
        base_transform_job_name='serial-inference-batch',
        sagemaker_session=sess,
        accept = CONTENT_TYPE_CSV
    )
    transformer.transform(data = input_data_path,
                          job_name = job_name,
                          content_type = CONTENT_TYPE_CSV, 
                          split_type = 'Line')
    transformer.wait()
