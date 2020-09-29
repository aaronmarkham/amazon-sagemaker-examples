Working with TFRecord Datasets
==============================

1.  `Introduction <#Introduction>`__
2.  `Prerequisites <#Prerequisites>`__
3.  `Converting a dataset from CSV to
    TFrecords <#Converting-a-dataset-from-CSV-to-TFrecords>`__
4.  `Upload dataset to S3 <#Upload-dataset-to-S3>`__
5.  `Construct a DNNClassifier <#Construct-a-DNNClassifier>`__
6.  `Train a Model <#Train-a-Model>`__
7.  `Run Batch Transform <#Run-Batch-Transform>`__
8.  `Build a container for transforming TFRecord
    input <#Build-a-container-for-transforming-TFRecord-input>`__
9.  `Push container to ECR <#Push-container-to-ECR>`__
10. `Create a model with an inference
    pipeline <#Create-a-model-with-an-inference-pipeline>`__
11. `Run a batch transform job <#Run-a-batch-transform-job>`__
12. `Inspect batch transform output <#Inspect-batch-transform-output>`__

Introduction
------------

TFRecord is a standard TensorFlow data format. It is a record-oriented
binary file format that allows for efficient storage and processing of
large datasets. In this notebook, we’ll demonstrate how to take an
existing CSV dataset and convert it to TFRecord files. We’ll also build
a TensorFlow training script that accepts serialized tf.Example protos
(the payload of our TFRecords) as input during training. Then, we’ll run
a training job using the TFRecord dataset we’ve generated as input.
Finally, we’ll demonstrate how to run a batch transform job with an
inference pipeline so that we can pass the TFRecord dataset as input.

Prerequisites
-------------

Let’s start by specifying: \* The S3 bucket and prefixes you’d like to
use for training and batch transform data. \* The IAM role that will be
used for training and batch transform jobs, as well as ECR repository
creation and image upload.

.. code:: ipython3

    import boto3
    import sagemaker
    import tensorflow as tf
    
    bucket = '<your_bucket_name>'
    training_prefix = 'training'
    batch_input_prefix = 'batch_input'
    batch_output_prefix ='batch_output'
    
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

Converting a dataset from CSV to TFRecords
------------------------------------------

First, we’ll take an existing CSV dataset (located in
``./dataset-csv/``) and convert it to the TFRecords file format:

.. code:: ipython3

    import os
    
    csv_root = './dataset-csv/'
    tfrecord_root = './dataset-tfrecord/'
    test_csv_file = 'iris_test.csv'
    train_csv_file = 'iris_train.csv'
    test_tfrecord_file = 'iris_test.tfrecords'
    train_tfrecord_file = 'iris_train.tfrecords'
    
    def _floatlist_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
    
    def _int64list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    # create the tfrecord dataset dir
    if not os.path.isdir(tfrecord_root):
        os.mkdir(tfrecord_root)
    
    for input_file, output_file in [(test_csv_file,test_tfrecord_file), (train_csv_file,train_tfrecord_file)]:
        # create the output file
        open(tfrecord_root + output_file, 'a').close()
        with tf.python_io.TFRecordWriter(tfrecord_root + output_file) as writer:
            with open(csv_root + input_file,'r') as f:
                f.readline() # skip first line
                for line in f:
                    feature = {
                        'sepal_length': _floatlist_feature(line.split(',')[0]),
                        'sepal_width': _floatlist_feature(line.split(',')[1]),
                        'petal_length': _floatlist_feature(line.split(',')[2]),
                        'petal_width': _floatlist_feature(line.split(',')[3]),
                    }
                    if f == train_csv_file:
                        feature['label'] = _int64list_feature(int(line.split(',')[4].rstrip()))
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=feature
                        )
                    )
                    writer.write(example.SerializeToString())

Upload dataset to S3
~~~~~~~~~~~~~~~~~~~~

Next, we’ll upload the TFRecord datasets to S3 so that we can use it in
training and batch transform jobs.

.. code:: ipython3

    def upload_to_s3(bucket, key, file):
        s3 = boto3.resource('s3')
        data = open(file, "rb")
        s3.Bucket(bucket).put_object(Key=key, Body=data)
        
    upload_to_s3(bucket, training_prefix + '/' + train_tfrecord_file, tfrecord_root + train_tfrecord_file)
    upload_to_s3(bucket, batch_input_prefix + '/' + test_tfrecord_file, tfrecord_root + test_tfrecord_file)

Construct a DNN Classifier
--------------------------

In ``./dnn-classifier/train.py`` we’ve defined a neural network
classifier using TensorFlow’s DNNClassifier. We can take a look at the
train script to see how the network and input functions are defined:

.. code:: ipython3

    !cat ./dnn-classifier/train.py

Train a Model
-------------

Next, we’ll kick off a training job using the training script defined
above.

.. code:: ipython3

    from sagemaker.tensorflow import TensorFlow
    
    train_data_location = 's3://{}/{}'.format(bucket, training_prefix)
    instance_type = 'ml.c4.xlarge'
    
    estimator = TensorFlow(entry_point='train.py',
                           source_dir='dnn-classifier',
                           model_dir='/opt/ml/model',
                           train_instance_type=instance_type,
                           train_instance_count=1,
                           role=sagemaker.get_execution_role(), # Passes to the container the AWS role that you are using on this notebook
                           framework_version='1.11.0', # Uses TensorFlow 1.11
                           py_version='py3',
                           script_mode=True)
    
    inputs = {'training': train_data_location}
    
    estimator.fit(inputs)

Run Batch Transform
-------------------

Build a container for transforming TFRecord input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker TensorFlow Serving container uses the TensorFlow
ModelServer RESTful API to serve predict requests. In the next step,
we’ll create a container to transform mini-batch TFRecord payloads into
JSON objects that can be forwarded to the TensorFlow serving container.
To do this, we’ve created a simple Python Flask app that does the
transformation, the code for this container is available in the
``./tfrecord-transformer-container/`` directory. First, we’ll build the
container:

.. code:: ipython3

    !docker build -t tfrecord-transformer ./tfrecord-transformer-container/

Push container to ECR
~~~~~~~~~~~~~~~~~~~~~

Next, we’ll push the docker container to an ECR repository in your
account. In order to push the container to ECR, the execution role
attached to this notebook should have permissions to create a
repository, set a repository policy, and upload an image.

.. code:: ipython3

    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name
    
    ecr_repository = 'tfrecord-transformer'
    tag = ':latest'
    uri_suffix = 'amazonaws.com'
    if region in ['cn-north-1', 'cn-northwest-1']:
        uri_suffix = 'amazonaws.com.cn'
    transformer_repository_uri = '{}.dkr.ecr.{}.{}/{}'.format(account_id, region, uri_suffix, ecr_repository + tag)
    
    # docker login
    !$(aws ecr get-login --region $region --registry-ids $account_id --no-include-email)
    # create ecr repository
    !aws ecr create-repository --repository-name $ecr_repository
    # attach policy allowing sagemaker to pull this image
    !aws ecr set-repository-policy --repository-name $ecr_repository --policy-text "$( cat ./tfrecord-transformer-container/ecr_policy.json )"
    
    !docker tag {ecr_repository + tag} $transformer_repository_uri
    !docker push $transformer_repository_uri

Create a model with an inference pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we’ll create a SageMaker model with the two containers chained
together (TFRecord transformer -> TensorFlow Serving).

.. code:: ipython3

    from sagemaker.tensorflow.serving import Model
    from sagemaker.utils import name_from_base
    
    client = boto3.client('sagemaker')
    
    model_name = name_from_base('tfrecord-to-tfserving')
    
    transform_container = {
        "Image": transformer_repository_uri
    }
    
    tf_serving_model = Model(model_data=estimator.model_data,
                             role=sagemaker.get_execution_role(),
                             image=estimator.image_name,
                             framework_version=estimator.framework_version,
                             sagemaker_session=estimator.sagemaker_session)
    tf_serving_container = tf_serving_model.prepare_container_def(instance_type)
    
    model_params = {
        "ModelName": model_name,
        "Containers": [
            transform_container,
            tf_serving_container
        ],
        "ExecutionRoleArn": sagemaker.get_execution_role()
    }
    
    client.create_model(**model_params)

Run a batch transform job
~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we’ll run a batch transform job using our inference pipeline
model. We’ll specify ``SplitType=TFRecord`` and
``BatchStrategy=MultiRecord`` to specify that our dataset will be split
by TFRecord boundaries, and multiple records will be batched in a single
request up to the ``MaxPayloadInMB=1`` limit.

.. code:: ipython3

    input_data_path = 's3://{}/{}'.format(bucket, batch_input_prefix)
    output_data_path = 's3://{}/{}'.format(bucket, batch_output_prefix)
    
    transformer = sagemaker.transformer.Transformer(
        model_name = model_name,
        instance_count = 1,
        instance_type = instance_type,
        strategy = 'MultiRecord',
        max_payload = 1,
        output_path = output_data_path,
        assemble_with= 'Line',
        base_transform_job_name='tfrecord-transform',
        sagemaker_session=sess,
    )
    transformer.transform(data = input_data_path,
                          split_type = 'TFRecord')
    transformer.wait()

Inspect batch transform output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we can inspect the output files of our batch transform job to
see the predictions.

.. code:: ipython3

    output_uri = transformer.output_path + '/' + test_tfrecord_file + '.out'
    !aws s3 cp $output_uri -
