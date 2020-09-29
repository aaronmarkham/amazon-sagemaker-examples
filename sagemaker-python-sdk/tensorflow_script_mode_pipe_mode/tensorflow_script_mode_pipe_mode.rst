TensorFlow Script Mode with Pipe Mode Input
===========================================

SageMaker Pipe Mode is an input mechanism for SageMaker training
containers based on Linux named pipes. SageMaker makes the data
available to the training container using named pipes, which allows data
to be downloaded from S3 to the container while training is running. For
larger datasets, this dramatically improves the time to start training,
as the data does not need to be first downloaded to the container. To
learn more about pipe mode, please consult the AWS documentation at:
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-trainingdata.

In this tutorial, we show you how to train a TensorFlow estimator using
data read with SageMaker Pipe Mode. We use the SageMaker PipeModeDataset
class - a special TensorFlow Dataset built specifically to read from
SageMaker Pipe Mode data. This Dataset is available in our TensorFlow
containers for TensorFlow versions 1.7.0 and up. It’s also open-sourced
at https://github.com/aws/sagemaker-tensorflow-extensions and can be
built into custom TensorFlow images for use in SageMaker.

Although you can also build the PipeModeDataset into your own
containers, in this tutorial we’ll show how you can use the
PipeModeDataset by launching training from the SageMaker Python SDK. The
SageMaker Python SDK helps you deploy your models for training and
hosting in optimized, production-ready containers in SageMaker. The
SageMaker Python SDK is easy to use, modular, extensible and compatible
with TensorFlow and many other deep learning frameworks.

Different collections of S3 files can be made available to the training
container while it’s running. These are referred to as “channels” in
SageMaker. In this example, we use two channels - one for training data
and one for evaluation data. Each channel is mapped to S3 files from
different directories. The SageMaker PipeModeDataset knows how to read
from the named pipes for each channel given just the channel name. When
we launch SageMaker training we tell SageMaker what channels we have and
where in S3 to read the data for each channel.

Setup
-----

The following code snippet sets up some variables we’ll need later on.

.. code:: ipython3

    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = Session().default_bucket()
    
    # Location to save your custom code in tar.gz format.
    custom_code_upload_location = 's3://{}/tensorflow_scriptmode_pipemode/customcode'.format(bucket)
    
    # Location where results of model training are saved.
    model_artifacts_location = 's3://{}/tensorflow_scriptmode_pipemode/artifacts'.format(bucket)
    
    # IAM execution role that gives SageMaker access to resources in your AWS account.
    role = get_execution_role()


Complete training source code
-----------------------------

In this tutorial we train a TensorFlow LinearClassifier using pipe mode
data. The TensorFlow training script is contained in following file:

.. code:: ipython3

    !pygmentize "pipemode.py"

The above script is compatible with the SageMaker TensorFlow script mode
container. (See: `Preparing TensorFlow Training
Script <https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/tensorflow#preparing-a-script-mode-training-script>`__).

Using a ``PipeModeDataset`` to train an estimator using a Pipe Mode
channel, we can construct an function that reads from the channel and
return an ``PipeModeDataset``. This is a TensorFlow Dataset specifically
created to read from a SageMaker Pipe Mode channel. A
``PipeModeDataset`` is a fully-featured TensorFlow Dataset and can be
used in exactly the same ways as a regular TensorFlow Dataset can be
used.

The training and evaluation data used in this tutorial is synthetic. It
contains a series of records stored in a TensorFlow Example protobuf
object. Each record contains a numeric class label and an array of 1024
floating point numbers. Each array is sampled from a multi-dimensional
Gaussian distribution with a class-specific mean. This means it is
possible to learn a model using a TensorFlow Linear classifier which can
classify examples well. Each record is separated using RecordIO encoding
(though the ``PipeModeDataset`` class also supports the TFRecord format
as well).

The training and evaluation data were produced using the benchmarking
source code in the sagemaker-tensorflow-extensions benchmarking
sub-package. If you want to investigate this further, please visit the
GitHub repository for sagemaker-tensorflow-extensions at
https://github.com/aws/sagemaker-tensorflow-extensions.

The following example code shows how to construct a ``PipeModeDataset``.

.. code:: python

   from sagemaker_tensorflow import `PipeModeDataset`


   # Simple example data - a labeled vector.
   features = {
       'data': tf.FixedLenFeature([], tf.string),
       'labels': tf.FixedLenFeature([], tf.int64),
   }

   # A function to parse record bytes to a labeled vector record
   def parse(record):
       parsed = tf.parse_single_example(record, features)
       return ({
           'data': tf.decode_raw(parsed['data'], tf.float64)
       }, parsed['labels'])

   # Construct a `PipeModeDataset` reading from a 'training' channel, using
   # the TF Record encoding.
   ds = `PipeModeDataset`(channel='training', record_format='TFRecord')

   # The `PipeModeDataset` is a TensorFlow Dataset and provides standard Dataset methods
   ds = ds.repeat(20)
   ds = ds.prefetch(10)
   ds = ds.map(parse, num_parallel_calls=10)
   ds = ds.batch(64)

Running training using the Python SDK
=====================================

We can use the SDK to run our local training script on SageMaker
infrastructure.

1. Pass the path to the pipemode.py file, which contains the functions
   for defining your estimator, to the
   ``sagemaker.tensorflow.TensorFlow`` init method.
2. Pass the S3 location that we uploaded our data to previously to the
   ``fit()`` method.

.. code:: ipython3

    from sagemaker.tensorflow import TensorFlow
    
    tensorflow = TensorFlow(entry_point='pipemode.py',
                            role=role,
                            framework_version='1.15.2',
                            input_mode='Pipe',
                            output_path=model_artifacts_location,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            py_version='py3',
                            train_instance_type='ml.c4.xlarge')

After we’ve created the SageMaker Python SDK TensorFlow object, we can
call ``fit()`` to launch TensorFlow training:

.. code:: ipython3

    %%time
    import boto3
    
    # use the region-specific sample data bucket
    region = boto3.Session().region_name
    
    train_data = 's3://sagemaker-sample-data-{}/tensorflow/pipe-mode/train'.format(region)
    eval_data = 's3://sagemaker-sample-data-{}/tensorflow/pipe-mode/eval'.format(region)
    
    tensorflow.fit({'train':train_data, 'eval':eval_data})

After training finishes, the trained model artifacts will be uploaded to
S3. This following example notebook shows how to deploy a model trained
with script mode:
https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/tensorflow_script_mode_training_and_serving
