Compile and Deploy a TensorFlow model on Inf1 instances
=======================================================

Amazon SageMaker now supports Inf1 instances for high performance and
cost-effective inferences. Inf1 instances are ideal for large scale
machine learning inference applications like image recognition, speech
recognition, natural language processing, personalization, and fraud
detection. In this example, we train a classification model on the MNIST
dataset using TensorFlow, compile it using Amazon SageMaker Neo, and
deploy the model on Inf1 instances on a SageMaker endpoint and use the
Neo Deep Learning Runtime to make inferences in real-time and with low
latency.

Inf 1 instances
~~~~~~~~~~~~~~~

Inf1 instances are built from the ground up to support machine learning
inference applications and feature up to 16 AWS Inferentia chips,
high-performance machine learning inference chips designed and built by
AWS. The Inferentia chips are coupled with the latest custom 2nd
generation Intel® Xeon® Scalable processors and up to 100 Gbps
networking to enable high throughput inference. With 1 to 16 AWS
Inferentia chips per instance, Inf1 instances can scale in performance
to up to 2000 Tera Operations per Second (TOPS) and deliver extremely
low latency for real-time inference applications. The large on-chip
memory on AWS Inferentia chips used in Inf1 instances allows caching of
machine learning models directly on the chip. This eliminates the need
to access outside memory resources during inference, enabling low
latency without impacting bandwidth.

Set up the environment
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import os
    import sagemaker
    from sagemaker import get_execution_role
    import boto3
    
    sagemaker_session = sagemaker.Session()
    
    role = get_execution_role()

Download the MNIST dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import utils
    from tensorflow.contrib.learn.python.learn.datasets import mnist
    import tensorflow as tf
    
    data_sets = mnist.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)
    
    utils.convert_to(data_sets.train, 'train', 'data')
    utils.convert_to(data_sets.validation, 'validation', 'data')
    utils.convert_to(data_sets.test, 'test', 'data')

Upload the data
~~~~~~~~~~~~~~~

We use the ``sagemaker.Session.upload_data`` function to upload our
datasets to an S3 location. The return value inputs identifies the
location – we will use this later when we start the training job.

.. code:: ipython3

    inputs = sagemaker_session.upload_data(path='data', key_prefix='data/DEMO-mnist')

Construct a script for distributed training
===========================================

Here is the full code for the network model:

.. code:: ipython3

    !cat 'mnist.py'

The script here is and adaptation of the `TensorFlow MNIST
example <https://github.com/tensorflow/models/tree/master/official/vision/image_classification>`__.
It provides a ``model_fn(features, labels, mode)``, which is used for
training, evaluation and inference. See [TensorFlow MNIST distributed
training notebook](The script here is and adaptation of the `TensorFlow
MNIST
example <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist>`__.
It provides a ``model_fn(features, labels, mode)``, which is used for
training, evaluation and inference. See `TensorFlow MNIST distributed
training
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_script_mode_training_and_serving/tensorflow_script_mode_training_and_serving.ipynb>`__
for more details about the training script.

At the end of the training script, there are two additional functions,
to be used with Neo Deep Learning Runtime: \*
``neo_preprocess(payload, content_type)``: Function that takes in the
payload and Content-Type of each incoming request and returns a NumPy
array \* ``neo_postprocess(result)``: Function that takes the prediction
results produced by Deep Learining Runtime and returns the response body

LeCun, Y., Cortes, C., & Burges, C. (2010). MNIST handwritten digit
databaseATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist,
2.

Create a training job using the sagemaker.TensorFlow estimator
--------------------------------------------------------------

.. code:: ipython3

    from sagemaker.tensorflow import TensorFlow
    
    mnist_estimator = TensorFlow(entry_point='mnist.py',
                                 role=role,
                                 framework_version='1.11.0',
                                 training_steps=1000, 
                                 evaluation_steps=100,
                                 train_instance_count=2,
                                 train_instance_type='ml.c5.xlarge',
                                 sagemaker_session=sagemaker_session)
    
    mnist_estimator.fit(inputs)

The **``fit``** method will create a training job in two
**ml.c5.xlarge** instances. The logs above will show the instances doing
training, evaluation, and incrementing the number of **training steps**.

In the end of the training, the training job will generate a saved model
for compilation.

Deploy the trained model on Inf1 instance for real-time inferences
==================================================================

Once the training is complete, we compile the model using Amazon
SageMaker Neo to optize performance for our desired deployment target.
Amazon SageMaker Neo enables you to train machine learning models once
and run them anywhere in the cloud and at the edge. To compile our
trained model for deploying on Inf1 instances, we are using the
``TensorFlowEstimator.compile_model`` method and select ``'ml_inf1'`` as
our deployment target. The compiled model will then be deployed on an
endpoint using Inf1 instances in Amazon SageMaker.

Compile the model
-----------------

The ``input_shape`` is the definition for the model’s input tensor and
``output_path`` is where the compiled model will be stored in S3.
**Important. If the following command result in a permission error,
scroll up and locate the value of execution role returned by
``get_execution_role()``. The role must have access to the S3 bucket
specified in ``output_path``.**

.. code:: ipython3

    output_path = '/'.join(mnist_estimator.output_path.split('/')[:-1])
    mnist_estimator.framework_version='1.15.0'
    
    optimized_estimator = mnist_estimator.compile_model(target_instance_family='ml_inf1', 
                                  input_shape={'data':[1, 784]},  # Batch size 1, 3 channels, 224x224 Images.
                                  output_path=output_path,
                                  framework='tensorflow', framework_version='1.15.0')

Deploy the compiled model on a SageMaker endpoint
-------------------------------------------------

Now that we have the compiled model, we will deploy it on an Amazon
SageMaker endpoint. Inf1 instances in Amazon SageMaker are available in
four sizes: ml.inf1.xlarge, ml.inf1.2xlarge, ml.inf1.6xlarge, and
ml.inf1.24xlarge. In this example, we are using ``'ml.inf1.xlarge'`` for
deploying our model.

.. code:: ipython3

    optimized_predictor = optimized_estimator.deploy(initial_instance_count = 1,
                                                     instance_type = 'ml.inf1.xlarge')

.. code:: ipython3

    import numpy as np
    def numpy_bytes_serializer(data):
        f = io.BytesIO()
        np.save(f, data)
        f.seek(0)
        return f.read()
    
    optimized_predictor.content_type = 'application/vnd+python.numpy+binary'
    optimized_predictor.serializer = numpy_bytes_serializer

Invoking the endpoint
---------------------

Once the endpoint is ready, you can send requests to it and receive
inference results in real-time with low latency.

.. code:: ipython3

    from tensorflow.examples.tutorials.mnist import input_data
    from IPython import display
    import PIL.Image
    import io
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    for i in range(10):
        data = mnist.test.images[i]
        # Display image
        im = PIL.Image.fromarray(data.reshape((28,28))*255).convert('L')
        display.display(im)
        # Invoke endpoint with image
        predict_response = optimized_predictor.predict(data)
        
        print("========================================")
        label = np.argmax(mnist.test.labels[i])
        print("label is {}".format(label))
        prediction = predict_response
        print("prediction is {}".format(prediction))

Deleting endpoint
-----------------

Delete the endpoint if you no longer need it.

.. code:: ipython3

    sagemaker_session.delete_endpoint(optimized_predictor.endpoint)

