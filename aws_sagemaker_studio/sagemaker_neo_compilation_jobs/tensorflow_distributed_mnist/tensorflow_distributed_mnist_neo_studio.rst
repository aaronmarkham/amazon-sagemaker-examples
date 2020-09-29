TensorFlow BYOM: Train with Custom Training Script, Compile with Neo, and Deploy on SageMaker
=============================================================================================

This notebook can be compared to `TensorFlow MNIST distributed training
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_distributed_mnist.ipynb>`__
in terms of its functionality. We will do the same classification task,
but this time we will compile the trained model using the Neo API
backend, to optimize for our choice of hardware. Finally, we setup a
real-time hosted endpoint in SageMaker for our compiled model using the
Neo Deep Learning Runtime.

Set up the environment
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %cd /root/amazon-sagemaker-examples/aws_sagemaker_studio/sagemaker_neo_compilation_jobs/tensorflow_distributed_mnist

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install tensorflow==1.13.1

.. code:: ipython3

    import os
    import sagemaker
    from sagemaker import get_execution_role
    
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
example <https://github.com/tensorflow/models/tree/master/official/mnist>`__.
It provides a ``model_fn(features, labels, mode)``, which is used for
training, evaluation and inference. See `TensorFlow MNIST distributed
training
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_distributed_mnist.ipynb>`__
for more details about the training script.

At the end of the training script, there are two additional functions,
to be used with Neo Deep Learning Runtime: \*
``neo_preprocess(payload, content_type)``: Function that takes in the
payload and Content-Type of each incoming request and returns a NumPy
array \* ``neo_postprocess(result)``: Function that takes the prediction
results produced by Deep Learining Runtime and returns the response body

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
                                 train_instance_type='ml.c4.xlarge')
    
    mnist_estimator.fit(inputs)

The **``fit``** method will create a training job in two
**ml.c4.xlarge** instances. The logs above will show the instances doing
training, evaluation, and incrementing the number of **training steps**.

In the end of the training, the training job will generate a saved model
for TF serving.

Deploy the trained model to prepare for predictions (the old way)
=================================================================

The deploy() method creates an endpoint which serves prediction requests
in real-time.

.. code:: ipython3

    mnist_predictor = mnist_estimator.deploy(initial_instance_count=1,
                                             instance_type='ml.m4.xlarge')

Invoking the endpoint
---------------------

.. code:: ipython3

    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    for i in range(10):
        data = mnist.test.images[i].tolist()
        tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)
        predict_response = mnist_predictor.predict(tensor_proto)
        
        print("========================================")
        label = np.argmax(mnist.test.labels[i])
        print("label is {}".format(label))
        prediction = np.argmax(predict_response['outputs']['probabilities']['float_val'])
        print("prediction is {}".format(prediction))

Deleting the endpoint
---------------------

.. code:: ipython3

    sagemaker.Session().delete_endpoint(mnist_predictor.endpoint)

Deploy the trained model using Neo
==================================

Now the model is ready to be compiled by Neo to be optimized for our
hardware of choice. We are using the
``TensorFlowEstimator.compile_model`` method to do this. For this
example, our target hardware is ``'ml_c5'``. You can changed these to
other supported target hardware if you prefer.

Compiling the model
-------------------

The ``input_shape`` is the definition for the model’s input tensor and
``output_path`` is where the compiled model will be stored in S3.
**Important. If the following command result in a permission error,
scroll up and locate the value of execution role returned by
``get_execution_role()``. The role must have access to the S3 bucket
specified in ``output_path``.**

.. code:: ipython3

    output_path = '/'.join(mnist_estimator.output_path.split('/')[:-1])
    optimized_estimator = mnist_estimator.compile_model(target_instance_family='ml_c5', 
                                  input_shape={'data':[1, 784]},  # Batch size 1, 3 channels, 28x28 Images.
                                  output_path=output_path,
                                  framework='tensorflow', framework_version='1.11.0')

Deploying the compiled model
----------------------------

.. code:: ipython3

    optimized_predictor = optimized_estimator.deploy(initial_instance_count = 1,
                                                     instance_type = 'ml.c5.4xlarge')

.. code:: ipython3

    def numpy_bytes_serializer(data):
        f = io.BytesIO()
        np.save(f, data)
        f.seek(0)
        return f.read()
    
    optimized_predictor.content_type = 'application/vnd+python.numpy+binary'
    optimized_predictor.serializer = numpy_bytes_serializer

Invoking the endpoint
---------------------

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
        print("prediction is {}".format(np.argmax(prediction)))

Deleting endpoint
-----------------

.. code:: ipython3

    sagemaker.Session().delete_endpoint(optimized_predictor.endpoint)
