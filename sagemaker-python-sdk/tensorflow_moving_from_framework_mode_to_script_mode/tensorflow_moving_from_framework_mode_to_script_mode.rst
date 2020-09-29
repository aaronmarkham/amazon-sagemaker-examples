Migrating scripts from Framework Mode to Script Mode
====================================================

This notebook focus on how to migrate scripts using Framework Mode to
Script Mode. The original notebook using Framework Mode can be find here
https://github.com/awslabs/amazon-sagemaker-examples/blob/4c2a93114104e0b9555d7c10aaab018cac3d7c04/sagemaker-python-sdk/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb

Set up the environment
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    import os
    import subprocess
    import sagemaker
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    
    role = get_execution_role()

Download the MNIST dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    import utils
    from tensorflow.examples.tutorials.mnist import input_data
    import tensorflow as tf
    
    data_sets = input_data.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)
    
    utils.convert_to(data_sets.train, 'train', 'data')
    utils.convert_to(data_sets.validation, 'validation', 'data')
    utils.convert_to(data_sets.test, 'test', 'data')

Upload the data
~~~~~~~~~~~~~~~

We use the ``sagemaker.Session.upload_data`` function to upload our
datasets to an S3 location. The return value inputs identifies the
location – we will use this later when we start the training job.

.. code:: ipython2

    inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')

Construct an entry point script for training
============================================

On this example, we assume that you aready have a Framework Mode
training script named ``mnist.py``:

.. code:: ipython2

    !pygmentize 'mnist.py'

The training script ``mnist.py`` include the Framework Mode functions
``model_fn``, ``train_input_fn``, ``eval_input_fn``, and
``serving_input_fn``. We need to create a entrypoint script that uses
the functions above to create a ``tf.estimator``:

.. code:: ipython2

    %%writefile train.py
    
    import argparse
    # import original framework mode script
    import mnist
    
    import tensorflow as tf
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
    
        # read hyperparameters as script arguments
        parser.add_argument('--training_steps', type=int)
        parser.add_argument('--evaluation_steps', type=int)
    
        args, _ = parser.parse_known_args()
    
        # creates a tf.Estimator using `model_fn` that saves models to /opt/ml/model
        estimator = tf.estimator.Estimator(model_fn=mnist.model_fn, model_dir='/opt/ml/model')
    
    
        # creates parameterless input_fn function required by the estimator
        def input_fn():
            return mnist.train_input_fn(training_dir='/opt/ml/input/data/training', params=None)
    
    
        train_spec = tf.estimator.TrainSpec(input_fn, max_steps=args.training_steps)
    
    
        # creates parameterless serving_input_receiver_fn function required by the exporter
        def serving_input_receiver_fn():
            return mnist.serving_input_fn(params=None)
    
    
        exporter = tf.estimator.LatestExporter('Servo',
                                               serving_input_receiver_fn=serving_input_receiver_fn)
    
    
        # creates parameterless input_fn function required by the evaluation
        def input_fn():
            return mnist.eval_input_fn(training_dir='/opt/ml/input/data/training', params=None)
    
    
        eval_spec = tf.estimator.EvalSpec(input_fn, steps=args.evaluation_steps, exporters=exporter)
        
        # start training and evaluation
        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

Changes in the SageMaker TensorFlow estimator
---------------------------------------------

We need to create a TensorFlow estimator pointing to ``train.py`` as the
entrypoint:

.. code:: ipython2

    from sagemaker.tensorflow import TensorFlow
    
    mnist_estimator = TensorFlow(entry_point='train.py',
                                 dependencies=['mnist.py'],
                                 role=role,
                                 framework_version='1.15.2',
                                 hyperparameters={'training_steps':10, 'evaluation_steps':10},
                                 py_version='py3',
                                 train_instance_count=1,
                                 train_instance_type='local')
    
    mnist_estimator.fit(inputs)

Deploy the trained model to prepare for predictions
===================================================

The deploy() method creates an endpoint (in this case locally) which
serves prediction requests in real-time.

.. code:: ipython2

    mnist_predictor = mnist_estimator.deploy(initial_instance_count=1, instance_type='local')

Invoking the endpoint
=====================

.. code:: ipython2

    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    for i in range(10):
        data = mnist.test.images[i].tolist()
    
        predict_response = mnist_predictor.predict(data)
        
        print("========================================")
        label = np.argmax(mnist.test.labels[i])
        print("label is {}".format(label))
        print("prediction is {}".format(predict_response))

Clean-up
========

Deleting the local endpoint when you’re finished is important since you
can only run one local endpoint at a time.

.. code:: ipython2

    mnist_estimator.delete_endpoint()
