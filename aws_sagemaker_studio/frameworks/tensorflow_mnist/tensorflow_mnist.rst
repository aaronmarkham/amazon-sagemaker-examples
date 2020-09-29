Training and Serving with TensorFlow on Amazon SageMaker
========================================================

*(This notebook was tested with the "Python 3 (Data Science)" kernel.)*

Amazon SageMaker is a fully-managed service that provides developers and
data scientists with the ability to build, train, and deploy machine
learning (ML) models quickly. Amazon SageMaker removes the heavy lifting
from each step of the machine learning process to make it easier to
develop high-quality models. The SageMaker Python SDK makes it easy to
train and deploy models in Amazon SageMaker with several different
machine learning and deep learning frameworks, including TensorFlow.

In this notebook, we use the SageMaker Python SDK to launch a training
job and deploy the trained model. We use a Python script to train a
classification model on the `MNIST
dataset <http://yann.lecun.com/exdb/mnist>`__, and show how to train
with both TensorFlow 1.x and TensorFlow 2.x scripts.

Set up the environment
----------------------

Let’s start by setting up the environment:

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    
    role = get_execution_role()
    region = sagemaker_session.boto_region_name

We also define the TensorFlow version here, and create a quick helper
function that lets us toggle between TF 1.x and 2.x in this notebook.

.. code:: ipython3

    tf_version = '2.1.0'  # replace with '1.15.2' for TF 1.x
    
    def use_tf2():
        return tf_version.startswith('2')

Training Data
-------------

The `MNIST dataset <http://yann.lecun.com/exdb/mnist>`__ is a dataset
consisting of handwritten digits. There is a training set of 60,000
examples, and a test set of 10,000 examples. The digits have been
size-normalized and centered in a fixed-size image.

The dataset has already been uploaded to an Amazon S3 bucket,
``sagemaker-sample-data-<REGION>``, under the prefix
``tensorflow/mnist``. There are four ``.npy`` file under this prefix: \*
``train_data.npy`` \* ``eval_data.npy`` \* ``train_labels.npy`` \*
``eval_labels.npy``

.. code:: ipython3

    training_data_uri = 's3://sagemaker-sample-data-{}/tensorflow/mnist'.format(region)

Construct a script for distributed training
-------------------------------------------

The training code is very similar to a training script we might run
outside of Amazon SageMaker. The SageMaker Python SDK handles
transferring our script to a SageMaker training instance. On the
training instance, SageMaker’s native TensorFlow support sets up
training-related environment variables and executes the training code.

We can use a Python script, a Python module, or a shell script for the
training code. This notebook’s training script is a Python script
adapted from a TensorFlow example of training a convolutional neural
network on the MNIST dataset.

We have modified the training script to handle the ``model_dir``
parameter passed in by SageMaker. This is an Amazon S3 path which can be
used for data sharing during distributed training and checkpointing
and/or model persistence. Our script also contains an argument-parsing
function to handle processing training-related variables.

At the end of the training job, our script exports the trained model to
the path stored in the environment variable ``SM_MODEL_DIR``, which
always points to ``/opt/ml/model``. This is critical because SageMaker
uploads all the model artifacts in this folder to S3 at end of training.

For more about writing a TensorFlow training script for SageMaker, see
`the SageMaker
documentation <https://sagemaker.readthedocs.io/en/stable/using_tf.html#prepare-a-script-mode-training-script>`__.

Here is the entire script:

.. code:: ipython3

    training_script = 'mnist-2.py' if use_tf2() else 'mnist.py'
    
    !pygmentize {training_script}

Create a SageMaker training job
-------------------------------

The SageMaker Python SDK’s ``sagemaker.tensorflow.TensorFlow`` estimator
class handles creating a SageMaker training job. Let’s call out a couple
important parameters here:

-  ``entry_point``: our training script
-  ``distributions``: configuration for the distributed training setup.
   It’s required only if we want distributed training either across a
   cluster of instances or across multiple GPUs. Here, we use parameter
   servers as the distributed training schema. SageMaker training jobs
   run on homogeneous clusters. To make parameter server more performant
   in the SageMaker setup, we run a parameter server on every instance
   in the cluster, so there is no need to specify the number of
   parameter servers to launch. Script mode also supports distributed
   training with `Horovod <https://github.com/horovod/horovod>`__. You
   can find the full documentation on how to configure ``distributions``
   in the `SageMaker Python SDK API
   documentation <https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html#sagemaker.tensorflow.estimator.TensorFlow>`__.

.. code:: ipython3

    from sagemaker.tensorflow import TensorFlow
    
    estimator = TensorFlow(entry_point=training_script,
                           role=role,
                           train_instance_count=1,
                           train_instance_type='ml.p2.xlarge',
                           framework_version=tf_version,
                           py_version='py3',
                           distributions={'parameter_server': {'enabled': True}})

To start a training job, we call ``estimator.fit(training_data_uri)``.

An S3 location is used here as the input. ``fit`` creates a default
channel named “training”, and the data at the S3 location is downloaded
to the “training” channel. In the training script, we can then access
the training data from the location stored in ``SM_CHANNEL_TRAINING``.
``fit`` accepts a couple other types of input as well. For details, see
the `API
documentation <https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.EstimatorBase.fit>`__.

When training starts, ``mnist.py`` is executed, with the estimator’s
``hyperparameters`` and ``model_dir`` passed as script arguments.
Because we didn’t define either in this example, no hyperparameters are
passed, and ``model_dir`` defaults to
``s3://<DEFAULT_BUCKET>/<TRAINING_JOB_NAME>``, so the script execution
is as follows:

.. code:: bash

   python mnist.py --model_dir s3://<DEFAULT_BUCKET>/<TRAINING_JOB_NAME>

When training is complete, the training job uploads the saved model to
S3 so that we can use it with TensorFlow Serving.

.. code:: ipython3

    estimator.fit(training_data_uri)

Deploy the trained model to an endpoint
---------------------------------------

After we train our model, we can deploy it to a SageMaker Endpoint,
which serves prediction requests in real-time. To do so, we simply call
``deploy()`` on our estimator, passing in the desired number of
instances and instance type for the endpoint. This creates a SageMaker
Model, which is then deployed to an endpoint.

The Docker image used for TensorFlow Serving runs an implementation of a
web server that is compatible with SageMaker hosting protocol. For more
about using TensorFlow Serving with SageMaker, see the `SageMaker
documentation <https://sagemaker.readthedocs.io/en/stable/using_tf.html#deploy-tensorflow-serving-models>`__.

.. code:: ipython3

    predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')

Invoke the endpoint
-------------------

We then use the returned predictor object to invoke our endpoint. For
demonstration purposes, let’s download the training data and use that as
input for inference.

.. code:: ipython3

    import numpy as np
    
    !aws --region {region} s3 cp s3://sagemaker-sample-data-{region}/tensorflow/mnist/train_data.npy train_data.npy
    !aws --region {region} s3 cp s3://sagemaker-sample-data-{region}/tensorflow/mnist/train_labels.npy train_labels.npy
    
    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')

The formats of the input and the output data correspond directly to the
request and response formats of the ``Predict`` method in the
`TensorFlow Serving REST
API <https://www.tensorflow.org/serving/api_rest>`__. SageMaker’s
TensforFlow Serving endpoints can also accept additional input formats
that are not part of the TensorFlow REST API, including the simplified
JSON format, line-delimited JSON objects (“jsons” or “jsonlines”), and
CSV data.

In this example we use a ``numpy`` array as input, which is serialized
into the simplified JSON format. In addtion, TensorFlow serving can also
process multiple items at once, which we utilize in the following code.
For complete documentation on how to make predictions against a
SageMaker Endpoint using TensorFlow Serving, see the `SageMaker
documentation <https://sagemaker.readthedocs.io/en/stable/using_tf.html#making-predictions-against-a-sagemaker-endpoint>`__.

.. code:: ipython3

    predictions = predictor.predict(train_data[:50])
    for i in range(0, 50):
        if use_tf2():
            prediction = np.argmax(predictions['predictions'][i])
        else:
            prediction = predictions['predictions'][i]['classes']
    
        label = train_labels[i]
        print('prediction: {}, label: {}, matched: {}'.format(prediction, label, prediction == label))

Delete the endpoint
-------------------

Let’s delete our to prevent incurring any extra costs.

.. code:: ipython3

    predictor.delete_endpoint()
