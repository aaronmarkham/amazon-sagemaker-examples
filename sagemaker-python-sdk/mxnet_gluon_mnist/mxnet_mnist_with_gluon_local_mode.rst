Local MNIST Training with MXNet and Gluon
-----------------------------------------

Pre-requisites
~~~~~~~~~~~~~~

This notebook shows how to use the SageMaker Python SDK to run your code
in a local container before deploying to SageMaker’s managed training or
hosting environments. This can speed up iterative testing and debugging
while using the same familiar Python SDK interface. Just change your
estimator’s ``train_instance_type`` to ``local``. You could also use
``local_gpu`` if you’re using an ml.p2 or ml.p3 notebook instance, but
then you’ll need to set ``train_instance_count=1`` since distributed,
local, GPU training is not yet supported.

In order to use this feature you’ll need to install docker-compose (and
nvidia-docker if training with a GPU). Running the setup.sh script below
will handle this for you.

**Note, you can only run a single local notebook at one time.**

.. code:: ipython3

    !/bin/bash ./setup.sh

Overview
~~~~~~~~

MNIST is a widely used dataset for handwritten digit classification. It
consists of 70,000 labeled 28x28 pixel grayscale images of hand-written
digits. The dataset is split into 60,000 training images and 10,000 test
images. There are 10 classes (one for each of the 10 digits). This
tutorial will show how to train and test an MNIST model on SageMaker
local mode using MXNet and the Gluon API.

.. code:: ipython3

    import os
    import subprocess
    import boto3
    import sagemaker
    from sagemaker.mxnet import MXNet
    from mxnet import gluon
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    
    instance_type = 'local'
    
    if subprocess.call('nvidia-smi') == 0:
        ## Set type to GPU if one is present
        instance_type = 'local_gpu'
        
    print("Instance type = " + instance_type)
    
    role = get_execution_role()

Download training and test data
-------------------------------

.. code:: ipython3

    gluon.data.vision.MNIST('./data/train', train=True)
    gluon.data.vision.MNIST('./data/test', train=False)

Uploading the data
------------------

We use the ``sagemaker.Session.upload_data`` function to upload our
datasets to an S3 location. The return value ``inputs`` identifies the
location – we will use this later when we start the training job.

.. code:: ipython3

    inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')

Implement the training function
-------------------------------

We need to provide a training script that can run on the SageMaker
platform. The training scripts are essentially the same as one you would
write for local training, except that you need to provide a ``train``
function. The ``train`` function will check for the validation accuracy
at the end of every epoch and checkpoints the best model so far, along
with the optimizer state, in the folder ``/opt/ml/checkpoints`` if the
folder path exists, else it will skip the checkpointing. When SageMaker
calls your function, it will pass in arguments that describe the
training environment. Check the script below to see how this works.

The script here is an adaptation of the `Gluon MNIST
example <https://github.com/apache/incubator-mxnet/blob/master/example/gluon/mnist.py>`__
provided by the `Apache MXNet <https://mxnet.incubator.apache.org/>`__
project.

.. code:: ipython3

    !cat 'mnist.py'

Run the training script on SageMaker
------------------------------------

The ``MXNet`` class allows us to run our training function on SageMaker
local mode. We need to configure it with our training script, an IAM
role, the number of training instances, and the training instance type.
This is the the only difference from
`mnist_with_gluon.ipynb <./mnist_with_gluon.ipynb>`__. Instead of
``train_instance_type='ml.c4.xlarge'``, we set it to
``train_instance_type='local'``. For local training with GPU, we could
set this to “local_gpu”. In this case, ``instance_type`` was set above
based on your whether you’re running a GPU instance.

.. code:: ipython3

    m = MXNet("mnist.py",
              role=role,
              train_instance_count=1,
              train_instance_type=instance_type,
              framework_version="1.6.0",
              py_version="py3",
              hyperparameters={'batch-size': 100,
                               'epochs': 20,
                               'learning-rate': 0.1,
                               'momentum': 0.9,
                               'log-interval': 100})

After we’ve constructed our ``MXNet`` object, we fit it using the data
we uploaded to S3. Even though we’re in local mode, using S3 as our data
source makes sense because it maintains consistency with how SageMaker’s
distributed, managed training ingests data.

.. code:: ipython3

    m.fit(inputs)

After training, we use the MXNet object to deploy an MXNetPredictor
object. This creates a SageMaker endpoint locally that we can use to
perform inference.

This allows us to perform inference on json encoded multi-dimensional
arrays.

.. code:: ipython3

    predictor = m.deploy(initial_instance_count=1, instance_type=instance_type)

We can now use this predictor to classify hand-written digits. Drawing
into the image box loads the pixel data into a ‘data’ variable in this
notebook, which we can then pass to the mxnet predictor.

.. code:: ipython3

    from IPython.display import HTML
    HTML(open("input.html").read())

The predictor runs inference on our input data and returns the predicted
digit (as a float value, so we convert to int for display).

.. code:: ipython3

    response = predictor.predict(data)
    print(int(response))

Clean-up
--------

Deleting the local endpoint when you’re finished is important since you
can only run one local endpoint at a time.

.. code:: ipython3

    predictor.delete_endpoint()
