PyTorch Cifar10 local training
==============================

Pre-requisites
--------------

This notebook shows how to use the SageMaker Python SDK to run your code
in a local container before deploying to SageMaker’s managed training or
hosting environments. This can speed up iterative testing and debugging
while using the same familiar Python SDK interface. Just change your
estimator’s ``train_instance_type`` to ``local`` (or ``local_gpu`` if
you’re using an ml.p2 or ml.p3 notebook instance).

In order to use this feature you’ll need to install docker-compose (and
nvidia-docker if training with a GPU).

**Note, you can only run a single local notebook at one time.**

.. code:: ipython2

    !/bin/bash ./setup.sh

Overview
--------

The **SageMaker Python SDK** helps you deploy your models for training
and hosting in optimized, productions ready containers in SageMaker. The
SageMaker Python SDK is easy to use, modular, extensible and compatible
with TensorFlow, MXNet, PyTorch and Chainer. This tutorial focuses on
how to create a convolutional neural network model to train the `Cifar10
dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`__ using **PyTorch
in local mode**.

Set up the environment
~~~~~~~~~~~~~~~~~~~~~~

This notebook was created and tested on a single ml.p2.xlarge notebook
instance.

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the sagemaker.get_execution_role() with
   appropriate full IAM role arn string(s).

.. code:: ipython2

    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-cnn-cifar10'
    
    role = sagemaker.get_execution_role()

.. code:: ipython2

    import os
    import subprocess
    
    instance_type = 'local'
    
    if subprocess.call('nvidia-smi') == 0:
        ## Set type to GPU if one is present
        instance_type = 'local_gpu'
        
    print("Instance type = " + instance_type)

Download the Cifar10 dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    from utils_cifar import get_train_data_loader, get_test_data_loader, imshow, classes
    
    trainloader = get_train_data_loader()
    testloader = get_test_data_loader()

Data Preview
~~~~~~~~~~~~

.. code:: ipython2

    import numpy as np
    import torchvision, torch
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # show images
    imshow(torchvision.utils.make_grid(images))
    
    # print labels
    print(' '.join('%9s' % classes[labels[j]] for j in range(4)))

Upload the data
~~~~~~~~~~~~~~~

We use the ``sagemaker.Session.upload_data`` function to upload our
datasets to an S3 location. The return value inputs identifies the
location – we will use this later when we start the training job.

.. code:: ipython2

    inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix='data/cifar10')

Construct a script for training
===============================

Here is the full code for the network model:

.. code:: ipython2

    !pygmentize source/cifar10.py

Script Functions
----------------

SageMaker invokes the main function defined within your training script
for training. When deploying your trained model to an endpoint, the
model_fn() is called to determine how to load your trained model. The
model_fn() along with a few other functions list below are called to
enable predictions on SageMaker.

`Predicting Functions <https://github.com/aws/sagemaker-pytorch-containers/blob/master/src/sagemaker_pytorch_container/serving.py>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  model_fn(model_dir) - loads your model.
-  input_fn(serialized_input_data, content_type) - deserializes
   predictions to predict_fn.
-  output_fn(prediction_output, accept) - serializes predictions from
   predict_fn.
-  predict_fn(input_data, model) - calls a model on data deserialized in
   input_fn.

The model_fn() is the only function that doesn’t have a default
implementation and is required by the user for using PyTorch on
SageMaker.

Create a training job using the sagemaker.PyTorch estimator
-----------------------------------------------------------

The ``PyTorch`` class allows us to run our training function on
SageMaker. We need to configure it with our training script, an IAM
role, the number of training instances, and the training instance type.
For local training with GPU, we could set this to “local_gpu”. In this
case, ``instance_type`` was set above based on your whether you’re
running a GPU instance.

After we’ve constructed our ``PyTorch`` object, we fit it using the data
we uploaded to S3. Even though we’re in local mode, using S3 as our data
source makes sense because it maintains consistency with how SageMaker’s
distributed, managed training ingests data.

.. code:: ipython2

    from sagemaker.pytorch import PyTorch
    
    cifar10_estimator = PyTorch(entry_point='source/cifar10.py',
                                role=role,
                                framework_version='1.4.0',
                                train_instance_count=1,
                                train_instance_type=instance_type)
    
    cifar10_estimator.fit(inputs)

Deploy the trained model to prepare for predictions
===================================================

The deploy() method creates an endpoint (in this case locally) which
serves prediction requests in real-time.

.. code:: ipython2

    from sagemaker.pytorch import PyTorchModel
    
    cifar10_predictor = cifar10_estimator.deploy(initial_instance_count=1,
                                                 instance_type=instance_type)

Invoking the endpoint
=====================

.. code:: ipython2

    # get some test images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%4s' % classes[labels[j]] for j in range(4)))
    
    outputs = cifar10_predictor.predict(images.numpy())
    
    _, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)
    
    print('Predicted: ', ' '.join('%4s' % classes[predicted[j]]
                                  for j in range(4)))

Clean-up
========

Deleting the local endpoint when you’re finished is important since you
can only run one local endpoint at a time.

.. code:: ipython2

    cifar10_estimator.delete_endpoint()
