MNIST Training using PyTorch
============================

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Data <#Data>`__
4. `Train <#Train>`__
5. `Host <#Host>`__

--------------

Background
----------

MNIST is a widely used dataset for handwritten digit classification. It
consists of 70,000 labeled 28x28 pixel grayscale images of hand-written
digits. The dataset is split into 60,000 training images and 10,000 test
images. There are 10 classes (one for each of the 10 digits). This
tutorial will show how to train and test an MNIST model on SageMaker
using PyTorch.

For more information about the PyTorch in SageMaker, please visit
`sagemaker-pytorch-containers <https://github.com/aws/sagemaker-pytorch-containers>`__
and
`sagemaker-python-sdk <https://github.com/aws/sagemaker-python-sdk>`__
github repositories.

--------------

Setup
-----

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by creating a SageMaker session and specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the ``sagemaker.get_execution_role()`` with a
   the appropriate full IAM role arn string(s).

.. code:: ipython2

    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-mnist'
    
    role = sagemaker.get_execution_role()

Data
----

Getting the data
~~~~~~~~~~~~~~~~

.. code:: ipython2

    from torchvision import datasets, transforms
    
    datasets.MNIST('data', download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

Uploading the data to S3
~~~~~~~~~~~~~~~~~~~~~~~~

We are going to use the ``sagemaker.Session.upload_data`` function to
upload our datasets to an S3 location. The return value inputs
identifies the location – we will use later when we start the training
job.

.. code:: ipython2

    inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
    print('input spec (in this case, just an S3 path): {}'.format(inputs))

Train
-----

Training script
~~~~~~~~~~~~~~~

The ``mnist.py`` script provides all the code we need for training and
hosting a SageMaker model (``model_fn`` function to load a model). The
training script is very similar to a training script you might run
outside of SageMaker, but you can access useful properties about the
training environment through various environment variables, such as:

-  ``SM_MODEL_DIR``: A string representing the path to the directory to
   write model artifacts to. These artifacts are uploaded to S3 for
   model hosting.
-  ``SM_NUM_GPUS``: The number of gpus available in the current
   container.
-  ``SM_CURRENT_HOST``: The name of the current container on the
   container network.
-  ``SM_HOSTS``: JSON encoded list containing all the hosts .

Supposing one input channel, ‘training’, was used in the call to the
PyTorch estimator’s ``fit()`` method, the following will be set,
following the format ``SM_CHANNEL_[channel_name]``:

-  ``SM_CHANNEL_TRAINING``: A string representing the path to the
   directory containing data in the ‘training’ channel.

For more information about training environment variables, please visit
`SageMaker Containers <https://github.com/aws/sagemaker-containers>`__.

A typical training script loads data from the input channels, configures
training with hyperparameters, trains a model, and saves a model to
``model_dir`` so that it can be hosted later. Hyperparameters are passed
to your script as arguments and can be retrieved with an
``argparse.ArgumentParser`` instance.

Because the SageMaker imports the training script, you should put your
training code in a main guard (``if __name__=='__main__':``) if you are
using the same script to host your model as we do in this example, so
that SageMaker does not inadvertently run your training code at the
wrong point in execution.

For example, the script run by this notebook:

.. code:: ipython2

    !pygmentize mnist.py

Run training in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``PyTorch`` class allows us to run our training function as a
training job on SageMaker infrastructure. We need to configure it with
our training script, an IAM role, the number of training instances, the
training instance type, and hyperparameters. In this case we are going
to run our training job on 2 ``ml.c4.xlarge`` instances. But this
example can be ran on one or multiple, cpu or gpu instances (`full list
of available
instances <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__).
The hyperparameters parameter is a dict of values that will be passed to
your training script – you can see how to access these values in the
``mnist.py`` script above.

.. code:: ipython2

    from sagemaker.pytorch import PyTorch
    
    estimator = PyTorch(entry_point='mnist.py',
                        role=role,
                        framework_version='1.4.0',
                        train_instance_count=2,
                        train_instance_type='ml.c4.xlarge',
                        hyperparameters={
                            'epochs': 6,
                            'backend': 'gloo'
                        })

After we’ve constructed our ``PyTorch`` object, we can fit it using the
data we uploaded to S3. SageMaker makes sure our data is available in
the local filesystem, so our training script can simply read the data
from disk.

.. code:: ipython2

    estimator.fit({'training': inputs})

Host
----

Create endpoint
~~~~~~~~~~~~~~~

After training, we use the ``PyTorch`` estimator object to build and
deploy a ``PyTorchPredictor``. This creates a Sagemaker Endpoint – a
hosted prediction service that we can use to perform inference.

As mentioned above we have implementation of ``model_fn`` in the
``mnist.py`` script that is required. We are going to use default
implementations of ``input_fn``, ``predict_fn``, ``output_fn`` and
``transform_fm`` defined in
`sagemaker-pytorch-containers <https://github.com/aws/sagemaker-pytorch-containers>`__.

The arguments to the deploy function allow us to set the number and type
of instances that will be used for the Endpoint. These do not need to be
the same as the values we used for the training job. For example, you
can train a model on a set of GPU-based instances, and then deploy the
Endpoint to a fleet of CPU-based instances, but you need to make sure
that you return or save your model as a cpu model similar to what we did
in ``mnist.py``. Here we will deploy the model to a single
``ml.m4.xlarge`` instance.

.. code:: ipython2

    predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Evaluate
~~~~~~~~

We can now use this predictor to classify hand-written digits. Drawing
into the image box loads the pixel data into a ``data`` variable in this
notebook, which we can then pass to the ``predictor``.

.. code:: ipython2

    from IPython.display import HTML
    HTML(open("input.html").read())

.. code:: ipython2

    import numpy as np
    
    image = np.array([data], dtype=np.float32)
    response = predictor.predict(image)
    prediction = response.argmax(axis=1)[0]
    print(prediction)

Cleanup
~~~~~~~

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it

.. code:: ipython2

    estimator.delete_endpoint()
