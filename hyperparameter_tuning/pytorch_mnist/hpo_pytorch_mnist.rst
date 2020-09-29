Hyperparameter Tuning using SageMaker PyTorch Container
=======================================================

Kernel ``Python 3 (PyTorch CPU (or GPU) Optimized)`` works well with
this notebook.

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
using PyTorch. It also shows how to use SageMaker Automatic Model Tuning
to select appropriate hyperparameters in order to get the best model.

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

.. code:: ipython3

    import sagemaker
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
    
    sagemaker_session = sagemaker.Session()
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-mnist'
    
    role = sagemaker.get_execution_role()

Data
----

Getting the data
~~~~~~~~~~~~~~~~

.. code:: ipython3

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

.. code:: ipython3

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
``fit()`` method, the following will be set, following the format
``SM_CHANNEL_[channel_name]``:

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

.. code:: ipython3

    !pygmentize mnist.py

Set up hyperparameter tuning job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Note, with the default setting below, the hyperparameter tuning job can
take about 20 minutes to complete.*

Now that we have prepared the dataset and the script, we are ready to
train models. Before we do that, one thing to note is there are many
hyperparameters that can dramtically affect the performance of the
trained models. For example, learning rate, batch size, number of
epochs, etc. Since which hyperparameter setting can lead to the best
result depends on the dataset as well, it is almost impossible to pick
the best hyperparameter setting without searching for it. Using
SageMaker Automatic Model Tuning, we can create a hyperparameter tuning
job to search for the best hyperparameter setting in an automated and
effective way.

In this example, we are using SageMaker Python SDK to set up and manage
a hyperparameter tuning job. Specifically, we specify a range, or a list
of possible values in the case of categorical hyperparameters, for each
of the hyperparameter that we plan to tune. The hyperparameter tuning
job will automatically launch multiple training jobs with different
hyperparameter settings, evaluate results of those training jobs based
on a predefined “objective metric”, and select the hyperparameter
settings for future attempts based on previous results. For each
hyperparameter tuning job, we will give it a budget (max number of
training jobs) and it will complete once that many training jobs have
been executed.

Now we will set up the hyperparameter tuning job using SageMaker Python
SDK, following below steps: \* Create an estimator to set up the PyTorch
training job \* Define the ranges of hyperparameters we plan to tune, in
this example, we are tuning learning_rate and batch size \* Define the
objective metric for the tuning job to optimize \* Create a
hyperparameter tuner with above setting, as well as tuning resource
configurations

Similar to training a single PyTorch job in SageMaker, we define our
PyTorch estimator passing in the PyTorch script, IAM role, and (per job)
hardware configuration.

.. code:: ipython3

    from sagemaker.pytorch import PyTorch
    
    estimator = PyTorch(entry_point="mnist.py",
                        role=role,
                        framework_version='1.4.0',
                        train_instance_count=1,
                        train_instance_type='ml.m4.xlarge',
                        hyperparameters={
                            'epochs': 6,
                            'backend': 'gloo'
                        })

Once we’ve defined our estimator we can specify the hyperparameters we’d
like to tune and their possible values. We have three different types of
hyperparameters. - Categorical parameters need to take one value from a
discrete set. We define this by passing the list of possible values to
``CategoricalParameter(list)`` - Continuous parameters can take any real
number value between the minimum and maximum value, defined by
``ContinuousParameter(min, max)`` - Integer parameters can take any
integer value between the minimum and maximum value, defined by
``IntegerParameter(min, max)``

*Note, if possible, it’s almost always best to specify a value as the
least restrictive type. For example, tuning learning rate as a
continuous value between 0.01 and 0.2 is likely to yield a better result
than tuning as a categorical parameter with values 0.01, 0.1, 0.15, or
0.2. We did specify batch size as categorical parameter here since it is
generally recommended to be the power of 2.*

.. code:: ipython3

    hyperparameter_ranges = {'lr': ContinuousParameter(0.001, 0.1),'batch-size': CategoricalParameter([32,64,128,256,512])}

Next we’ll specify the objective metric that we’d like to tune and its
definition, which includes the regular expression (Regex) needed to
extract that metric from the CloudWatch logs of the training job. In
this particular case, our script emits average loss value and we will
use it as the objective metric, we also set the objective_type to be
‘minimize’, so that hyperparameter tuning seeks to minize the objective
metric when searching for the best hyperparameter setting. By default,
objective_type is set to ‘maximize’.

.. code:: ipython3

    objective_metric_name = 'average test loss'
    objective_type = 'Minimize'
    metric_definitions = [{'Name': 'average test loss',
                           'Regex': 'Test set: Average loss: ([0-9\\.]+)'}]

Now, we’ll create a ``HyperparameterTuner`` object, to which we pass: -
The PyTorch estimator we created above - Our hyperparameter ranges -
Objective metric name and definition - Tuning resource configurations
such as Number of training jobs to run in total and how many training
jobs can be run in parallel.

.. code:: ipython3

    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                max_jobs=9,
                                max_parallel_jobs=3,
                                objective_type=objective_type)

Launch hyperparameter tuning job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

And finally, we can start our hyperprameter tuning job by calling
``.fit()`` and passing in the S3 path to our train and test dataset.

After the hyperprameter tuning job is created, you should be able to
describe the tuning job to see its progress in the next step, and you
can go to SageMaker console->Jobs to check out the progress of the
progress of the hyperparameter tuning job.

.. code:: ipython3

    tuner.fit({'training': inputs})

Host
----

Create endpoint
~~~~~~~~~~~~~~~

After training, we use the tuner object to build and deploy a
``PyTorchPredictor``. This creates a Sagemaker Endpoint – a hosted
prediction service that we can use to perform inference, based on the
best model in the tuner. Remember in previous steps, the tuner launched
multiple training jobs during tuning and the resulting model with the
best objective metric is defined as the best model.

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

.. code:: ipython3

    predictor = tuner.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Evaluate
~~~~~~~~

We can now use this predictor to classify hand-written digits.

You will see an empty image box once you’ve executed cell below. Then
you can draw a number in it and pixel data will be loaded into a
``data`` variable in this notebook, which we can then pass to the
``predictor``.

.. code:: ipython3

    from IPython.display import HTML
    HTML(open("input.html").read())

.. code:: ipython3

    import numpy as np
    
    image = np.array([data], dtype=np.float32)
    response = predictor.predict(image)
    prediction = response.argmax(axis=1)[0]
    print(prediction)

Cleanup
~~~~~~~

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it

.. code:: ipython3

    tuner.delete_endpoint()
