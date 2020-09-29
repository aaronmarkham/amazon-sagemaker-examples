Using the Apache MXNet Module API with SageMaker Training and Batch Transformation
==================================================================================

*(This notebook was tested with the “Python 3 (Data Science)” kernel.)*

The SageMaker Python SDK makes it easy to train MXNet models and use
them for batch transformation. In this example, we train a simple neural
network using the Apache MXNet `Module
API <https://mxnet.incubator.apache.org/api/python/module.html>`__ and
the MNIST dataset. The MNIST dataset is widely used for handwritten
digit classification, and consists of 70,000 labeled 28x28 pixel
grayscale images of hand-written digits. The dataset is split into
60,000 training images and 10,000 test images. There are 10 classes (one
for each of the 10 digits). The task at hand is to train a model using
the 60,000 training images and subsequently test its classification
accuracy on the 10,000 test images.

Setup
~~~~~

First, we define a few variables that are be needed later in the
example.

.. code:: ipython3

    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    sagemaker_session = Session()
    region = sagemaker_session.boto_session.region_name
    sample_data_bucket = 'sagemaker-sample-data-{}'.format(region)
    
    # S3 bucket for saving files. Feel free to redefine this variable to the bucket of your choice.
    bucket = sagemaker_session.default_bucket()
    
    # Bucket location where your custom code will be saved in the tar.gz format.
    custom_code_upload_location = 's3://{}/mxnet-mnist-example/code'.format(bucket)
    
    # Bucket location where results of model training are saved.
    model_artifacts_location = 's3://{}/mxnet-mnist-example/artifacts'.format(bucket)
    
    # IAM execution role that gives SageMaker access to resources in your AWS account.
    # We can use the SageMaker Python SDK to get the role from our notebook environment. 
    role = get_execution_role()

Training and inference script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``mnist.py`` script provides all the code we need for training and
and inference. The script also checkpoints the model at the end of every
epoch and saves the model graph, params and optimizer state in the
folder ``/opt/ml/checkpoints``. If the folder path does not exist then
it skips checkpointing. The script we use is adaptated from the Apache
MXNet `MNIST
tutorial <https://mxnet.incubator.apache.org/tutorials/python/mnist.html>`__.

.. code:: ipython3

    !pygmentize mnist.py

SageMaker’s MXNet estimator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker ``MXNet`` estimator allows us to run single machine or
distributed training in SageMaker, using CPU or GPU-based instances.

When we create the estimator, we pass in the filename of our training
script, the name of our IAM execution role, and the S3 locations we
defined in the setup section. We also provide a few other parameters.
``train_instance_count`` and ``train_instance_type`` determine the
number and type of SageMaker instances that are used for the training
job. The ``hyperparameters`` parameter is a ``dict`` of values that is
passed to your training script – you can see how to access these values
in the ``mnist.py`` script above.

For this example, we choose one ``ml.m4.xlarge`` instance for our
training job.

.. code:: ipython3

    from sagemaker.mxnet import MXNet
    
    mnist_estimator = MXNet(entry_point='mnist.py',
                            role=role,
                            output_path=model_artifacts_location,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            train_instance_type='ml.m4.xlarge',
                            framework_version='1.6.0',
                            py_version='py3',
                            hyperparameters={'learning-rate': 0.1})

Running a training job
~~~~~~~~~~~~~~~~~~~~~~

After we’ve constructed our ``MXNet`` object, we can fit it using data
stored in S3. Below we run SageMaker training on two input channels:
train and test.

During training, SageMaker makes this data stored in S3 available in the
local filesystem where the ``mnist.py`` script is running. The script
then simply loads the train and test data from disk.

.. code:: ipython3

    %%time
    
    train_data_location = 's3://{}/mxnet/mnist/train'.format(sample_data_bucket)
    test_data_location = 's3://{}/mxnet/mnist/test'.format(sample_data_bucket)
    
    mnist_estimator.fit({'train': train_data_location, 'test': test_data_location})

SageMaker’s transformer class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training, we use our ``MXNet`` estimator object to create a
``Transformer`` by invoking the ``transformer()`` method. This method
takes arguments for configuring our options with the batch transform
job; these do not need to be the same values as the one we used for the
training job. The method also creates a SageMaker Model to be used for
the batch transform jobs.

The ``Transformer`` class is responsible for running batch transform
jobs, which deploys the trained model to an endpoint and send requests
for performing inference.

.. code:: ipython3

    transformer = mnist_estimator.transformer(instance_count=1, instance_type='ml.m4.xlarge')

Running a batch transform job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can perform some inference with the model we’ve trained by
running a batch transform job. The request handling behavior during the
transform job is determined by the ``mnist.py`` script.

For demonstration purposes, we’re going to use input data that contains
1000 MNIST images, located in the public SageMaker sample data S3
bucket. To create the batch transform job, we simply call
``transform()`` on our transformer with information about the input
data.

.. code:: ipython3

    input_file_path = 'batch-transform/mnist-1000-samples'
    
    transformer.transform('s3://{}/{}'.format(sample_data_bucket, input_file_path), content_type='text/csv')

Now we wait for the batch transform job to complete. We have a
convenience method, ``wait()``, that blocks until the batch transform
job has completed. We call that here to see if the batch transform job
is still running; the cell finishes running when the batch transform job
has completed.

.. code:: ipython3

    transformer.wait()

Downloading the results
~~~~~~~~~~~~~~~~~~~~~~~

The batch transform job uploads its predictions to S3. Since we did not
specify ``output_path`` when creating the Transformer, one was generated
based on the batch transform job name:

.. code:: ipython3

    print(transformer.output_path)

The output here will be a list of predictions, where each prediction is
a list of probabilities, one for each possible label. Since we read the
output as a string, we use ``ast.literal_eval()`` to turn it into a list
and find the maximum element of the list gives us the predicted label.
Here we define a convenience method to take the output and produce the
predicted label.

.. code:: ipython3

    import ast
    
    def predicted_label(transform_output):
        output = ast.literal_eval(transform_output)
        probabilities = output[0]
        return probabilities.index(max(probabilities))

Now let’s download the first ten results from S3:

.. code:: ipython3

    import json
    
    from sagemaker.s3 import S3Downloader
    
    predictions = []
    for i in range(10):
        file_key = '{}/data-{}.csv.out'.format(transformer.output_path, i)
        output = S3Downloader.read_file(file_key)
    
        predictions.append(predicted_label(output))

For demonstration purposes, we also download and display the
corresponding original input data so that we can see how the model did
with its predictions:

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.rcParams['figure.figsize'] = (2,10)
    
    def show_digit(img, caption='', subplot=None):
        if subplot == None:
            _,(subplot) = plt.subplots(1,1)
        imgr = img.reshape((28,28))
        subplot.axis('off')
        subplot.imshow(imgr, cmap='gray')
        plt.title(caption)
    
    for i in range(10):
        input_file_name = 'data-{}.csv'.format(i)
        input_file_uri = 's3://{}/{}/{}'.format(sample_data_bucket, input_file_path, input_file_name)
    
        input_data = np.fromstring(S3Downloader.read_file(input_file_uri), sep=',')
        show_digit(input_data)

Here, we can see the original labels are:

::

   7, 2, 1, 0, 4, 1, 4, 9, 5, 9

Now let’s print out the predictions to compare:

.. code:: ipython3

    print(predictions)
