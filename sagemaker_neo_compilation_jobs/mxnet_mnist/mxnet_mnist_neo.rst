MNIST Training, Compilation and Deployment with MXNet Module and Sagemaker Neo
------------------------------------------------------------------------------

The **SageMaker Python SDK** makes it easy to train and deploy MXNet
models. In this example, we train a simple neural network using the
Apache MXNet `Module
API <https://mxnet.apache.org/api/python/module/module.html>`__ and the
MNIST dataset. The MNIST dataset is widely used for handwritten digit
classification, and consists of 70,000 labeled 28x28 pixel grayscale
images of hand-written digits. The dataset is split into 60,000 training
images and 10,000 test images. There are 10 classes (one for each of the
10 digits). The task at hand is to train a model using the 60,000
training images and subsequently test its classification accuracy on the
10,000 test images.

Setup
~~~~~

First we need to define a few variables that will be needed later in the
example.

.. code:: ipython3

    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = Session().default_bucket()
    
    # Location to save your custom code in tar.gz format.
    custom_code_upload_location = 's3://{}/customcode/mxnet'.format(bucket)
    
    # Location where results of model training are saved.
    model_artifacts_location = 's3://{}/artifacts'.format(bucket)
    
    # IAM execution role that gives SageMaker access to resources in your AWS account.
    # We can use the SageMaker Python SDK to get the role from our notebook environment. 
    role = get_execution_role()

The training script
~~~~~~~~~~~~~~~~~~~

The ``mnist.py`` script provides all the code we need for training and
hosting a SageMaker model. The script we will use is adaptated from
Apache MXNet `MNIST
tutorial <https://mxnet.incubator.apache.org/tutorials/python/mnist.html>`__.

.. code:: ipython3

    !cat mnist.py

In the training script, there are two additional functions, to be used
with Neo Deep Learning Runtime: \*
``neo_preprocess(payload, content_type)``: Function that takes in the
payload and Content-Type of each incoming request and returns a NumPy
array. Here, the payload is byte-encoded NumPy array, so the function
simply decodes the bytes to obtain the NumPy array. \*
``neo_postprocess(result)``: Function that takes the prediction results
produced by Deep Learning Runtime and returns the response body

SageMaker’s MXNet estimator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker ``MXNet`` estimator allows us to run single machine or
distributed training in SageMaker, using CPU or GPU-based instances.

When we create the estimator, we pass in the filename of our training
script, the name of our IAM execution role, and the S3 locations we
defined in the setup section. We also provide a few other parameters.
``train_instance_count`` and ``train_instance_type`` determine the
number and type of SageMaker instances that will be used for the
training job. The ``hyperparameters`` parameter is a ``dict`` of values
that will be passed to your training script – you can see how to access
these values in the ``mnist.py`` script above.

For this example, we will choose one ``ml.m4.xlarge`` instance.

.. code:: ipython3

    from sagemaker.mxnet import MXNet
    
    mnist_estimator = MXNet(entry_point='mnist.py',
                            role=role,
                            output_path=model_artifacts_location,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            train_instance_type='ml.m4.xlarge',
                            framework_version='1.4.1',
                            py_version='py3',
                            distributions={'parameter_server': {'enabled': True}},
                            hyperparameters={'learning-rate': 0.1})

Running the Training Job
~~~~~~~~~~~~~~~~~~~~~~~~

After we’ve constructed our MXNet object, we can fit it using data
stored in S3. Below we run SageMaker training on two input channels:
**train** and **test**.

During training, SageMaker makes this data stored in S3 available in the
local filesystem where the mnist script is running. The ``mnist.py``
script simply loads the train and test data from disk.

.. code:: ipython3

    %%time
    import boto3
    
    region = boto3.Session().region_name
    train_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)
    test_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)
    
    mnist_estimator.fit({'train': train_data_location, 'test': test_data_location})

Optimize your model with Neo API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neo API allows to optimize our model for a specific hardware type. When
calling ``compile_model()`` function, we specify the target instance
family (C5) as well as the S3 bucket to which the compiled model would
be stored.

**Important. If the following command result in a permission error,
scroll up and locate the value of execution role returned by
``get_execution_role()``. The role must have access to the S3 bucket
specified in ``output_path``.**

.. code:: ipython3

    output_path = '/'.join(mnist_estimator.output_path.split('/')[:-1])
    compiled_model = mnist_estimator.compile_model(target_instance_family='ml_c5', 
                                                   input_shape={'data':[1, 784]},
                                                   role=role,
                                                   output_path=output_path)

Creating an inference Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can deploy this compiled model, note that we need to use the same
instance that the target we used for compilation. This creates a
SageMaker endpoint that we can use to perform inference.

The arguments to the ``deploy`` function allow us to set the number and
type of instances that will be used for the Endpoint. Make sure to
choose an instance for which you have compiled your model, so in our
case ``ml_c5``. Neo API uses a special runtime (DLR runtime), in which
our optimzed model will run.

.. code:: ipython3

    predictor = compiled_model.deploy(initial_instance_count = 1, instance_type = 'ml.c5.4xlarge')

This endpoint will receive uncompressed NumPy arrays, whose Content-Type
is given as ``application/vnd+python.numpy+binary``:

.. code:: ipython3

    import io
    import numpy as np
    def numpy_bytes_serializer(data):
        f = io.BytesIO()
        np.save(f, data)
        f.seek(0)
        return f.read()
    
    predictor.content_type = 'application/vnd+python.numpy+binary'
    predictor.serializer = numpy_bytes_serializer

Making an inference request
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that our Endpoint is deployed and we have a ``predictor`` object, we
can use it to classify handwritten digits.

To see inference in action, draw a digit in the image box below. The
pixel data from your drawing will be loaded into a ``data`` variable in
this notebook.

*Note: after drawing the image, you’ll need to move to the next notebook
cell.*

.. code:: ipython3

    from IPython.display import HTML
    HTML(open("input.html").read())

Now we can use the ``predictor`` object to classify the handwritten
digit:

.. code:: ipython3

    data = np.array(data)
    response = predictor.predict(data)
    print('Raw prediction result:')
    print(response)
    
    labeled_predictions = list(zip(range(10), response))
    print('Labeled predictions: ')
    print(labeled_predictions)
    
    labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
    print('Most likely answer: {}'.format(labeled_predictions[0]))

## Conclusion
-------------

SageMaker Neo automatically optimizes machine learning models to perform
at up to fourth the speed with no loss in accuracy. In the diagram below
shows you how our neo-optimized model performs better than the original
mxnet mnist model. The originl model stands for the uncompiled model
deployed on Flask container on May 26th, 2019 and neo-optimized model
stands for the compiled model deployed on Neo-AI-DLR container. The data
for each trial is the average of 1000 trys for each endpoint. |alt text|

.. |alt text| image:: mxnet-byom-latency.png

(Optional) Delete the Endpoint
==============================

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it.

.. code:: ipython3

    print("Endpoint name: " + predictor.endpoint)

.. code:: ipython3

    import sagemaker
    predictor.delete_endpoint()
