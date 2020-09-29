Using Amazon Elastic Inference with MXNet on Amazon SageMaker
=============================================================

This notebook demonstrates how to enable and use Amazon Elastic
Inference with the prebuilt Amazon SageMaker MXNet images.

Amazon Elastic Inference (EI) is a resource you can attach to your
Amazon EC2 instances to accelerate your deep learning (DL) inference
workloads. EI allows you to add inference acceleration to an Amazon
SageMaker hosted endpoint or Jupyter notebook for a fraction of the cost
of using a full GPU instance. For more information please visit:
https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

This notebook is an adaption of the `SageMaker MXNet MNIST
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/aws_sagemaker_studio/frameworks/mxnet_mnist/mxnet_mnist.ipynb>`__,
with modifications showing the changes needed to enable and use EI with
MXNet on SageMaker.

1.  `Using Amazon Elastic Inference with MXNet on Amazon
    SageMaker <#Using-Amazon-Elastic-Inference-with-MXNet-on-Amazon-SageMaker>`__
2.  `MNIST dataset <#MNIST-dataset>`__
3.  `Setup <#Setup>`__
4.  `The training script <#The-training-script>`__
5.  `SageMaker’s MXNet estimator
    class <#SageMaker's-MXNet-estimator-class>`__
6.  `Running the training job <#Running-the-training-Job>`__
7.  `Creating an inference endpoint and attaching an EI
    accelerator <#Creating-an-inference-endpoint-and-attaching-an-EI-accelerator>`__
8.  `How our models are loaded <#How-our-models-are-loaded>`__
9.  `Using EI with a SageMaker notebook
    instance <#Using-EI-with-a-SageMaker-notebook-instance>`__
10. `Making an inference request <#Making-an-inference-request>`__
11. `Delete the Endpoint <#Delete-the-endpoint>`__

If you are familiar with SageMaker and already have a trained model,
skip ahead to the `“Creating an inference endpoint”
section <#Creating-an-inference-endpoint-with-EI>`__

For this example, we use the SageMaker Python SDK, which makes it easy
to train and deploy MXNet models. For our MXNet model, we train a simple
neural network using the Apache MXNet `Module
API <https://mxnet.apache.org/api/python/module/module.html>`__ and the
MNIST dataset.

MNIST dataset
~~~~~~~~~~~~~

The MNIST dataset is widely used for handwritten digit classification,
and consists of 70,000 labeled 28x28 pixel grayscale images of
hand-written digits. The dataset is split into 60,000 training images
and 10,000 test images. There are 10 classes (one for each of the 10
digits). The task at hand is to train a model using the 60,000 training
images and then test its classification accuracy on the 10,000 test
images.

Setup
~~~~~

Let’s start by creating a SageMaker session and specifying the IAM role
arn used to give training and hosting access to your data. See the
`documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__
for how to create these. Note, if more than one role is required for
notebook instances, training, and/or hosting, please replace the
``sagemaker.get_execution_role()`` with a the appropriate full IAM role
arn string(s).

.. code:: ipython3

    import sagemaker
    
    role = sagemaker.get_execution_role()

The training script
~~~~~~~~~~~~~~~~~~~

The ``mnist.py`` script provides all the code we need to train and host
a SageMaker model. The script also checkpoints the model at the end of
every epoch and saves the model graph, params and optimizer state in the
folder ``/opt/ml/checkpoints``. If the folder path does not exist then
it will skip checkpointing. The script we will use is adaptated from
Apache MXNet `MNIST
tutorial <https://mxnet.incubator.apache.org/tutorials/python/mnist.html>`__.

.. code:: ipython3

    !pygmentize mnist.py

SageMaker’s MXNet estimator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker ``MXNet`` estimator allows us to run single-machine or
distributed training in SageMaker, using CPU or GPU-based instances.

When we create the estimator, we pass in the filename of our training
script, the name of our IAM execution role, and the S3 locations we
defined in the setup section. We also provide a few other parameters.
``train_instance_count`` and ``train_instance_type`` determine the
number and type of SageMaker instances that are used for the training
job. The ``hyperparameters`` parameter is a ``dict`` of values that are
passed to your training script. You can see how to access these values
in the ``mnist.py`` script above.

For this example, we use one ``ml.m4.xlarge`` instance for our training
job.

.. code:: ipython3

    from sagemaker.mxnet import MXNet
    
    mnist_estimator = MXNet(entry_point='mnist.py',
                            role=role,
                            train_instance_count=1,
                            train_instance_type='ml.m4.xlarge',
                            framework_version='1.4.1',
                            py_version='py3',
                            hyperparameters={'learning-rate': 0.1})

Running the training Job
~~~~~~~~~~~~~~~~~~~~~~~~

After we’ve constructed our ``MXNet`` object, we can fit it using data
stored in S3. In the next cell we run SageMaker training on two input
channels: train and test.

During training, SageMaker makes this data stored in S3 available in the
local filesystem where the MNIST script is running. The ``mnist.py``
script simply loads the train and test data from disk.

.. code:: ipython3

    %%time
    import boto3
    
    region = boto3.Session().region_name
    train_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)
    test_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)
    
    mnist_estimator.fit({'train': train_data_location, 'test': test_data_location})

Creating an inference endpoint and attaching an EI accelerator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training, we call the ``deploy`` method of the ``MXNet`` estimator
object to build and deploy an ``MXNetPredictor``. This creates a
Sagemaker endpoint, which is a hosted prediction service that we can use
to perform inference.

We pass the following arguments to the ``deploy`` method:

-  ``instance_count`` - how many instances to back the endpoint.
-  ``instance_type`` - which EC2 instance type to use for the endpoint.
   For information on supported instance, please check `the AWS
   documentation <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__.
-  ``accelerator_type`` - which EI accelerator type to attach to each of
   our instances. The supported types of accelerators can be found in
   `the AWS
   documentation <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__.

How our models are loaded
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the predefined SageMaker MXNet containers have a default
``model_fn``, which determines how your model is loaded. The default
``model_fn`` loads an MXNet Module object with a context based on the
instance type of the endpoint.

This applies for EI as well. If an EI accelerator is attached to your
endpoint and a custom ``model_fn`` isn’t provided, then the default
``model_fn`` will load the MXNet Module object with an EI context,
``mx.eia()``. This default ``model_fn`` works with the default ``save``
function. If a custom ``save`` function was defined, then you may need
to write a custom ``model_fn`` function. For more information on
``model_fn``, see `this documentation for using MXNet with
SageMaker <https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#load-a-model>`__.

For examples on how to load and serve a MXNet Module object explicitly,
please see our `predefined default ``model_fn`` for
MXNet <https://github.com/aws/sagemaker-mxnet-serving-container/blob/master/src/sagemaker_mxnet_serving_container/default_inference_handler.py#L36>`__.

.. code:: ipython3

    %%time
    
    predictor = mnist_estimator.deploy(initial_instance_count=1,
                                       instance_type='ml.m4.xlarge',
                                       accelerator_type='ml.eia1.medium')

The request handling behavior of the endpoint is determined by the
``mnist.py`` script. In this case, the script doesn’t include any
request handling functions, so the endpoint uses the default handlers
provided by SageMaker. These default handlers allow us to perform
inference on input data encoded as a multi-dimensional JSON array.

Making an inference request
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that our endpoint is deployed and we have a ``predictor`` object, we
can use it to classify handwritten digits.

To see inference in action, draw a digit in the image box below. The
pixel data from your drawing is loaded into a variable named ``data``.

**Note**\ *: after drawing the image, you’ll need to move to the next
notebook cell.*

.. code:: ipython3

    from IPython.display import HTML
    HTML(open("input.html").read())

Now we can use the ``predictor`` object to classify the handwritten
digit:

.. code:: ipython3

    %%time 
    response = predictor.predict(data)
    print('Raw prediction result:')
    print(response)
    
    labeled_predictions = list(zip(range(10), response[0]))
    print('Labeled predictions: ')
    print(labeled_predictions)
    
    labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
    print('Most likely answer: {}'.format(labeled_predictions[0]))

Delete the endpoint
~~~~~~~~~~~~~~~~~~~

After you have finished with this example, remember to delete the
prediction endpoint.

.. code:: ipython3

    print("Endpoint name: " + predictor.endpoint)

.. code:: ipython3

    import sagemaker
    
    predictor.delete_endpoint()
