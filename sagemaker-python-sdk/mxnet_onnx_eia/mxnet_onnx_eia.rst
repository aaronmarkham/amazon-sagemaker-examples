Hosting ONNX models with Amazon Elastic Inference
=================================================

Amazon Elastic Inference (EI) is a resource you can attach to your
Amazon EC2 instances to accelerate your deep learning (DL) inference
workloads. EI allows you to add inference acceleration to an Amazon
SageMaker hosted endpoint or Jupyter notebook and reduce the cost of
running deep learning inference by up to 75%, when compared to using GPU
instances. For more information, please visit:
https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

Amazon EI provides support for a variety of frameworks, including Apache
MXNet and ONNX models. The `Open Neural Network
Exchange <https://onnx.ai/>`__ (ONNX) is an open standard format for
deep learning models that enables interoperability between deep learning
frameworks such as Apache MXNet, Microsoft Cognitive Toolkit (CNTK),
PyTorch and more. This means that we can use any of these frameworks to
train the model, export these pretrained models in ONNX format and then
import them in MXNet for inference.

In this example, we use the ResNet-152v1 model from `Deep residual
learning for image recognition <https://arxiv.org/abs/1512.03385>`__.
This model, alongside many others, can be found at the `ONNX Model
Zoo <https://github.com/onnx/models>`__.

We use the SageMaker Python SDK to host this ONNX model in SageMaker and
perform inference requests.

Setup
-----

First, we get the IAM execution role from our notebook environment, so
that SageMaker can access resources in your AWS account later in the
example.

.. code:: ipython3

    from sagemaker import get_execution_role
    
    role = get_execution_role()

The inference script
--------------------

We need to provide an inference script that can run on the SageMaker
platform. This script is invoked by SageMaker when we perform inference.

The script we’re using here implements two functions:

-  ``model_fn()`` - loads the model
-  ``transform_fn()`` - uses the model to take the input and produce the
   output

.. code:: ipython3

    !pygmentize resnet152.py

Preparing the model
-------------------

To create a SageMaker Endpoint, we first need to prepare the model to be
used in SageMaker.

Downloading the model
~~~~~~~~~~~~~~~~~~~~~

For this example, we use a pre-trained ONNX model from the `ONNX Model
Zoo <https://github.com/onnx/models>`__, where you can find a collection
of pre-trained models to work with. Here, we download the `ResNet-152v1
model <https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.onnx>`__
trained on ImageNet dataset.

.. code:: ipython3

    import mxnet as mx
    
    mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.onnx')

Compressing the model data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have the model data locally, we need to compress it, and
then upload it to S3.

.. code:: ipython3

    import tarfile
    
    from sagemaker import s3, session
    
    with tarfile.open('onnx_model.tar.gz', mode='w:gz') as archive:
        archive.add('resnet152v1.onnx')
    
    bucket = session.Session().default_bucket()
    model_data = s3.S3Uploader.upload('onnx_model.tar.gz',
                                      's3://{}/mxnet-onnx-resnet152-example/model'.format(bucket))

Creating a SageMaker Python SDK Model instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the model data uploaded to S3, we now have everything we need to
instantiate a SageMaker Python SDK Model. We provide the constructor the
following arguments:

-  ``model_data``: the S3 location of the model data
-  ``entry_point``: the script for model hosting that we looked at above
-  ``role``: the IAM role used
-  ``framework_version``: the MXNet version in use, in this case ‘1.4.1’

For more about creating an ``MXNetModel`` object, see the `SageMaker
Python SDK API
docs <https://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-model>`__.

.. code:: ipython3

    from sagemaker.mxnet import MXNetModel
    
    mxnet_model = MXNetModel(model_data=model_data,
                             entry_point='resnet152.py',
                             role=role,
                             py_version='py3',
                             framework_version='1.4.1')

Creating an inference endpoint and attaching an Elastic Inference(EI) accelerator
---------------------------------------------------------------------------------

Now we can use our ``MXNetModel`` object to build and deploy an
``MXNetPredictor``. This creates a SageMaker Model and Endpoint, the
latter of which we can use for performing inference.

We pass the following arguments to the ``deploy()`` method:

-  ``instance_count`` - how many instances to back the endpoint.
-  ``instance_type`` - which EC2 instance type to use for the endpoint.
-  ``accelerator_type`` - which EI accelerator type to attach to each of
   our instances.

For information on supported instance types and accelerator types,
please see `the AWS
documentation <https://aws.amazon.com/sagemaker/pricing/instance-types>`__.

How our models are loaded
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the predefined SageMaker MXNet containers have a default
``model_fn``, which loads the model. The default ``model_fn`` loads an
MXNet Module object with a context based on the instance type of the
endpoint.

This applies for EI as well. If an EI accelerator is attached to your
endpoint and a custom ``model_fn`` isn’t provided, then the default
``model_fn`` loads the MXNet Module object with an EI context,
``mx.eia()``. This default ``model_fn`` works with the default save
function provided by the pre-built SageMaker MXNet Docker image for
training. If the model is saved in a different manner, then a custom
``model_fn`` implementation may be needed. For more information on
``model_fn``, see `the SageMaker
documentation <https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#load-a-model>`__.

Choosing instance types
~~~~~~~~~~~~~~~~~~~~~~~

Here, we deploy our model with instance type ``ml.m5.xlarge`` and
``ml.eia1.medium``. For this model, we found that it requires more CPU
memory and thus chose an M5 instance, which has more memory than C5
instances, making it more cost effective. With other models, you may
want to experiment with other instance types and accelerators based on
your model requirements.

.. code:: ipython3

    %%time
    
    predictor = mxnet_model.deploy(initial_instance_count=1,
                                   instance_type='ml.m5.xlarge',
                                   accelerator_type='ml.eia1.medium')

Performing inference
--------------------

With our Endpoint deployed, we can now send inference requests to it. We
use one image as an example here.

Preparing the image
~~~~~~~~~~~~~~~~~~~

First, we download the image (and view it).

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    img_path = mx.test_utils.download('https://s3.amazonaws.com/onnx-mxnet/examples/mallard_duck.jpg')
    img = mx.image.imread(img_path)
    plt.imshow(img.asnumpy())

Next, we preprocess inference image. We resize it to 256x256, take
center crop of 224x224, normalize image, and add a dimension to batchify
the image.

.. code:: ipython3

    from mxnet.gluon.data.vision import transforms
    
    def preprocess(img):
        transform_fn = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform_fn(img)
        img = img.expand_dims(axis=0)
        return img
    
    input_image = preprocess(img)

Sending the inference request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can use the predictor object to classify the input image:

.. code:: ipython3

    scores = predictor.predict(input_image.asnumpy())

To see the inference result, let’s download and load ``synset.txt`` file
containing class labels for ImageNet. The top 5 classes generated in
order, along with the probabilities are:

.. code:: ipython3

    import numpy as np
    
    mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/synset.txt')
    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]
    
    a = np.argsort(scores)[::-1]
    
    for i in a[0:5]:
        print('class=%s; probability=%f' %(labels[i],scores[i]))

Deleting the Endpoint
---------------------

Since we’ve reached the end, we delete the SageMaker Endpoint to release
the instance associated with it.

.. code:: ipython3

    predictor.delete_endpoint()
