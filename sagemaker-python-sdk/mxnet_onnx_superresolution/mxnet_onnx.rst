Importing and hosting an ONNX model with MXNet
==============================================

The `Open Neural Network Exchange <https://onnx.ai/>`__ (ONNX) is an
open format for representing deep learning models with its extensible
computation graph model and definitions of built-in operators and
standard data types.

In this example, we will use the Super Resolution model from `Image
Super-Resolution Using Deep Convolutional
Networks <https://ieeexplore.ieee.org/document/7115171>`__, where Dong
et al. trained a model for taking a low-resolution image as input and
producing a high-resolution one. This model, along with many others, can
be found at the `ONNX Model Zoo <https://github.com/onnx/models>`__.

We will use the SageMaker Python SDK to host this ONNX model in
SageMaker, and perform inference requests.

Setup
-----

First, we’ll get the IAM execution role from our notebook environment,
so that SageMaker can access resources in your AWS account later in the
example.

.. code:: ipython3

    from sagemaker import get_execution_role
    
    role = get_execution_role()

The hosting script
------------------

We’ll need to provide a hosting script that can run on the SageMaker
platform. This script will be invoked by SageMaker when we perform
inference.

The script we’re using here implements two functions:

-  ``model_fn()`` - the SageMaker model server uses this function to
   load the model
-  ``transform_fn()`` - this function is for using the model to take the
   input and produce the output

The script here is an adaptation of the `ONNX Super Resolution
example <https://github.com/apache/incubator-mxnet/blob/master/example/onnx/super_resolution.py>`__
provided by the `Apache MXNet <https://mxnet.incubator.apache.org/>`__
project.

.. code:: ipython3

    !pygmentize super_resolution.py

Preparing the model
-------------------

To create a SageMaker Endpoint, we’ll first need to prepare the model to
be used in SageMaker.

Downloading the model
~~~~~~~~~~~~~~~~~~~~~

For this example, we will use a pre-trained ONNX model from the `ONNX
Model Zoo <https://github.com/onnx/models>`__, where you can find a
collection of pre-trained models to work with. Here, we will download
the `Super
Resolution <https://github.com/onnx/models#super-resolution>`__ model.

.. code:: ipython3

    !wget https://onnx-mxnet.s3.amazonaws.com/examples/super_resolution.onnx

Compressing the model data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have the model data locally, we will need to compress it and
upload the tarball to S3 for the SageMaker Python SDK to create a Model

.. code:: ipython3

    import tarfile
    
    from sagemaker.session import Session
    
    with tarfile.open('onnx_model.tar.gz', mode='w:gz') as archive:
        archive.add('super_resolution.onnx')
    
    model_data = Session().upload_data(path='onnx_model.tar.gz', key_prefix='model')

Creating a SageMaker Python SDK Model instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the model data uploaded to S3, we now have everything we need to
instantiate a SageMaker Python SDK Model. We’ll provide the constructor
the following arguments:

-  ``model_data``: the S3 location of the model data
-  ``entry_point``: the script for model hosting that we looked at above
-  ``role``: the IAM role used
-  ``framework_version``: the MXNet version in use

You can read more about creating an ``MXNetModel`` object in the
`SageMaker Python SDK API
docs <https://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-model>`__.

.. code:: ipython3

    from sagemaker.mxnet import MXNetModel
    
    mxnet_model = MXNetModel(model_data=model_data,
                             entry_point='super_resolution.py',
                             role=role,
                             py_version='py3',
                             framework_version='1.3.0')

Creating an Endpoint
--------------------

Now we can use our ``MXNetModel`` object to build and deploy an
``MXNetPredictor``. This creates a SageMaker Model and Endpoint, the
latter of which we can use for performing inference.

The arguments to the ``deploy()`` function allow us to set the number
and type of instances that will be used for the Endpoint. Here we will
deploy the model to a single ``ml.m4.xlarge`` instance.

.. code:: ipython3

    %%time
    
    predictor = mxnet_model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Performing inference
--------------------

With our Endpoint deployed, we can now send inference requests to it.
We’ll use one image as an example here.

Preparing the image
~~~~~~~~~~~~~~~~~~~

First, we’ll download the image (and view it).

.. code:: ipython3

    from IPython.display import Image as Img
    from mxnet.test_utils import download
    
    img_name = 'super_res_input.jpg'
    img_url = 'https://s3.amazonaws.com/onnx-mxnet/examples/{}'.format(img_name)
    download(img_url, img_name)
    
    Img(filename=img_name)

Next, we’ll resize it to be 224x224 pixels. In addition, we’ll use a
grayscale version of the image (or, more accurately, taking the ‘Y’
channel after converting it to
`YCbCr <https://en.wikipedia.org/wiki/YCbCr>`__) to match the images
that were used for training the model.

.. code:: ipython3

    import numpy as np
    from PIL import Image
    
    input_image_dim = 224
    img = Image.open(img_name).resize((input_image_dim, input_image_dim))
    
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    input_image = np.array(img_y)[np.newaxis, np.newaxis, :, :]

Sending the inference request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We’ll now call ``predict()`` on our predictor to use our model to create
a bigger image from the input image.

.. code:: ipython3

    out = predictor.predict(input_image)

Viewing the result
~~~~~~~~~~~~~~~~~~

Now we’ll look at the resulting image from our inference request. First
we’ll convert it and save it.

.. code:: ipython3

    img_out_y = Image.fromarray(np.uint8(np.asarray(out)), mode='L')
    result_img = Image.merge('YCbCr', [img_out_y,
                             img_cb.resize(img_out_y.size, Image.BICUBIC),
                             img_cr.resize(img_out_y.size, Image.BICUBIC)]).convert("RGB")
    output_img_dim = 672
    assert result_img.size == (output_img_dim, output_img_dim)
    
    result_img_file = 'output.jpg'
    result_img.save(result_img_file)

And now we’ll look at the image itself. We can see that it is indeed a
larger version of the image we started with.

.. code:: ipython3

    Img(filename=result_img_file)

For comparison, we can look at the original image simply resized,
without using the model. The lack of detail in this version is
especially noticeable with the dog’s fur.

.. code:: ipython3

    naive_output = Image.open(img_name).resize((output_img_dim, output_img_dim))
    
    naive_output_file = 'naive_output.jpg'
    naive_output.save(naive_output_file)
    
    Img(naive_output_file)

Deleting the Endpoint
---------------------

Since we’ve reached the end, we’ll delete the SageMaker Endpoint to
release the instance associated with it.

.. code:: ipython3

    predictor.delete_endpoint()
