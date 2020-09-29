Exporting ONNX Models with MXNet
--------------------------------

The `Open Neural Network Exchange <https://onnx.ai/>`__ (ONNX) is an
open format for representing deep learning models with an extensible
computation graph model, definitions of built-in operators, and standard
data types. Starting with MXNet 1.3, models trained using MXNet can now
be saved as ONNX models.

In this example, we show how to train a model on Amazon SageMaker and
save it as an ONNX model. This notebook is based on the `MXNet MNIST
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb>`__
and the `MXNet example for exporting to
ONNX <https://mxnet.incubator.apache.org/tutorials/onnx/export_mxnet_to_onnx.html>`__.

Setup
~~~~~

First we need to define a few variables that we’ll need later in the
example.

.. code:: ipython3

    import boto3
    
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # AWS region
    region = boto3.Session().region_name
    
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

    !pygmentize mnist.py

Exporting to ONNX
~~~~~~~~~~~~~~~~~

The important part of this script can be found in the ``save`` method.
This is where the ONNX model is exported:

.. code:: python

   import os

   from mxnet.contrib import onnx as onnx_mxnet
   import numpy as np

   def save(model_dir, model):
       symbol_file = os.path.join(model_dir, 'model-symbol.json')
       params_file = os.path.join(model_dir, 'model-0000.params')

       model.symbol.save(symbol_file)
       model.save_params(params_file)

       data_shapes = [[dim for dim in data_desc.shape] for data_desc in model.data_shapes]
       output_path = os.path.join(model_dir, 'model.onnx')
       
       onnx_mxnet.export_model(symbol_file, params_file, data_shapes, np.float32, output_path)

The last line in that method, ``onnx_mxnet.export_model``, saves the
model in the ONNX format. We pass the following arguments:

-  ``symbol_file``: path to the saved input symbol file
-  ``params_file``: path to the saved input params file
-  ``data_shapes``: list of the input shapes
-  ``np.float32``: input data type
-  ``output_path``: path to save the generated ONNX file

For more information, see the `MXNet
Documentation <https://mxnet.incubator.apache.org/api/python/contrib/onnx.html#mxnet.contrib.onnx.mx2onnx.export_model.export_model>`__.

Training the model
~~~~~~~~~~~~~~~~~~

With the training script written to export an ONNX model, the rest of
training process looks like any other Amazon SageMaker training job
using MXNet. For a more in-depth explanation of these steps, see the
`MXNet MNIST
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb>`__.

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
    
    train_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)
    test_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)
    
    mnist_estimator.fit({'train': train_data_location, 'test': test_data_location})

Next steps
~~~~~~~~~~

Now that we have an ONNX model, we can deploy it to an endpoint in the
same way we do in the `MXNet MNIST
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb>`__.

For examples on how to write a ``model_fn`` to load the ONNX model,
please refer to: \* the `MXNet ONNX Super Resolution
notebook <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/mxnet_onnx_superresolution>`__
\* the `MXNet
documentation <https://mxnet.incubator.apache.org/api/python/contrib/onnx.html#mxnet.contrib.onnx.onnx2mx.import_model.import_model>`__
