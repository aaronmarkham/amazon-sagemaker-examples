PyTorch Batch Inference
=======================

In this notebook, we’ll examine how to do batch transform task with
PyTorch in Amazon SageMaker.

First, an image classification model is build on MNIST dataset. Then, we
demonstrate batch transform by using SageMaker Python SDK PyTorch
framework with different configurations - ``data_type=S3Prefix``: uses
all objects that match the specified S3 key name prefix for batch
inference. - ``data_type=ManifestFile``: a manifest file containing a
list of object keys that you want to batch inference. -
``instance_count>1``: distribute the batch inference dataset to multiple
inference instance

For batch transform in TensorFlow in Amazon SageMaker, you can follow
other Jupyter notebooks
`here <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker_batch_transform>`__

Setup
-----

We’ll begin with some necessary imports, and get an Amazon SageMaker
session to help perform certain tasks, as well as an IAM role with the
necessary permissions.

.. code:: ipython3

    %matplotlib inline
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from os import listdir
    from os.path import isfile, join
    from shutil import copyfile
    import sagemaker
    from sagemaker.pytorch import PyTorchModel
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-batch-inference-script'
    print('Bucket:\n{}'.format(bucket))

Model Training
--------------

Since the main purpose of this notebook is to demonstrate SageMaker
PyTorch batch transform, **we reuse this SageMaker Python
SDK**\ `PyTorch
example <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/pytorch_mnist>`__\ **to
train a PyTorch model**. It takes around 7 minutes to finish the
training.

.. code:: ipython3

    from torchvision import datasets, transforms
    
    datasets.MNIST('data', download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    
    inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
    print('input spec (in this case, just an S3 path): {}'.format(inputs))
    
    from sagemaker.pytorch import PyTorch
    
    estimator = PyTorch(entry_point='mnist.py',
                        role=role,
                        framework_version='1.5.1',
                        train_instance_count=2,
                        train_instance_type='ml.c4.xlarge',
                        hyperparameters={
                            'epochs': 6,
                            'backend': 'gloo'
                        })
    
    estimator.fit({'training': inputs})

Prepare batch inference data
============================

In this section, we run the bash script ``prep_inference_data.sh`` to
download MNIST dataset in PNG format, subsample 1000 images and upload
to S3 for batch inference.

.. code:: ipython3

    sample_folder = 'mnist_sample'

.. code:: ipython3

    # silence the output of the bash command so that the jupyter notebook will not response slowly
    !sh prep_inference_data.sh {sample_folder} > /dev/null

.. code:: ipython3

    # upload sample images to s3, it will take around 1~2 minutes
    inference_inputs = sagemaker_session.upload_data(path=sample_folder, key_prefix=f'{prefix}/images')
    display(inference_inputs)

Create model transformer
========================

Now, we will create a transformer object for handling creating and
interacting with Amazon SageMaker transform jobs. We can create the
transformer in two ways as shown in the following notebook cells. - use
fitted estimator directly - first create PyTorchModel from saved model
artefect, then create transformer from PyTorchModel object

Here, we implement the ``model_fn``, ``input_fn``, ``predict_fn`` and
``output_fn`` function to override the default `PyTorch inference
handler <https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py>`__.

It is noted that in ``input_fn`` function, the inferenced images are
encoded as a Python ByteArray. That’s why we use ``load_from_bytearray``
function to load image from ``io.BytesIO`` then use ``PIL.image`` to
read.

.. code:: python

   def model_fn(model_dir):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model = torch.nn.DataParallel(Net())
       with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
           model.load_state_dict(torch.load(f))
       return model.to(device)

       
   def load_from_bytearray(request_body):
       image_as_bytes = io.BytesIO(request_body)
       image = Image.open(image_as_bytes)
       image_tensor = ToTensor()(image).unsqueeze(0)    
       return image_tensor


   def input_fn(request_body, request_content_type):
       # if set content_type as 'image/jpg' or 'applicaiton/x-npy', 
       # the input is also a python bytearray
       if request_content_type == 'application/x-image': 
           image_tensor = load_from_bytearray(request_body)
       else:
           print("not support this type yet")
           raise ValueError("not support this type yet")
       return image_tensor


   # Perform prediction on the deserialized object, with the loaded model
   def predict_fn(input_object, model):
       output = model.forward(input_object)
       pred = output.max(1, keepdim=True)[1]

       return {'predictions':pred.item()}


   # Serialize the prediction result into the desired response content type
   def output_fn(predictions, response_content_type):
       return json.dumps(predictions)

.. code:: ipython3

    # Use fitted estimator directly
    transformer = estimator.transformer(instance_count=1,
                                        instance_type='ml.c4.xlarge')

.. code:: ipython3

    # You can also create a Transformer object from saved model artefect
    
    # get model artefect location by estimator.model_data, or give a S3 key directly
    model_artefect_s3_location = estimator.model_data  #'s3://BUCKET/PREFIX/model.tar.gz'
    
    # create PyTorchModel from saved model artefect
    pytorch_model = PyTorchModel(model_data=model_artefect_s3_location,
                                 role=role,
                                 framework_version='1.5.1',
                                 py_version='py3',
                                 source_dir='.',
                                 entry_point='mnist.py')
    
    # then create transformer from PyTorchModel object
    transformer = pytorch_model.transformer(instance_count=1, instance_type='ml.c4.xlarge')

Batch inference
---------------

Next, we will inference the sampled 1000 MNIST images in a batch manner.

input images directly from S3 location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We set ``S3DataType=S3Prefix`` to uses all objects that match the
specified S3 key name prefix for batch inference.

.. code:: ipython3

    transformer.transform(data=inference_inputs, 
                          data_type='S3Prefix',
                          content_type='application/x-image', 
                          wait=False)

input images by manifest file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we generate a manifest file. Then we use the manifest file
containing a list of object keys that you want to batch inference. Some
key points: - content_type = ‘application/x-image’ (!!! here the
content_type is for the actual object to be inference, not for the
manifest file) - data_type = ‘ManifestFile’ - Manifest file format must
follow the format as `this
document <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_S3DataSource.html#SageMaker-Type-S3DataSource-S3DataType>`__
pointed out. We create the manifest file by using jsonlines package.

.. code:: json

   [ {"prefix": "s3://customer_bucket/some/prefix/"},
   "relative/path/to/custdata-1",
   "relative/path/custdata-2",
   ...
   "relative/path/custdata-N"
   ]

.. code:: ipython3

    !pip install -q jsonlines

.. code:: ipython3

    import jsonlines
    
    # build image list
    manifest_prefix = f's3://{bucket}/{prefix}/images/'
    
    path = './mnist_sample/'
    img_files = [f for f in listdir(path) if isfile(join(path, f))]
    
    manifest_content = [{'prefix': manifest_prefix}]
    manifest_content.extend(img_files)
    
    # write jsonl file
    manifest_file = 'manifest.json'
    with jsonlines.open(manifest_file, mode='w') as writer:
        writer.write(manifest_content)
    
    # upload to S3
    manifest_obj = sagemaker_session.upload_data(path=manifest_file,
                                                 key_prefix=prefix)
    
    # batch transform with manifest file
    transformer.transform(data=manifest_obj, 
                          data_type='ManifestFile',
                          content_type='application/x-image', 
                          wait=False)

Multiple instance
~~~~~~~~~~~~~~~~~

We use ``instance_count > 1`` to create multiple inference instances.
When a batch transform job starts, Amazon SageMaker initializes compute
instances and distributes the inference or preprocessing workload
between them. Batch Transform partitions the Amazon S3 objects in the
input by key and maps Amazon S3 objects to instances. When you have
multiples files, one instance might process input1.csv, and another
instance might process the file named input2.csv.

https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html

.. code:: ipython3

    dist_transformer = estimator.transformer(instance_count=2,
                                             instance_type='ml.c4.xlarge')
    
    dist_transformer.transform(data=inference_inputs, 
                               data_type='S3Prefix',
                               content_type='application/x-image', 
                               wait=False)

