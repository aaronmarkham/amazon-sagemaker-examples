fastai: lesson 1 - Pets example with Amazon SageMaker
=====================================================

Pre-requisites
--------------

This notebook shows how to use the SageMaker Python SDK to run your
fastai library based model in a local container before deploying to
SageMaker’s managed training or hosting environments. This can speed up
iterative testing and debugging while using the same familiar Python SDK
interface. Just change your estimator’s ``train_instance_type`` to
``local``.

In order to use this feature you’ll need to install docker-compose (and
nvidia-docker if training with a GPU).

**Note, you can only run a single local notebook at one time.**

.. code:: ipython3

    import os
    import io
    import subprocess
    
    import PIL
    
    import sagemaker
    from sagemaker.pytorch import PyTorch, PyTorchModel
    from sagemaker.predictor import RealTimePredictor, json_deserializer
    
    from fastai.vision import *

Overview
--------

The **SageMaker Python SDK** helps you deploy your models for training
and hosting in optimized, productions ready containers in SageMaker. The
SageMaker Python SDK is easy to use, modular, extensible and compatible
with TensorFlow, MXNet, PyTorch and Chainer. This tutorial focuses on
how to create a convolutional neural network model to train the `Oxford
IIIT Pet dataset <http://www.robots.ox.ac.uk/~vgg/data/pets/>`__ as per
`Lesson 1 of the fast.ai MOOC
course <https://course.fast.ai/videos/?lesson=1>`__ using **PyTorch in
local mode**.

Set up the environment
~~~~~~~~~~~~~~~~~~~~~~

To setup a new SageMaker notebook instance with fastai library installed
then follow steps outlined
`here <https://course.fast.ai/start_sagemaker.html>`__.

This notebook was created and tested on a single ml.p3.2xlarge notebook
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

If you want to test your training or hosting of your fastai model then
run the following cell to update the Docker daemon default shared memory
to 2gb. Only run this command if you are using the ``ml.p3.2xlarge``
instance type.

.. code:: ipython3

    ! sudo cp daemon.json /etc/docker/daemon.json && sudo pkill -SIGHUP dockerd

.. code:: ipython3

    sagemaker_session = sagemaker.Session()
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-fastai-pets'
    
    role = sagemaker.get_execution_role()

Download the Oxford Pets dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will download the dataset and save locally on our notebook instance.

.. code:: ipython3

    path = untar_data(URLs.PETS); path

Data Preview
~~~~~~~~~~~~

.. code:: ipython3

    path_anno = path/'annotations'
    path_img = path/'images'
    
    fnames = get_image_files(path_img)
    fnames[:5]

Upload the data
~~~~~~~~~~~~~~~

We use the ``sagemaker.Session.upload_data`` function to upload our
datasets to an S3 location. The return value inputs identifies the
location – we will use this later when we start the training job.

.. code:: ipython3

    inputs = sagemaker_session.upload_data(path=path, bucket=bucket, key_prefix=prefix)
    print('input spec (in this case, just an S3 path): {}'.format(inputs))

Construct a script for training and inference
=============================================

Here is the full code that both trains the model and does model
inference.

.. code:: ipython3

    !pygmentize source/pets.py

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

For local training with GPU, we could set this to ``local_gpu``. In this
case, ``instance_type`` was set below based on whether you’re running a
GPU instance. If ``instance_type`` is set to a SageMaker instance type
(e.g. ml.p2.xlarge) then the training will happen on SageMaker.

The parameter ``data_location`` determines where the training data is.
If training locally then it can be set to the local file system to avoid
having to download from S3. If training on SageMaker then it needs to
reference the training data on S3.

After we’ve constructed our ``PyTorch`` object, we fit it using the data
we uploaded to S3. Even though we’re in local mode, using S3 as our data
source makes sense because it maintains consistency with how SageMaker’s
distributed, managed training ingests data.

If you want to train locally then uncomment out all of the lines in the
code block below.

.. code:: ipython3

    # Comment out all lines below if not training locally
    data_location='file://'+str(path)
    instance_type = 'local'
    if subprocess.call('nvidia-smi') == 0:
        ## Set type to GPU if one is present
        instance_type = 'local_gpu'

If you want to train your model on SageMaker then comment out the cell
above and uncomment the cell below.

.. code:: ipython3

    # Comment out all lines below if not training on SageMaker
    #data_location=inputs
    #instance_type = 'ml.p3.2xlarge'

.. code:: ipython3

    pets_estimator = PyTorch(entry_point='source/pets.py',
                             base_job_name='fastai-pets',
                             role=role,
                             framework_version='1.0.0',
                             train_instance_count=1,
                             train_instance_type=instance_type)
    
    pets_estimator.fit(data_location)

Deploy the trained model to prepare for predictions
===================================================

First we need to create a ``PyTorchModel`` object from the estimator.
The ``deploy()`` method on the model object creates an endpoint (in this
case locally) which serves prediction requests in real-time. If the
``instance_type`` is set to a SageMaker instance type (e.g. ml.m5.large)
then the model will be deployed on SageMaker. If the ``instance_type``
parameter is set to ``local`` then it will be deployed locally as a
Docker container and ready for testing locally.

First we need to create a ``RealTimePredictor`` class to accept ``jpeg``
images as input and output JSON. The default behaviour is to accept a
numpy array.

.. code:: ipython3

    class ImagePredictor(RealTimePredictor):
        def __init__(self, endpoint_name, sagemaker_session):
            super(ImagePredictor, self).__init__(endpoint_name, sagemaker_session=sagemaker_session, serializer=None, 
                                                deserializer=json_deserializer, content_type='image/jpeg')

If you want to deploy your model locally then comment out the
``instance_type`` declaration below.

If you want to deploy your model on SageMaker then uncomment the the
``instance_type`` declaration below.

.. code:: ipython3

    # Uncomment out for SageMaker Deployment
    #instance_type = 'ml.c5.large'
    
    pets_model=PyTorchModel(model_data=pets_estimator.model_data,
                            name=pets_estimator._current_job_name,
                            role=role,
                            framework_version=pets_estimator.framework_version,
                            entry_point=pets_estimator.entry_point,
                            predictor_cls=ImagePredictor)
    
    pets_predictor = pets_model.deploy(initial_instance_count=1,
                                           instance_type=instance_type)

Invoking the endpoint
=====================

.. code:: ipython3

    urls = []
    # English Cocker Spaniel
    urls.append('https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/16105011/English-Cocker-Spaniel-Slide03.jpg')
    # Shiba Inu
    urls.append('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Taka_Shiba.jpg/1200px-Taka_Shiba.jpg')
    # German Short haired
    urls.append('https://vetstreet.brightspotcdn.com/dims4/default/232fcc6/2147483647/crop/0x0%2B0%2B0/resize/645x380/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2Fda%2Fa44590a0d211e0a2380050568d634f%2Ffile%2FGerman-Shorthair-Pointer-2-645mk062111.jpg')

.. code:: ipython3

    # get a random selection
    img_bytes = requests.get(random.choice(urls)).content
    img = PIL.Image.open(io.BytesIO(img_bytes))
    img

.. code:: ipython3

    response = pets_predictor.predict(img_bytes)
    response

Clean-up
========

Deleting the local endpoint when you’re finished is important since you
can only run one local endpoint at a time.

.. code:: ipython3

    pets_estimator.delete_endpoint()
