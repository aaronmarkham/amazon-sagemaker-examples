TensorFlow Script Mode - Using Shell scripts
============================================

Starting from TensorFlow version 1.11, you can use a shell script as
your training entry point. Shell scripts are useful for many use cases
including:

-  Invoking Python scripts with specific parameters
-  Configuring framework dependencies
-  Training using different programming languages

For this example, we use `a Keras implementation of the Deep Dream
algorithm <https://github.com/keras-team/keras/blob/2.2.4/examples/deep_dream.py>`__.
We can use the same technique for other scripts or repositories
including `TensorFlow Model
Zoo <https://github.com/tensorflow/models>`__ and `TensorFlow benchmark
scripts <https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks>`__.

Getting the image for training
==============================

For training data, let’s download a public domain image:

.. code:: ipython3

    import os
    data_dir = os.path.join(os.getcwd(), 'training')
    
    os.makedirs(data_dir, exist_ok=True)
    data_dir

.. code:: ipython3

    !wget -O training/dark-forest-landscape.jpg https://www.goodfreephotos.com/albums/other-landscapes/dark-forest-landscape.jpg

.. code:: ipython3

    from IPython.display import Image
    Image(filename='training/dark-forest-landscape.jpg') 

Download the training script
----------------------------

Let’s start by downloading the
`deep_dream <https://github.com/keras-team/keras/blob/2.2.4/examples/deep_dream.py>`__
example script from Keras repository. This script takes an image and
uses deep dream algorithm to generate transformations of that image.

.. code:: ipython3

    !wget https://raw.githubusercontent.com/keras-team/keras/2.2.4/examples/deep_dream.py

The script **deep_dream.py** takes two positional arguments: -
``base_image_path``: Path to the image to transform. -
``result_prefix``: Prefix of all generated images.

Creating the launcher script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need to create a launcher script that sets the ``base_image_path``
and ``result_prefix``, and invokes **deep_dream.py**:

.. code:: ipython3

    %%writefile launcher.sh 
    
    BASE_IMAGE_PATH="${SM_CHANNEL_TRAINING}/dark-forest-landscape.jpg"
    RESULT_PREFIX="${SM_MODEL_DIR}/dream"
    
    python deep_dream.py ${BASE_IMAGE_PATH} ${RESULT_PREFIX}
    
    echo "Generated image $(ls ${SM_MODEL_DIR})"

**SM_CHANNEL_TRAINING** and **SM_MODEL** are environment variables
created by the SageMaker TensorFlow Container in the beginning of
training. Let’s take a more detailed look at then:

-  **SM_MODEL_DIR**: the directory inside the container where the
   training model data must be saved inside the container, i.e.
   /opt/ml/model.
-  **SM_TRAINING_CHANNEL**: the directory containing data in the
   ‘training’ channel.

For more information about training environment variables, please visit
`SageMaker
Containers <https://github.com/aws/sagemaker-containers#list-of-provided-environment-variables-by-sagemaker-containers>`__.

Test locally using SageMaker Python SDK TensorFlow Estimator
------------------------------------------------------------

You can use the SageMaker Python SDK TensorFlow estimator to easily
train locally and in SageMaker. Let’s set **launcher.sh** as the
entry-point and **deep_dream.py** as a dependency:

.. code:: ipython3

    entry_point='launcher.sh'
    dependencies=['deep_dream.py']

For more information about the arguments ``entry_point`` and
``dependencies`` see the `SageMaker
TensorFlow <https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/README.rst#sagemakertensorflowtensorflow-class>`__
documentation.

This notebook shows how to use the SageMaker Python SDK to run your code
in a local container before deploying to SageMaker’s managed training or
hosting environments. Just change your estimator’s train_instance_type
to local or local_gpu. For more information, see:
https://github.com/aws/sagemaker-python-sdk#local-mode.

In order to use this feature you’ll need to install docker-compose (and
nvidia-docker if training with a GPU). Running following script will
install docker-compose or nvidia-docker-compose and configure the
notebook environment for you.

Note, you can only run a single local notebook at a time.

.. code:: ipython3

    !/bin/bash ./setup.sh

Let’s train locally here to make sure everything runs smoothly first.

.. code:: ipython3

    train_instance_type='local'

We create the TensorFlow Estimator, passing the flag
``script_mode=True``. For more information about script mode, see
https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/README.rst#preparing-a-script-mode-training-script:

.. code:: ipython3

    import sagemaker
    from sagemaker.tensorflow import TensorFlow
    
    estimator = TensorFlow(entry_point=entry_point,
                           dependencies=dependencies,
                           train_instance_type='local',
                           train_instance_count=1,
                           role=sagemaker.get_execution_role(),
                           framework_version='1.14',
                           py_version='py3',
                           script_mode=True)

To start a training job, we call ``estimator.fit(inputs)``, where inputs
is a dictionary where the keys, named **channels**, have values pointing
to the data location:

.. code:: ipython3

    inputs = {'training': f'file://{data_dir}'}
    
    estimator.fit(inputs)

``estimator.model_data`` contains the S3 location where the contents of
**/opt/ml/model** were save as tar.gz file. Let’s untar and download the
model:

.. code:: ipython3

    !aws s3 cp {estimator.model_data} model.tar.gz
    !tar -xvzf model.tar.gz

We can see the resulting image now:

.. code:: ipython3

    from IPython.display import Image
    Image(filename='dream.png')

Training in SageMaker
=====================

After you test the training job locally, upload the dataset to an S3
bucket so SageMaker can access the data during training:

.. code:: ipython3

    import sagemaker
    
    training_data = sagemaker.Session().upload_data(path='training', key_prefix='datasets/deep-dream')

The ``upload_data`` call above returns an S3 location that can be used
during the SageMaker Training Job

.. code:: ipython3

    training_data

To train in SageMaker: change the estimator argument
**train_instance_type** to any SageMaker ML Instance Type available for
training.

.. code:: ipython3

    estimator = TensorFlow(entry_point='launcher.sh',
                           dependencies=['deep_dream.py'],
                           train_instance_type='ml.c4.xlarge',
                           train_instance_count=1,
                           role=sagemaker.get_execution_role(),
                           framework_version='1.14',
                           py_version='py3',
                           script_mode=True)


The ``estimator.fit`` call bellow starts training and creates a data
channel named ``training`` with the contents of the S3 location
``training_data``.

.. code:: ipython3

    estimator.fit(training_data)
