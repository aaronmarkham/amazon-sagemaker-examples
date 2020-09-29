Using TensorFlow Scripts in SageMaker - Quickstart
==================================================

Starting with TensorFlow version 1.11, you can use SageMaker’s
TensorFlow containers to train TensorFlow scripts the same way you would
train outside SageMaker. This feature is named **Script Mode**.

This example uses `Multi-layer Recurrent Neural Networks (LSTM, RNN) for
character-level language models in Python using
Tensorflow <https://github.com/sherjilozair/char-rnn-tensorflow>`__. You
can use the same technique for other scripts or repositories, including
`TensorFlow Model Zoo <https://github.com/tensorflow/models>`__ and
`TensorFlow benchmark
scripts <https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks>`__.

Get the data
~~~~~~~~~~~~

For training data, we use plain text versions of Sherlock Holmes
stories. Let’s create a folder named **sherlock** to store our dataset:

.. code:: ipython3

    import os
    data_dir = os.path.join(os.getcwd(), 'sherlock')
    
    os.makedirs(data_dir, exist_ok=True)

We need to download the dataset to this folder:

.. code:: ipython3

    !wget https://sherlock-holm.es/stories/plain-text/cnus.txt --force-directories --output-document=sherlock/input.txt

Preparing the training script
-----------------------------

For training scripts, let’s use Git integration for SageMaker Python SDK
here. That is, you can specify a training script that is stored in a
GitHub, CodeCommit or other Git repository as the entry point for the
estimator, so that you don’t have to download the scripts locally. If
you do so, source directory and dependencies should be in the same repo
if they are needed.

To use Git integration, pass a dict ``git_config`` as a parameter when
you create the ``TensorFlow`` Estimator object. In the ``git_config``
parameter, you specify the fields ``repo``, ``branch`` and ``commit`` to
locate the specific repo you want to use. If authentication is required
to access the repo, you can specify fields ``2FA_enabled``,
``username``, ``password`` and token accordingly.

The scripts we want to use for this example is stored in GitHub repo
https://github.com/awslabs/amazon-sagemaker-examples/tree/training-scripts,
under the branch ``training-scripts``. It is a public repo so we don’t
need authentication to access it. Let’s specify the ``git_config``
argument here:

.. code:: ipython3

    git_config = {'repo': 'https://github.com/awslabs/amazon-sagemaker-examples.git', 'branch': 'training-scripts'}

Note that we did not specify ``commit`` in ``git_config`` here, so the
latest commit of the specified repo and branch will be used by default.

The scripts we will use are under the ``char-rnn-tensorflow`` directory
in the repo. The directory also includes a
`README.md <https://github.com/awslabs/amazon-sagemaker-examples/blob/training-scripts/README.md#basic-usage>`__
with an overview of the project, requirements, and basic usage:

   .. rubric:: **Basic Usage**
      :name: basic-usage

   *To train with default parameters on the tinyshakespeare corpus,
   run*\ **python train.py**\ *. To access all the parameters
   use*\ **python train.py –help.**

`train.py <https://github.com/awslabs/amazon-sagemaker-examples/blob/training-scripts/char-rnn-tensorflow/train.py#L11>`__
uses the `argparse <https://docs.python.org/3/library/argparse.html>`__
library and requires the following arguments:

.. code:: python

   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   # Data and model checkpoints directories
   parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare', help='data directory containing input.txt with training examples')
   parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
   ...
   args = parser.parse_args()

When SageMaker training finishes, it deletes all data generated inside
the container with exception of the directories ``_/opt/ml/model_`` and
``_/opt/ml/output_``. To ensure that model data is not lost during
training, training scripts are invoked in SageMaker with an additional
argument ``--model_dir``. The training script should save the model data
that results from the training job to this directory..

The training script executes in the container as shown bellow:

.. code:: bash

   python train.py --num-epochs 1 --data_dir /opt/ml/input/data/training --model_dir /opt/ml/model

Test locally using SageMaker Python SDK TensorFlow Estimator
------------------------------------------------------------

You can use the SageMaker Python SDK
```TensorFlow`` <https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/README.rst#training-with-tensorflow>`__
estimator to easily train locally and in SageMaker.

Let’s start by setting the training script arguments ``--num_epochs``
and ``--data_dir`` as hyperparameters. Remember that we don’t need to
provide ``--model_dir``:

.. code:: ipython3

    hyperparameters = {'num_epochs': 1, 'data_dir': '/opt/ml/input/data/training'}

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

To train locally, you set ``train_instance_type`` to
`local <https://github.com/aws/sagemaker-python-sdk#local-mode>`__:

.. code:: ipython3

    train_instance_type='local'

We create the ``TensorFlow`` Estimator, passing the ``git_config``
argument and the flag ``script_mode=True``. Note that we are using Git
integration here, so ``source_dir`` should be a relative path inside the
Git repo; otherwise it should be a relative or absolute local path. the
``Tensorflow`` Estimator is created as following:

.. code:: ipython3

    import os
    
    import sagemaker
    from sagemaker.tensorflow import TensorFlow
    
    
    estimator = TensorFlow(entry_point='train.py',
                           source_dir='char-rnn-tensorflow',
                           git_config=git_config,
                           train_instance_type=train_instance_type,
                           train_instance_count=1,
                           hyperparameters=hyperparameters,
                           role=sagemaker.get_execution_role(), # Passes to the container the AWS role that you are using on this notebook
                           framework_version='1.15.2',
                           py_version='py3',
                           script_mode=True)

To start a training job, we call ``estimator.fit(inputs)``, where inputs
is a dictionary where the keys, named **channels**, have values pointing
to the data location. ``estimator.fit(inputs)`` downloads the TensorFlow
container with TensorFlow Python 3, CPU version, locally and simulates a
SageMaker training job. When training starts, the TensorFlow container
executes **train.py**, passing ``hyperparameters`` and ``model_dir`` as
script arguments, executing the example as follows:

.. code:: bash

   python -m train --num-epochs 1 --data_dir /opt/ml/input/data/training --model_dir /opt/ml/model

.. code:: ipython3

    inputs = {'training': f'file://{data_dir}'}
    
    estimator.fit(inputs)

Let’s explain the values of ``--data_dir`` and ``--model_dir`` with more
details:

-  **/opt/ml/input/data/training** is the directory inside the container
   where the training data is downloaded. The data is downloaded to this
   folder because ``training`` is the channel name defined in
   ``estimator.fit({'training': inputs})``. See `training
   data <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-trainingdata>`__
   for more information.

-  **/opt/ml/model** use this directory to save models, checkpoints, or
   any other data. Any data saved in this folder is saved in the S3
   bucket defined for training. See `model
   data <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-envvariables>`__
   for more information.

Reading additional information from the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, a user script needs additional information from the container
that is not available in ``hyperparameters``. SageMaker containers write
this information as **environment variables** that are available inside
the script.

For example, the example above can read information about the
``training`` channel provided in the training job request by adding the
environment variable ``SM_CHANNEL_TRAINING`` as the default value for
the ``--data_dir`` argument:

.. code:: python

   if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     # reads input channels training and testing from the environment variables
     parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

Script mode displays the list of available environment variables in the
training logs. You can find the `entire list
here <https://github.com/aws/sagemaker-containers/blob/master/README.rst#list-of-provided-environment-variables-by-sagemaker-containers>`__.

Training in SageMaker
=====================

After you test the training job locally, upload the dataset to an S3
bucket so SageMaker can access the data during training:

.. code:: ipython3

    import sagemaker
    
    inputs = sagemaker.Session().upload_data(path='sherlock', key_prefix='datasets/sherlock')

The returned variable inputs above is a string with a S3 location which
SageMaker Tranining has permissions to read data from.

.. code:: ipython3

    inputs

To train in SageMaker: - change the estimator argument
``train_instance_type`` to any SageMaker ml instance available for
training. - set the ``training`` channel to a S3 location.

.. code:: ipython3

    estimator = TensorFlow(entry_point='train.py',
                           source_dir='char-rnn-tensorflow',
                           git_config=git_config,
                           train_instance_type='ml.c4.xlarge', # Executes training in a ml.c4.xlarge instance
                           train_instance_count=1,
                           hyperparameters=hyperparameters,
                           role=sagemaker.get_execution_role(),
                           framework_version='1.15.2',
                           py_version='py3',
                           script_mode=True)
                 
    
    estimator.fit({'training': inputs})
