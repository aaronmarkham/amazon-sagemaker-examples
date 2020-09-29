# Distributed Neural Network Compression using Reinforcement Learning
---------------------------------------------------------------------

Introduction
------------

In this notebook, we demonstrate how to compress a neural network
(Resnet-18) using reinforcement learning. The work in this notebook is
based on [1], even though heavily adapted to work with Amazon SageMaker
RL. The following are the key highlights of AWS SageMaker RL
demonstrated in this notebook. 1. A custom environment for neural
network compression. 2. Usage of the Ray container in SageMaker with
distributed training. 3. Using tensorflow within the environment in the
container. 4. Network compression through RL.

[1] `Ashok, Anubhav, Nicholas Rhinehart, Fares Beainy, and Kris M.
Kitani. “N2N learning: network to network compression via policy
gradient reinforcement learning.” arXiv preprint arXiv:1709.06030
(2017) <https://arxiv.org/abs/1709.06030>`__.

The RL problem here can be defined as follows:

**Objective:** Search and find the smallest possible network
architecture from a pre-trained network architecture, while producing
the best accuracy possible.

**Environment:** A custom developed environment that accepts a Boolean
array of layers to remove from the RL agent and produces an observation
that is some description of every layer in the network. This environment
is sub-classed from OpenAI Gym’s environment. It can be found in the
`environment file <./src/environment.py>`__.

**State:** For every layer in the network there is a :math:`1 \times 8`
array of floats. In Resnet-18, there are 40 removable layers.

**Action:** A boolean array one for each layer. ``False`` implies don’t
remove the layer and ``True`` implies remove the layer.

**Reward:** Consider, :math:`C = 1 - \frac{M_s}{M}`, where :math:`C` is
the compression ratio, :math:`M_s` is the number of parameters in a
network that the RL agent explores, :math:`M` is the number of
parameters in the master network to be compressed. The reward is
:math:`r = \frac{CA_s}{(2-C)A}`, where :math:`A_s` is the accuracy of
the network that the RL agent explores and :math:`A` is the accuracy of
the master network. If the explored network can’t even train or is
out-of-memory, the reward is :math:`r = -1`.

Attribution
-----------

1. Cifar10 Dataset: We use the cifar10 dataset in this notebook [2] to
   conduct our experiments.
2. We rely on the open-source codebase from `tensorflow/models
   repository <https://github.com/tensorflow/models>`__, released under
   Apache 2.0 to build the backend resnet models. Please refer to the
   `license <https://github.com/tensorflow/models/blob/master/LICENSE>`__
   of that repository.

[2] `Learning Multiple Layers of Features from Tiny Images, Alex
Krizhevsky, 2009. <https://www.cs.toronto.edu/~kriz/cifar.html>`__

Pre-requisites
--------------

Roles and permissions
~~~~~~~~~~~~~~~~~~~~~

To get started, we’ll import the sagemaker python library and setup the
permissions and IAM role.

.. code:: ipython3

    from time import gmtime, strftime
    import sagemaker 
    role = sagemaker.get_execution_role()

Auxiliary services and settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run this notebook, we require the use of AWS services all of which
are accessible right from the sagemaker library using the role that we
just created. For instance, we need an S3 bucket where we need to store
our output models, which can be created as follows:

.. code:: ipython3

    sage_session = sagemaker.session.Session()
    s3_bucket = sage_session.default_bucket()  
    s3_output_path = 's3://{}/'.format(s3_bucket)
    print("S3 bucket path: {}".format(s3_output_path))

For logs on cloudwatch or tracking the job on sagemaker console, we need
a job_name. Let us create a prefix

.. code:: ipython3

    job_name_prefix = 'rl-nnc'

Running the RL containers in sagemaker produces logs on cloudwatch. It
is tedious to migrate to cloudwatch just to monitor the algorithm logs.
Let us therefore create some metric crawlers using simple regex that
will help us bring the detail we need here. Since we are using the Ray
container image, the following regex definitions will work.

.. code:: ipython3

    float_regex = "[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
    
    metric_definitions = [
        {'Name': 'episode_reward_mean',
         'Regex': 'episode_reward_mean: (%s)' % float_regex},
        {'Name': 'episode_reward_max',
         'Regex': 'episode_reward_max: (%s)' % float_regex},
        {'Name': 'episode_reward_min',
         'Regex': 'episode_reward_min: (%s)' % float_regex},
    ]

The gamification of neural network compression
----------------------------------------------

We now need an environment for our RL agent to work on. This environment
has the following behavior. It accepts from our RL agent, a list of
layers to remove from the master network. Once it received its list, it
will create a network with the removed layers. It will then use the
master network’s original weights to initialize the smaller network.
Once initialized, the environment will train the small network with both
a cross-entropy loss and a distillation loss from the master network as
described in [2]. It will then output the reward.

[2] `Hinton, G., Vinyals, O. and Dean, J., 2015. Distilling the
knowledge in a neural network. arXiv preprint
arXiv:1503.02531. <https://arxiv.org/abs/1503.02531>`__

A custom gym environment
~~~~~~~~~~~~~~~~~~~~~~~~

To construct and formalize this world, we use the gym environment’s
formulations. The environment itself is described in the
`environment.py <./src/environment.py>`__ file. The environment
implements a constructor that sets it up, a ``step`` method that accepts
actions and produces reward, and other functions that describe how the
environment behaves. This is consistent with OpenAI Gym interfaces for
defining an environment. Let us briefly look at the environment
definition below.

.. code:: ipython3

    !pygmentize ./src/environment.py

Of prominent notice in this file is the ``NetworkCompression`` class
described in the
`network_compression.py <./src/tensorflow_resnet/network_compression.py>`__.
This file contains all of the tensorflow implementation of ResNet-18,
its training, distillation and others that are abstracted away from the
environment. By changing the definition here, other networks can be
implemented as well without altering the environment file.

Setup data and upload to S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to download the dataset and have it uploaded to S3. We
use some helper codes from `tensorflow’s
model <https://github.com/tensorflow/models>`__ repository to download
and setup the `Cifar10
dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`__. The cifar10
dataset contains 50,000 training images and 10,000 validation images
each :math:`32 \times 32` in RGB. Running the cell below will download
the data into ``cifar10_data`` directory and upload to S3.

.. code:: ipython3

    %%time
    !python src/tensorflow_resnet/dataset/cifar10_download_and_extract.py
    cifar_inputs = sage_session.upload_data(path='cifar10_data', key_prefix='cifar10_data')

Prepare teacher weights
~~~~~~~~~~~~~~~~~~~~~~~

A teacher network is used to train the child network using distillation
loss. The code uses a pickle file dumped from the checkpoint for loading
teacher weights and already has a pickle file for cifar10 dataset in the
teacher directory

.. code:: ipython3

    _ = sage_session.upload_data(path='teacher', key_prefix='cifar10_data')

The RL agent
~~~~~~~~~~~~

For an RL agent we use the `asynchronous advantage actor-critic
(A3C) <https://arxiv.org/abs/1602.01783>`__ agent from the `Ray
toolkit <https://ray.readthedocs.io/en/latest/example-a3c.html>`__. We
run training with 5 rollouts (architectures searched). We train the
agent for 20 iterations in a GPU machine. The GPUs are also used to
train the network in the environment. The A3C definitions and parameters
of training can be found in the launcher file. We can also find the code
that will register the custom environment that we have created below.

.. code:: ipython3

    !pygmentize ./src/train-ray.py

Training
~~~~~~~~

Now that everything is setup, we can run our training job. For the
training, we can use ``sagemaker.rl.RLEstimator``. This class is a
simple API that will take all our parameters and create the sagemker job
for us. The following cell will do this. Refer the cell for description
of each parameter.

.. code:: ipython3

    from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
    
    estimator = RLEstimator(entry_point="train-ray.py", # Our launcher code
                            source_dir='src', # Directory where the supporting files are at. All of this will be
                                              # copied into the container.
                            dependencies=["common/sagemaker_rl"], # some other utils files.
                            toolkit=RLToolkit.RAY, # We want to run using the Ray toolkit against the ray container image.
                            framework=RLFramework.TENSORFLOW, # The code is in tensorflow backend.
                            toolkit_version='0.5.3', # Toolkit version. This will also choose an apporpriate tf version.                        
                            role=role, # The IAM role that we created at the begining.
                            train_instance_type="ml.p3.2xlarge", # Since we want to run fast, lets run on GPUs.
                            train_instance_count=2, # Single instance will also work, but running distributed makes things 
                                                    # fast, particularly in the case of multiple rollout training.
                            output_path=s3_output_path, # The path where we can expect our trained model.
                            base_job_name=job_name_prefix, # This is the name we setup above to be to track our job.
                            hyperparameters = {      # Some hyperparameters for Ray toolkit to operate.
                              "s3_bucket": s3_bucket,
                              "rl.training.stop.training_iteration": 1, # Number of iterations.
                              "rl.training.checkpoint_freq": 1,
                            },
                            metric_definitions=metric_definitions, # This will bring all the logs out into the notebook.
                        )

Now that the training job is setup, all that is needed is to run the
``fit`` call with the appropriate input buckets. The training should
take about 25 mins to complete.

.. code:: ipython3

    estimator.fit(cifar_inputs)

Process Outputs
---------------

Now that the training is complete, we can look at the best compressed
network architecture were found during training. The list of networks
with their accuracies and other metrics are stored in the output S3
bucket. This can be downloaded from S3. The file is named as
``output.tar.gz`` and is at the same location as the model file
``model.tar.gz``. Let us download and extract this output directory. But
before that, we need to clean any files leftover from previous runs, if
any.

.. code:: ipython3

    !rm *_metrics.txt

.. code:: ipython3

    model_data = estimator.model_data
    print('Model data path: ', model_data)
    output_data = model_data.replace('model', 'output')
    print('Output data path: ', output_data)
    
    #Download the output file and extract.
    !aws s3 cp {output_data} .
    !tar xvfz output.tar.gz

Since the training runs across multiple workers, each worker stores the
best model that it generates in it’s own file. We will consolidate the
files from all the workers to get the top networks from the training
job.

.. code:: ipython3

    metrics_file_name = 'consolidated_metrics.csv'
    !cat *_metrics.txt > {metrics_file_name}
    import pandas as pd
    df = pd.read_csv(metrics_file_name, sep=',', names = ["reward", "x-factor", "accuracy", "dir"])
    df = df.sort_values('reward')
    print(df.tail(10).to_string(index=False))

The code above prints the best networks that were found during training
and these are printed in the ascending order of reward. ``x-factor`` is
how much compression has been performed and ``accuracy`` is the accuracy
of the compressed model (trained only for 1 epoch). The ``dir`` is the
directory where the compressed model is stored. This is in comparison
with the master accuracy of ``0.81``. While the best models produced
here are trained, it always gives a performance boost when fine-tuned.
We only train the network for a few epochs during reward calculation and
hence the accuracy of the network can further be improved by
fine-tuning. This can be done by using the checkpoint of the best
network and fine-tuning it further for more epochs. While we only ran
``1`` iteration for the sake of demonstration, running more iterations
will provide better results. For instance, by running for ``1500``
timesteps, we were able to achieve ``5.7x`` compression with ``0.71``
accuracy, which when fine-tuned further gave an accuracy of ``.80``.
