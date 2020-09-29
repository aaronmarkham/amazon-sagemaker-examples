Autoscaling a service with Amazon SageMaker
===========================================

This notebook shows an example of how to use reinforcement learning
technique to address a very common problem in production operation of
software systems: scaling a production service by adding and removing
resources (e.g. servers or EC2 instances) in reaction to dynamically
changing load. This example is a simple toy demonstrating how one might
begin to address this real and challenging problem. We build up a fake
simulated system with daily and weekly variations and occassional
spikes. It also has a delay between when new resources are requested and
when they become available for serving requests. The customized
environment is constructed using Open AI gym and the RL agents are
trained using Amazon SageMaker.

Problem Statement
-----------------

Autoscaling enables services to dynamically update capacity up or down
automatically depending on conditions you define. Today, this requires
setting up alarms, scaling policies, thresholds etc. Under the
customized simulator, the RL problem for autoscaling can be defined as:

1. *Objective*: Optimize profit of a scalable web service by adapting
   instance capacity to load profile. Meanwhile, ensure the
   servers/instances are sufficient when a spike occurs.
2. *Environment*: Custom developed environment that includes the load
   profile. It generates a fake simulated load with daily and weekly
   variations and occasional spikes. The simulated system has a delay
   between when new resources are requested and when they become
   available for serving requests.
3. *State*: A time-weighted combination of previous and current
   observations. At each timestamp, an observation includes current load
   (transactions this minute), number of failed transactions, a boolean
   variable indicating whether the service is in downtime (when
   availability drops below 99.5%), and the current number of active
   machines.
4. *Action*: Remove or add machines. The agent can do both at the same
   time.
5. *Reward*: A customized reward function based on a simple financial
   model. On top of positive reward for successful transactions, we take
   costs for running machines into consideration. We also apply a high
   penalty for downtime.

Using Amazon SageMaker for RL
-----------------------------

Amazon SageMaker allows you to train your RL agents in cloud machines
using docker containers. You do not have to worry about setting up your
machines with the RL toolkits and deep learning frameworks. You can
easily switch between many different machines setup for you, including
powerful GPU machines that give a big speedup. You can also choose to
use multiple machines in a cluster to further speedup training, often
necessary for production level loads.

Pre-requisites
--------------

Roles and permissions
~~~~~~~~~~~~~~~~~~~~~

To get started, we’ll import the Python libraries we need, set up the
environment with a few prerequisites for permissions and configurations.

.. code:: ipython3

    import sagemaker
    import boto3
    import sys
    import os
    import glob
    import re
    import subprocess
    from IPython.display import HTML
    import time
    from time import gmtime, strftime
    sys.path.append("common")
    from misc import get_execution_role, wait_for_s3_object
    from sagemaker.rl import RLEstimator, RLToolkit, RLFramework

Steup S3 buckets
~~~~~~~~~~~~~~~~

Set up the linkage and authentication to the S3 bucket that you want to
use for checkpoint and the metadata.

.. code:: ipython3

    # S3 bucket
    sage_session = sagemaker.session.Session()
    s3_bucket = sage_session.default_bucket()  
    s3_output_path = 's3://{}/'.format(s3_bucket) # SDK appends the job name and output folder
    print("S3 bucket path: {}".format(s3_output_path))

Define Variables
~~~~~~~~~~~~~~~~

We define variables such as the job prefix for the training jobs *and
the image path for the container (only when this is BYOC).*

.. code:: ipython3

    # create unique job name 
    job_name_prefix = 'rl-auto-scaling'

Configure settings
~~~~~~~~~~~~~~~~~~

You can run your RL training jobs on a SageMaker notebook instance or on
your own machine. In both of these scenarios, you can run the following
in either ``local`` or ``SageMaker`` modes. The ``local`` mode uses the
SageMaker Python SDK to run your code in a local container before
deploying to SageMaker. This can speed up iterative testing and
debugging while using the same familiar Python SDK interface. You just
need to set ``local_mode = True``.

.. code:: ipython3

    %%time
    
    # run in local mode?
    local_mode = False

Create an IAM role
~~~~~~~~~~~~~~~~~~

Either get the execution role when running from a SageMaker notebook
``role = sagemaker.get_execution_role()`` or, when running from local
machine, use utils method ``role = get_execution_role()`` to create an
execution role.

.. code:: ipython3

    try:
        role = sagemaker.get_execution_role()
    except:
        role = get_execution_role()
        
    print("Using IAM role arn: {}".format(role))

Install docker for ``local`` mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to work in ``local`` mode, you need to have docker installed.
When running from you local machine, please make sure that you have
docker or docker-compose (for local CPU machines) and nvidia-docker (for
local GPU machines) installed. Alternatively, when running from a
SageMaker notebook instance, you can simply run the following script to
install dependenceis.

Note, you can only run a single local notebook at one time.

.. code:: ipython3

    # only run from SageMaker notebook instance
    if local_mode:
        !/bin/bash ./common/setup.sh

Set up the environment
----------------------

The environment is defined in a Python file called ``autoscalesim.py``
and the file is uploaded on ``/src`` directory.

The environment also implements the ``init()``, ``step()`` and
``reset()`` functions that describe how the environment behaves. This is
consistent with Open AI Gym interfaces for defining an environment.

1. init() - initialize the environment in a pre-defined state
2. step() - take an action on the environment
3. reset()- restart the environment on a new episode
4. [if applicable] render() - get a rendered image of the environment in
   its current state

.. code:: ipython3

    !pygmentize src/autoscalesim.py

Visualize the simulated load
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shape of the simulated load is critical to an auto-scaling
simulation. We use the this toy load simulator for visualization. The
simulator has two components to load: periodic load and spikes. The
periodic load is a simple daily cycle of fixed mean & amplitude, with
multiplicative gaussian noise. The spike load start instantly and decay
linearly until gone, and have a variable random delay between them.

.. code:: ipython3

    # if open AI Gym is not installed
    ! pip install gym

.. code:: ipython3

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    sys.path.append('src')
    import autoscalesim

.. code:: ipython3

    def xy_data(days_to_simulate=3):
        loadsim = autoscalesim.LoadSimulator()
        load = []
        x = np.arange(0, days_to_simulate, 1.0/(24*60))
        for t in x:
            load.append(loadsim.time_step_load())
        load = np.asarray(load)
        return (x, load)

.. code:: ipython3

    plt.rcParams["figure.figsize"] = (20,8)
    
    for n in range(5):  # Draw 5 plots
        (x,y) = xy_data()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='time (days)', ylabel='load (tpm)',
               title='Load simulation #%d' % n)
        ax.grid()
        plt.show()

Configure the presets for RL algorithm
--------------------------------------

The presets that configure the RL training jobs are defined in the
``preset-autoscale-ppo.py`` file which is also uploaded on the ``/src``
directory. Using the preset file, you can define agent parameters to
select the specific agent algorithm. You can also set the environment
parameters, define the schedule and visualization parameters, and define
the graph manager. The schedule presets will define the number of heat
up steps, periodic evaluation steps, training steps between evaluations.

These can be overridden at runtime by specifying the ``RLCOACH_PRESET``
hyperparameter. Additionally, it can be used to define custom
hyperparameters.

.. code:: ipython3

    !pygmentize src/preset-autoscale-ppo.py

Write the Training Code
-----------------------

The training code is written in the file “train-coach.py” which is
uploaded in the /src directory. First import the environment files and
the preset files, and then define the ``main()`` function.

.. code:: ipython3

    !pygmentize src/train-coach.py

Train the RL model using the Python SDK Script mode
---------------------------------------------------

If you are using local mode, the training will run on the notebook
instance. When using SageMaker for training, you can select a GPU or CPU
instance. The RLEstimator is used for training RL jobs.

1. Specify the source directory where the environment, presets and
   training code is uploaded.
2. Specify the entry point as the training code
3. Specify the choice of RL toolkit and framework. This automatically
   resolves to the ECR path for the RL Container.
4. Define the training parameters such as the instance count, job name,
   S3 path for output and job name.
5. Specify the hyperparameters for the RL agent algorithm. The
   ``RLCOACH_PRESET`` can be used to specify the RL agent algorithm you
   want to use.
6. [Optional] Choose the metrics that you are interested in capturing in
   your logs. These can also be visualized in CloudWatch and SageMaker
   Notebooks. The metrics are defined using regular expression matching.

.. code:: ipython3

    %%time
    
    if local_mode:
        instance_type = 'local'
    else:
        instance_type = "ml.m4.xlarge"
            
    estimator = RLEstimator(entry_point="train-coach.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            toolkit=RLToolkit.COACH,
                            toolkit_version='0.11.0',
                            framework=RLFramework.TENSORFLOW,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            hyperparameters = {
                              "RLCOACH_PRESET": "preset-autoscale-ppo",
                              "rl.agent_params.algorithm.discount": 0.9,
                              "rl.evaluation_steps:EnvironmentEpisodes": 8,
                              # save model for deployment
                              "save_model": 1
                            }
                        )
    estimator.fit()

Store intermediate training output and model checkpoints
--------------------------------------------------------

The output from the training job above is either stored in a local
directory (``local`` mode) or on S3 (``SageMaker``) mode.

.. code:: ipython3

    %%time
    
    job_name=estimator._current_job_name
    print("Job name: {}".format(job_name))
    
    s3_url = "s3://{}/{}".format(s3_bucket,job_name)
    
    if local_mode:
        output_tar_key = "{}/output.tar.gz".format(job_name)
    else:
        output_tar_key = "{}/output/output.tar.gz".format(job_name)
    
    intermediate_folder_key = "{}/output/intermediate/".format(job_name)
    output_url = "s3://{}/{}".format(s3_bucket, output_tar_key)
    intermediate_url = "s3://{}/{}".format(s3_bucket, intermediate_folder_key)
    
    print("S3 job path: {}".format(s3_url))
    print("Output.tar.gz location: {}".format(output_url))
    print("Intermediate folder path: {}".format(intermediate_url))
        
    tmp_dir = "/tmp/{}".format(job_name)
    os.system("mkdir {}".format(tmp_dir))
    print("Create local folder {}".format(tmp_dir))

Visualization
-------------

Plot rate of learning
~~~~~~~~~~~~~~~~~~~~~

We can view the rewards during training using the code below. This
visualization helps us understand how the performance of the model
represented as the reward has improved over time.

.. code:: ipython3

    %matplotlib inline
    import pandas as pd
    
    csv_file_name = "worker_0.simple_rl_graph.main_level.main_level.agent_0.csv"
    key = os.path.join(intermediate_folder_key, csv_file_name)
    wait_for_s3_object(s3_bucket, key, tmp_dir)
    
    csv_file = "{}/{}".format(tmp_dir, csv_file_name)
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['Training Reward'])
    x_axis = 'Episode #'
    y_axis = 'Training Reward'
    
    plt = df.plot(x=x_axis,y=y_axis, figsize=(12,5), legend=True, style='b-')
    plt.set_ylabel(y_axis);
    plt.set_xlabel(x_axis);

Evaluation of RL models
-----------------------

We use the latest checkpointed model to run evaluation for the RL Agent.

Load the checkpointed models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkpointed data from the previously trained models will be passed on
for evaluation / inference in the ``checkpoint`` channel. In ``local``
mode, we can simply use the local directory, whereas in the
``SageMaker`` mode, it needs to be moved to S3 first.

Since TensorFlow stores ckeckpoint file containes absolute paths from
when they were generated (see
`issue <https://github.com/tensorflow/tensorflow/issues/9146>`__), we
need to replace the absolute paths to relative paths. This is
implemented within ``evaluate-coach.py``

.. code:: ipython3

    %%time
    
    wait_for_s3_object(s3_bucket, output_tar_key, tmp_dir)  
    
    if not os.path.isfile("{}/output.tar.gz".format(tmp_dir)):
        raise FileNotFoundError("File output.tar.gz not found")
    os.system("tar -xvzf {}/output.tar.gz -C {}".format(tmp_dir, tmp_dir))
    
    if local_mode:
        checkpoint_dir = "{}/data/checkpoint".format(tmp_dir)
    else:
        checkpoint_dir = "{}/checkpoint".format(tmp_dir)
    
    print("Checkpoint directory {}".format(checkpoint_dir))

.. code:: ipython3

    %%time
    
    if local_mode:
        checkpoint_path = 'file://{}'.format(checkpoint_dir)
        print("Local checkpoint file path: {}".format(checkpoint_path))
    else:
        checkpoint_path = "s3://{}/{}/checkpoint/".format(s3_bucket, job_name)
        if not os.listdir(checkpoint_dir):
            raise FileNotFoundError("Checkpoint files not found under the path")
        os.system("aws s3 cp --recursive {} {}".format(checkpoint_dir, checkpoint_path))
        print("S3 checkpoint file path: {}".format(checkpoint_path))

Run the evaluation step
~~~~~~~~~~~~~~~~~~~~~~~

Use the checkpointed model to run the evaluation step.

.. code:: ipython3

    %%time
    
    estimator_eval = RLEstimator(role=role,
                          source_dir='src/',
                          dependencies=["common/sagemaker_rl"],
                          toolkit=RLToolkit.COACH,
                          toolkit_version='0.11.0',
                          framework=RLFramework.TENSORFLOW,
                          entry_point="evaluate-coach.py",
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          hyperparameters = {
                                     "RLCOACH_PRESET": "preset-autoscale-ppo",
                                     "evaluate_steps": 10001*2 # evaluate on 2 episodes
                                 }
                        )
    estimator_eval.fit({'checkpoint': checkpoint_path})

Hosting
-------

Once the training is done, we can deploy the trained model as an Amazon
SageMaker real-time hosted endpoint. This will allow us to make
predictions (or inference) from the model. Note that we don’t have to
host on the same insantance (or type of instance) that we used to train.
The endpoint deployment can be accomplished as follows:

.. code:: ipython3

    predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')

Inference
~~~~~~~~~

Now that the trained model is deployed at an endpoint that is
up-and-running, we can use this endpoint for inference. The format of
input should match that of ``observation_space`` in the defined
environment. In this example, the observation space is a 25 dimensional
vector formulated from previous and current observations. For the sake
of space, this demo doesn’t include the non-trivial construction
process. Instead, we provide a dummy input below. For more details,
please check ``src/autoscalesim.py``.

.. code:: ipython3

    observation = np.arange(1, 26)
    action = predictor.predict(observation)
    print(action)

Delete the Endpoint
~~~~~~~~~~~~~~~~~~~

Having an endpoint running will incur some costs. Therefore as a
clean-up job, we should delete the endpoint.

.. code:: ipython3

    predictor.delete_endpoint()
