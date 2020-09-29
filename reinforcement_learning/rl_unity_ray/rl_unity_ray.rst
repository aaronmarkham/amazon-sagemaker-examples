Unity3D Game with Amazon SageMaker RL
=====================================

--------------

Introduction
------------

`Unity <https://unity.com/>`__ is currently the most popular gaming
engine used by game developers around the world. Unity engine can be
used to create 3D, 2D, virtual reality, and augmented reality games, as
well as simulations and other experiences.
`ML-Agents <https://github.com/Unity-Technologies/ml-agents>`__ is an
open-sourced toolkit developed by Unity to enable games and simulations
to serve as environments for training intelligent agents. It provides
capabilities on how to interact with Unity executables as well as how to
train a RL agent. For how to use ML-Agents to train a Unity game agent
on SageMaker, please refer to this
`notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/unity_ml_agents/unity_mlagents_learn.ipynb>`__.

In this notebook, we are using capabilities from ML-Agent to interact
with Unity executables, but leveraging Amazon SageMaker RL to train the
game agent. We will show how you can apply reinforcement learning to
train an agent to play against a Unity game with Amazon SageMaker RL.

Pre-requisites
--------------

Imports
~~~~~~~

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
    import numpy as np
    from IPython.display import HTML
    import time
    from time import gmtime, strftime
    sys.path.append("common")
    from misc import get_execution_role, wait_for_s3_object
    from docker_utils import build_and_push_docker_image
    from sagemaker.rl import RLEstimator, RLToolkit, RLFramework

Setup S3 bucket
~~~~~~~~~~~~~~~

Set up the linkage and authentication to the S3 bucket that you want to
use for checkpoint and the metadata.

.. code:: ipython3

    sage_session = sagemaker.session.Session()
    s3_bucket = sage_session.default_bucket()  
    s3_output_path = 's3://{}/'.format(s3_bucket)
    print("S3 bucket path: {}".format(s3_output_path))

Define Variables
~~~~~~~~~~~~~~~~

We define variables such as the job prefix for the training jobs *and
the image path for the container (only when this is BYOC).*

.. code:: ipython3

    # create a descriptive job name 
    job_name_prefix = 'rl-unity-ray'

Configure where training happens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can train your RL training jobs using the SageMaker notebook
instance or local notebook instance. In both of these scenarios, you can
run the following in either local or SageMaker modes. The local mode
uses the SageMaker Python SDK to run your code in a local container
before deploying to SageMaker. This can speed up iterative testing and
debugging while using the same familiar Python SDK interface. You just
need to set ``local_mode = True``.

.. code:: ipython3

    # run in local_mode on this machine, or as a SageMaker TrainingJob?
    local_mode = False
    
    if local_mode:
        instance_type = 'local'
    else:
        # If on SageMaker, pick the instance type
        instance_type = "ml.c5.2xlarge"

Create an IAM role
~~~~~~~~~~~~~~~~~~

Either get the execution role when running from a SageMaker notebook
instance ``role = sagemaker.get_execution_role()`` or, when running from
local notebook instance, use utils method
``role = get_execution_role()`` to create an execution role.

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
docker and docker-compose (for local CPU machines) and nvidia-docker
(for local GPU machines) installed. Alternatively, when running from a
SageMaker notebook instance, you can simply run the following script to
install dependenceis.

Note, you can only run a single local notebook at one time.

.. code:: ipython3

    # only run from SageMaker notebook instance
    if local_mode:
        !/bin/bash ./common/setup.sh

Build docker container
----------------------

We must build a custom docker container with Roboschool installed. This
takes care of everything:

1. Fetching base container image
2. Installing Roboschool and its dependencies
3. Uploading the new container image to ECR

This step can take a long time if you are running on a machine with a
slow internet connection. If your notebook instance is in SageMaker or
EC2 it should take 3-10 minutes depending on the instance type.

.. code:: ipython3

    %%time
    
    cpu_or_gpu = 'gpu' if instance_type.startswith('ml.p') else 'cpu'
    repository_short_name = "sagemaker-unity-ray-%s" % cpu_or_gpu
    docker_build_args = {
        'CPU_OR_GPU': cpu_or_gpu, 
        'AWS_REGION': boto3.Session().region_name,
    }
    custom_image_name = build_and_push_docker_image(repository_short_name, build_args=docker_build_args)
    print("Using ECR image %s" % custom_image_name)

Use Unity Example Environment
-----------------------------

The Unity ML-Agents Toolkit provides an expanding set of `example
environments <https://github.com/Unity-Technologies/ml-agents/blob/742c2fbf01188fbf27e82d5a7d9b5fd42f0de67a/docs/Learning-Environment-Examples.md>`__.
You can specify one of the environments name in the config and SageMaker
RL will start to train a RL agent against that environment. The training
code loads an example environment (``Basic`` by default) from the
`default
registry <https://github.com/Unity-Technologies/ml-agents/blob/742c2fbf01188fbf27e82d5a7d9b5fd42f0de67a/docs/Unity-Environment-Registry.md>`__
and start the training. Currently SageMaker RL can only support example
environment with a single agent.

Write the Training Code
~~~~~~~~~~~~~~~~~~~~~~~

The training code is written in the file “train-unity.py” which is
uploaded in the /src directory. First import the environment files and
the preset files, and then define the main() function.

.. code:: ipython3

    !pygmentize src/train-unity.py

Train the RL model using the Python SDK Script mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
5. Specify the hyperparameters for the RL agent algorithm.
6. Define the metrics definitions that you are interested in capturing
   in your logs. These can also be visualized in CloudWatch and
   SageMaker Notebooks.

.. code:: ipython3

    %%time
    
    metric_definitions = RLEstimator.default_metric_definitions(RLToolkit.RAY)
        
    estimator = RLEstimator(entry_point="train-unity.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            image_name=custom_image_name,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            metric_definitions=metric_definitions,
                            hyperparameters={
                              # Attention scientists!  You can override any Ray algorithm parameter here:
                              # "rl.training.config.env_config.env_name": "Basic",
                              # "rl.training.stop.timesteps_total": 10000,
                              # "rl.training.config.num_sgd_iter": 10,
                            }
                        )
    
    estimator.fit(wait=local_mode)
    job_name = estimator.latest_training_job.job_name
    print("Training job: %s" % job_name)

Plot metrics for training job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can see the reward metric of the training as it’s running, using
algorithm metrics that are recorded in CloudWatch metrics. We can plot
this to see the performance of the model over time.

.. code:: ipython3

    %matplotlib inline
    from sagemaker.analytics import TrainingJobAnalytics
    import time
    
    sm_client = boto3.client(service_name='sagemaker')
    state = sm_client.describe_training_job(
        TrainingJobName=job_name).get('SecondaryStatus')
    
    if not local_mode:
        while state == 'Starting':
            state = sm_client.describe_training_job(
                TrainingJobName=job_name).get('SecondaryStatus')
            print("Training job starting...")
            time.sleep(30)
        print("Training job started. Waiting for algorithm metric...")
        df = TrainingJobAnalytics(job_name, ['episode_reward_mean']).dataframe()
        num_metrics = len(df)
        if num_metrics == 0:
            print("No algorithm metrics found in CloudWatch, please check later.")
        else:
            plt = df.plot(x='timestamp', y='value', figsize=(12,5), legend=True, style='b-')
            plt.set_ylabel('Mean reward per episode')
            plt.set_xlabel('Training time (s)')
    else:
        print("Can't plot metrics in local mode.")

(Optional) Bring customized Unity environment
---------------------------------------------

We have shown how to load a sample environment from the Unity
Environment Registry and train a RL agent. However, you can use your
custom Unity executable as the environment to start the training. 1.
`Create a Unity
executable <https://github.com/Unity-Technologies/ml-agents/blob/742c2fbf01188fbf27e82d5a7d9b5fd42f0de67a/docs/Learning-Environment-Executable.md>`__
of your Unity environment for Linux platform. Please make sure your
environment only contains one single agent. 2. Upload the executable
file, dependency data files and library files to s3. 3. Specify the s3
path as a train channel of the training job.

Train the RL model using the Python SDK Script mode with provided Unity executables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    
    metric_definitions = RLEstimator.default_metric_definitions(RLToolkit.RAY)
        
    estimator = RLEstimator(entry_point="train-unity.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            image_name=custom_image_name,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            metric_definitions=metric_definitions,
                            hyperparameters={
                              # Attention scientists!  You can override any Ray algorithm parameter here:
                              # "rl.training.config.env_config.env_name": "<file name of your Unity executables> e.g. basic_env_linux.x86_64",
                              # "rl.training.stop.timesteps_total": 1000,
                              # "rl.training.config.num_sgd_iter": 10,
                            }
                        )
    s3_binary_path = '<s3 path for your Unity files> e.g. s3://bucket/unity-data'
    estimator.fit({'train': s3_binary_path}, wait=local_mode)
    job_name = estimator.latest_training_job.job_name
    print("Training job: %s" % job_name)

Evaluation of RL models
-----------------------

We use the last checkpointed model to run evaluation for the RL Agent.

Load checkpointed model
~~~~~~~~~~~~~~~~~~~~~~~

Checkpointed data from the previously trained models will be passed on
for evaluation / inference in the checkpoint channel. In local mode, we
can simply use the local directory, whereas in the SageMaker mode, it
needs to be moved to S3 first.

.. code:: ipython3

    tmp_dir = "/tmp/{}".format(job_name)
    os.system("mkdir {}".format(tmp_dir))
    print("Create local folder {}".format(tmp_dir))
    
    if local_mode:
        model_tar_key = "{}/model.tar.gz".format(job_name)
    else:
        model_tar_key = "{}/output/model.tar.gz".format(job_name)
        
    local_checkpoint_dir = "{}/model".format(tmp_dir)
    
    wait_for_s3_object(s3_bucket, model_tar_key, tmp_dir, training_job_name=job_name)  
    
    if not os.path.isfile("{}/model.tar.gz".format(tmp_dir)):
        raise FileNotFoundError("File model.tar.gz not found")
        
    os.system("mkdir -p {}".format(local_checkpoint_dir))
    os.system("tar -xvzf {}/model.tar.gz -C {}".format(tmp_dir, local_checkpoint_dir))
    
    print("Checkpoint directory {}".format(local_checkpoint_dir))

.. code:: ipython3

    if local_mode:
        checkpoint_path = 'file://{}'.format(local_checkpoint_dir)
        print("Local checkpoint file path: {}".format(local_checkpoint_dir))
    else:
        checkpoint_path = "s3://{}/{}/checkpoint/".format(s3_bucket, job_name)
        if not os.listdir(local_checkpoint_dir):
            raise FileNotFoundError("Checkpoint files not found under the path")
        os.system("aws s3 cp --recursive {} {}".format(local_checkpoint_dir, checkpoint_path))
        print("S3 checkpoint file path: {}".format(checkpoint_path))

.. code:: ipython3

    %%time
        
    estimator_eval = RLEstimator(entry_point="evaluate-unity.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            image_name=custom_image_name,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            base_job_name=job_name_prefix + "-evaluation",
                            hyperparameters={
                                "evaluate_episodes": 5,
                                "algorithm": "PPO",
                                # change the env name here acoording to the traine environment
                                "env": 'Basic'
                            }
                        )
    
    estimator_eval.fit({'model': checkpoint_path}, wait=local_mode) # specify a train channel for the custom Unity executable option
    job_name = estimator_eval.latest_training_job.job_name

Model deployment
================

Now let us deploy the RL policy so that we can get the optimal action,
given an environment observation.

.. code:: ipython3

    from sagemaker.tensorflow.serving import Model
    
    model = Model(model_data=estimator.model_data,
                  framework_version='2.1.0',
                  role=role)
    
    predictor = model.deploy(initial_instance_count=1, 
                             instance_type=instance_type)

.. code:: ipython3

    # ray 0.8.2 requires all the following inputs
    # 'prev_action', 'is_training', 'prev_reward' and 'seq_lens' are placeholders for this example
    # they won't affect prediction results
    
    # observation shpae passed here must match with the environment specs
    input = {"inputs": {'observations': np.ones(shape=(1, 20)).tolist(),
                        'prev_action': [0, 0],
                        'is_training': False,
                        'prev_reward': -1,
                        'seq_lens': -1
                       }
            }

.. code:: ipython3

    result = predictor.predict(input)
    
    result['outputs']['actions']

Clean up endpoint
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    predictor.delete_endpoint()
