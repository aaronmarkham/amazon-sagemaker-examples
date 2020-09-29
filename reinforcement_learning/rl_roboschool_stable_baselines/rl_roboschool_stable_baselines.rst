Roboschool simulations training with stable baselines on Amazon SageMaker RL
============================================================================

Introductions
-------------

Roboschool is an `open
source <https://github.com/openai/roboschool/tree/master/roboschool>`__
physics simulator that is commonly used to train RL policies for robotic
systems. Roboschool defines a
`variety <https://github.com/openai/roboschool/blob/master/roboschool/__init__.py>`__
of Gym environments that correspond to different robotics problems. One
of them is **HalfCheetah** which is a two-legged robot, restricted to a
vertical plane, meaning it can only run forward or backward.

In this notebook example, we will make **HalfCheetah** learn to walk
using the
`stable-baselines <https://stable-baselines.readthedocs.io/en/master/>`__
a set of improved implementations of Reinforcement Learning (RL)
algorithms based on `OpenAI
Baselines <https://github.com/openai/baselines>`__.

.. code:: ipython3

    roboschool_problem = 'half-cheetah'

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
    import subprocess
    from IPython.display import HTML
    import time
    from time import gmtime, strftime
    sys.path.append("common")
    from misc import get_execution_role, wait_for_s3_object
    from docker_utils import build_and_push_docker_image
    from sagemaker.rl import RLEstimator

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
    job_name_prefix = 'rl-roboschool-'+roboschool_problem

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
        instance_type = "ml.c4.xlarge"
        

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
3. Installing stable-baselines and its dependencies such as OpenMPI,
   etc.
4. Uploading the new container image to ECR

This step can take a long time if you are running on a machine with a
slow internet connection. If your notebook instance is in SageMaker or
EC2 it should take 3-10 minutes depending on the instance type.

.. code:: ipython3

    %%time
    
    cpu_or_gpu = 'gpu' if instance_type.startswith('ml.p') else 'cpu'
    repository_short_name = "sagemaker-roboschool-stablebaselines-%s" % cpu_or_gpu
    docker_build_args = { 
        'AWS_REGION': boto3.Session().region_name,
    }
    custom_image_name = build_and_push_docker_image(repository_short_name, build_args=docker_build_args)
    print("Using ECR image %s" % custom_image_name)

Write the Training Code
-----------------------

Configure the presets for RL algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The presets that configure the RL training jobs are defined in the
``preset-half-cheetah.py`` in the ``./src`` directory. Using the preset
file, you can define agent parameters to select the specific agent
algorithm. You can also set the environment parameters, define the
schedule and visualization parameters, and define the graph manager. The
schedule presets will define following hyper-parameters for PPO1
training: \* ``num_timesteps``: (int) Number of training steps - Preset:
1e4 \* ``timesteps_per_actorbatch`` – (int) timesteps per actor per
update - Preset: 2048 \* ``clip_param`` – (float) clipping parameter
epsilon - Preset: 0.2 \* ``entcoeff`` – (float) the entropy loss weight
- Preset: 0.0 \* ``optim_epochs`` – (float) the optimizer’s number of
epochs - Preset: 10 \* ``optim_stepsize`` – (float) the optimizer’s
stepsize - Preset: 3e-4 \* ``optim_batchsize`` – (int) the optimizer’s
the batch size - Preset: 64 \* ``gamma`` – (float) discount factor -
Preset: 0.99 \* ``lam`` – (float) advantage estimation - Preset: 0.95 \*
``schedule`` – (str) The type of scheduler for the learning rate update
(‘linear’, ‘constant’, ‘double_linear_con’, ‘middle_drop’ or
‘double_middle_drop’) - Preset: linear \* ``verbose`` – (int) the
verbosity level: 0 none, 1 training information, 2 tensorflow debug -
Preset: 1

You can refer the complete list of args and documentation for PPO1
algorithm here:
https://stable-baselines.readthedocs.io/en/master/modules/ppo1.html

These can be overridden at runtime by specifying the
RLSTABLEBASELINES_PRESET hyperparameter. Additionally, it can be used to
define custom hyperparameters.

.. code:: ipython3

    !pygmentize src/preset-{roboschool_problem}.py

Write the Training Code
^^^^^^^^^^^^^^^^^^^^^^^

The training code is in the file ``train-coach.py`` which is also the
``./src`` directory.

.. code:: ipython3

    !pygmentize src/train_stable_baselines.py

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
   ``RLSTABLEBASELINES_PRESET`` can be used to specify the RL agent
   algorithm you want to use.
6. Define the metrics definitions that you are interested in capturing
   in your logs. These can also be visualized in CloudWatch and
   SageMaker Notebooks.

Please note all the configured preset parameters in
``preset-half-cheetah.py`` can be overriden by specifying the overriden
value in ``hyperparameters`` block.

**Note**: For MPI based jobs, local mode is only supported for single
instance jobs. Please use ``instance_type`` as ``1`` if using local
mode.

.. code:: ipython3

    %%time
    
    estimator = RLEstimator(entry_point="train_stable_baselines.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            image_name=custom_image_name,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=2,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            hyperparameters={
                                "RLSTABLEBASELINES_PRESET":"preset-{}.py".format(roboschool_problem),
                                "num_timesteps":1e4,
                                "instance_type":instance_type
                            },
                            metric_definitions= [
                                {
                                    "Name":"EpisodesLengthMean",
                                    "Regex":"\[.*,.*\]\<stdout\>\:\| *EpLenMean *\| *([-+]?[0-9]*\.?[0-9]*) *\|"
                                },
                                {
                                    "Name":"EpisodesRewardMean",
                                    "Regex":"\[.*,.*\]\<stdout\>\:\| *EpRewMean *\| *([-+]?[0-9]*\.?[0-9]*) *\|"
                                },
                                {
                                    "Name":"EpisodesSoFar",
                                    "Regex":"\[.*,.*\]\<stdout\>\:\| *EpisodesSoFar *\| *([-+]?[0-9]*\.?[0-9]*) *\|"
                                }
                            ]
                        )
    
    estimator.fit(wait=True)

Visualization
-------------

RL training can take a long time. So while it’s running there are a
variety of ways we can track progress of the running training job. Some
intermediate output gets saved to S3 during training, so we’ll set up to
capture that.

Fetch videos of training rollouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Videos of certain rollouts get written to S3 during training. Here we
fetch all that are available, and render the last one.

.. code:: ipython3

    
    job_name = estimator.latest_training_job.job_name
    print("Training job: %s" % job_name)
    
    s3_url = "s3://{}/{}".format(s3_bucket,job_name)
    
    if local_mode:
        output_tar_key = "{}/output.tar.gz".format(job_name)
    else:
        output_tar_key = "{}/output/output.tar.gz".format(job_name)
    
    intermediate_folder_key = "{}/output/intermediate".format(job_name)
    output_url = "s3://{}/{}".format(s3_bucket, output_tar_key)
    intermediate_url = "s3://{}/{}".format(s3_bucket, intermediate_folder_key)
    
    print("S3 job path: {}".format(s3_url))
    print("Output.tar.gz location: {}".format(output_url))
    print("Intermediate folder path: {}".format(intermediate_url))
        
    tmp_dir = "/tmp/{}".format(job_name)
    os.system("mkdir {}".format(tmp_dir))
    print("Create local folder {}".format(tmp_dir))
    wait_for_s3_object(s3_bucket, intermediate_folder_key, tmp_dir) 

RL output video
~~~~~~~~~~~~~~~

.. code:: ipython3

    import io
    import base64
    video = io.open("{}/rl_out.mp4".format(tmp_dir), 'r+b').read()
    encoded = base64.b64encode(video)
    HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))

Example of trained walking HalfCheetah
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the output of the training job triggered bu above code, with
following additional configurations: \* ``train_instance_count``: 10 \*
``train_instance_type``: ml.c4.xlarge \* ``num_timesteps``: 1e7

It took 40 min to train the model with the above settings. You can have
similar output with lesser instances and more training duration.

.. code:: ipython3

    import io
    import base64
    video = io.open("examples/robo_half_cheetah_10x_40min.mp4", 'r+b').read()
    encoded = base64.b64encode(video)
    HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))
