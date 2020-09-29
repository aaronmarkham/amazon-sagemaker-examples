Solving Knapsack Problem with Amazon SageMaker RL
=================================================

Knapsack is a canonical operations research problem. We start with a bag
and a set of items. We choose which items to put in the bag. Our
objective is to maximize the value of the items in the bag; but we
cannot put all the items in as the bag capacity is limited. The problem
is hard because the items have different values and weights, and there
are many combinations to consider. In the classic version of the
problem, we pick the items in one shot. But in this baseline, we instead
consider the items one at a time over a fixed time horizon.

Problem Statement
-----------------

We start with an empty bag and an item. We need to either put the item
in the bag or throw it away. If we put it in the bag, we get a reward
equal to the value of the item. If we throw the item away, we get a
fixed penalty. In case the bag is too full to accommodate the item, we
are forced to throw it away. In the next step, another item appears and
we need to decide again if we want to put it in the bag or throw it
away. This process repeats for a fixed number of steps. Since we do not
know the value and weight of items that will come in the future, and the
bag can only hold so many items, it is not obvious what is the right
thing to do.

At each time step, our agent is aware of the following information: -
Weight capacity of the bag - Volume capacity of the bag - Sum of item
weight in the bag - Sum of item volume in the bag - Sum of item value in
the bag - Current item weight - Current item volume - Current item value
- Time remaining

At each time step, our agent can take one of the following actions: -
Put the item in the bag - Throw the item away

At each time step, our agent gets the following reward depending on
their action: - Item value if you put it in the bag and bag does not
overflow - A penalty if you throw the item away or if the item does not
fit in the bag

The time horizon is 20 steps. You can see the specifics in the
``KnapSackMediumEnv`` class in ``knapsack_env.py``. There are a couple
of other classes that provide an easier (``KnapSackEnv``) and a more
difficult version (``KnapSackHardEnv``) of this problem.

Using Amazon SageMaker RL
-------------------------

Amazon SageMaker RL allows you to train your RL agents in cloud machines
using docker containers. You do not have to worry about setting up your
machines with the RL toolkits and deep learning frameworks. You can
easily switch between many different machines setup for you, including
powerful GPU machines that give a big speedup. You can also choose to
use multiple machines in a cluster to further speedup training, often
necessary for production level loads.

Pre-requsites
~~~~~~~~~~~~~

Imports
^^^^^^^

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

Settings
^^^^^^^^

You can run this notebook from your local host or from a SageMaker
notebook instance. In both of these scenarios, you can run the following
in either ``local`` or ``SageMaker`` modes. The ``local`` mode uses the
SageMaker Python SDK to run your code in a local container before
deploying to SageMaker. This can speed up iterative testing and
debugging while using the same familiar Python SDK interface. You just
need to set ``local_mode = True``.

.. code:: ipython3

    # run in local mode?
    local_mode = False
    
    # create unique job name 
    job_name_prefix = 'rl-knapsack'
    
    # S3 bucket
    sage_session = sagemaker.session.Session()
    s3_bucket = sage_session.default_bucket()  
    print("Using s3 bucket %s" % s3_bucket)  # create this bucket if it doesn't exist
    s3_output_path = 's3://{}/'.format(s3_bucket) # SDK appends the job name and output folder

Install docker for ``local`` mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to work in ``local`` mode, you need to have docker installed.
When running from you local instance, please make sure that you have
docker or docker-compose (for local CPU machines) and nvidia-docker (for
local GPU machines) installed. Alternatively, when running from a
SageMaker notebook instance, you can simply run the following script

Note, you can only run a single local notebook at one time.

.. code:: ipython3

    if local_mode:
        !/bin/bash ./common/setup.sh

Create an IAM role
^^^^^^^^^^^^^^^^^^

Either get the execution role when running from a SageMaker notebook
``role = sagemaker.get_execution_role()`` or, when running locally, set
it to an IAM role with ``AmazonSageMakerFullAccess`` and
``CloudWatchFullAccess permissions``.

.. code:: ipython3

    try:
        role = sagemaker.get_execution_role()
    except:
        role = get_execution_role()
    
    print("Using IAM role arn: {}".format(role))

Setup the environment
^^^^^^^^^^^^^^^^^^^^^

The environment is defined in a Python file called ``knapsack_env.py``
in the ``./src`` directory. It implements the init(), step(), reset()
and render() functions that describe how the environment behaves. This
is consistent with Open AI Gym interfaces for defining an environment.

-  Init() - initialize the environment in a pre-defined state
-  Step() - take an action on the environment
-  reset()- restart the environment on a new episode
-  render() - get a rendered image of the environment in its current
   state

Configure the presets for RL algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The presets that configure the RL training jobs are defined in the
``preset-knapsack-clippedppo.py`` in the ``./src`` directory. Using the
preset file, you can define agent parameters to select the specific
agent algorithm. You can also set the environment parameters, define the
schedule and visualization parameters, and define the graph manager. The
schedule presets will define the number of heat up steps, periodic
evaluation steps, training steps between evaluations.

These can be overridden at runtime by specifying the RLCOACH_PRESET
hyperparameter. Additionally, it can be used to define custom
hyperparameters.

.. code:: ipython3

    !pygmentize src/preset-knapsack-clippedppo.py

Write the Training Code
^^^^^^^^^^^^^^^^^^^^^^^

The training code is in the file ``train-coach.py`` which is also the
``./src`` directory.

.. code:: ipython3

    !pygmentize src/train-coach.py

Train the model using Python SDK/ script mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using local mode, the training will run on the notebook
instance. When using SageMaker for training, you can select a GPU or CPU
instance. The RLEstimator is used for training RL jobs.

-  Specify the source directory where the environment, presets and
   training code is uploaded.
-  Specify the entry point as the training code
-  Specify the choice of RL toolkit and framework. This automatically
   resolves to the ECR path for the RL Container.
-  Define the training parameters such as the instance count, job name,
   S3 path for output and job name.
-  Specify the hyperparameters for the RL agent algorithm. The
   RLCOACH_PRESET can be used to specify the RL agent algorithm you want
   to use.
-  Define the metrics definitions that you are interested in capturing
   in your logs. These can also be visualized in CloudWatch and
   SageMaker Notebooks.

.. code:: ipython3

    if local_mode:
        instance_type = 'local'
    else:
        instance_type = "ml.m4.4xlarge"
        
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
                              "RLCOACH_PRESET":"preset-knapsack-clippedppo",
                              "rl.agent_params.algorithm.discount": 0.9,
                              "rl.evaluation_steps:EnvironmentEpisodes": 8,
                            }
                        )
    
    estimator.fit(wait=local_mode)

Store intermediate training output and model checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output from the training job above is stored on S3. The intermediate
folder contains gifs and metadata of the training

.. code:: ipython3

    job_name=estimator._current_job_name
    print("Job name: {}".format(job_name))
    
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

Visualization
~~~~~~~~~~~~~

Plot metrics for training job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can pull the reward metric of the training and plot it to see the
performance of the model over time.

.. code:: ipython3

    %matplotlib inline
    import pandas as pd
    
    csv_file_name = "worker_0.simple_rl_graph.main_level.main_level.agent_0.csv"
    key = intermediate_folder_key + "/" + csv_file_name
    wait_for_s3_object(s3_bucket, key, tmp_dir)
    
    csv_file = "{}/{}".format(tmp_dir, csv_file_name)
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['Training Reward'])
    x_axis = 'Episode #'
    y_axis = 'Training Reward'
    
    plt = df.plot(x=x_axis,y=y_axis, figsize=(12,5), legend=True, style='b-')
    plt.set_ylabel(y_axis);
    plt.set_xlabel(x_axis);

Visualize the rendered gifs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The latest gif file found in the gifs directory is displayed. You can
replace the tmp.gif file below to visualize other files generated.

.. code:: ipython3

    key = intermediate_folder_key + '/gifs'
    wait_for_s3_object(s3_bucket, key, tmp_dir)    
    print("Copied gifs files to {}".format(tmp_dir))
    
    glob_pattern = os.path.join("{}/*.gif".format(tmp_dir))
    gifs = [file for file in glob.iglob(glob_pattern, recursive=True)]
    extract_episode = lambda string: int(re.search('.*episode-(\d*)_.*', string, re.IGNORECASE).group(1))
    gifs.sort(key=extract_episode)
    print("GIFs found:\n{}".format("\n".join([os.path.basename(gif) for gif in gifs])))    
    
    # visualize a specific episode
    gif_index = -1 # since we want last gif
    gif_filepath = gifs[gif_index]
    gif_filename = os.path.basename(gif_filepath)
    print("Selected GIF: {}".format(gif_filename))
    os.system("mkdir -p ./src/tmp_render/ && cp {} ./src/tmp_render/{}.gif".format(gif_filepath, gif_filename))
    HTML('<img src="./src/tmp_render/{}.gif">'.format(gif_filename))

Evaluation of RL models
~~~~~~~~~~~~~~~~~~~~~~~

We use the last checkpointed model to run evaluation for the RL Agent.

Load checkpointed model
^^^^^^^^^^^^^^^^^^^^^^^

Checkpointed data from the previously trained models will be passed on
for evaluation / inference in the checkpoint channel. In local mode, we
can simply use the local directory, whereas in the SageMaker mode, it
needs to be moved to S3 first.

.. code:: ipython3

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
^^^^^^^^^^^^^^^^^^^^^^^

Use the checkpointed model to run the evaluation step.

.. code:: ipython3

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
                                     "RLCOACH_PRESET":"preset-knapsack-clippedppo",
                                     "evaluate_steps": 250, #5 episodes
                                 }
                        )
    estimator_eval.fit({'checkpoint': checkpoint_path})

Visualize the output
~~~~~~~~~~~~~~~~~~~~

Optionally, you can run the steps defined earlier to visualize the
output
