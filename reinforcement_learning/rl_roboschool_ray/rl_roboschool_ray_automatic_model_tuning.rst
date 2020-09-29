Tune hyperparameters for your RL training job
---------------------------------------------

This notebook shows how to use SageMaker’s Automatic Model Tuning
functionality to optimize the training of an RL model, using the
Roboschool environment. Note that the bayesian hyperparameter
optimization algorithm used in SageMaker Automatic Model Tuning expects
a stable objective function, which means that running the same training
job multiple times with the same configuration should give about the
same result. Some RL training processes are highly non-deterministic,
such that the same configuration sometimes performs well and sometimes
fails to train. These environments will not work well with the automatic
model tuner. However, the Roboschool environments are fairly stable and
tend to perform similarly given the same hyperparameters. So the
hyperparameter tuner should work well here.

Pick which Roboschool problem to solve
--------------------------------------

Roboschool is an `open
source <https://github.com/openai/roboschool/tree/master/roboschool>`__
physics simulator that is commonly used to train RL policies for robotic
systems. Roboschool defines a
`variety <https://github.com/openai/roboschool/blob/master/roboschool/__init__.py>`__
of Gym environments that correspond to different robotics problems. Here
we’re highlighting a few of them at varying levels of difficulty:

-  **Reacher (easy)** - a very simple robot with just 2 joints reaches
   for a target
-  **Hopper (medium)** - a simple robot with one leg and a foot learns
   to hop down a track
-  **Humanoid (difficult)** - a complex 3D robot with two arms, two
   legs, etc. learns to balance without falling over and then to run on
   a track

The simpler problems train faster with less computational resources. The
more complex problems are more fun.

.. code:: ipython3

    # Uncomment the problem to work on
    #roboschool_problem = 'reacher'
    roboschool_problem = 'hopper'
    #roboschool_problem = 'humanoid'

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
    job_name_prefix = 'tune-'+roboschool_problem

Configure resources to be used for tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tuning jobs need to happen in hosted SageMaker, not local mode. So we
pick the instance type. Also we need to specify how many training jobs
and how quickly to run them.

.. code:: ipython3

    # Note that Tuning cannot happen in local mode.
    instance_type = "ml.m4.xlarge"
    
    # Pick the total number of training jobs to run in this tuning job
    max_jobs = 50
    
    # How many jobs should run at a time.  Higher numbers here mean the tuning job runs much faster,
    # while lower numbers can sometimes get better results
    max_parallel_jobs = 5

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
    repository_short_name = "sagemaker-roboschool-ray-%s" % cpu_or_gpu
    docker_build_args = {
        'CPU_OR_GPU': cpu_or_gpu, 
        'AWS_REGION': boto3.Session().region_name,
    }
    custom_image_name = build_and_push_docker_image(repository_short_name, build_args=docker_build_args)
    print("Using ECR image %s" % custom_image_name)

Configure Tuning
----------------

Tuning jobs need to know what to vary when looking for a great
configuration. Sometimes this is called the search space, or the
configuration space. We do this by picking a set of hyperparameters to
vary, and specify the ranges for each of them. Unless you’re experienced
at automatically tuning hyperparameters, you should probably start with
just one or two hyperparameters at a time to see how they effect the
result.

.. code:: ipython3

    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
    
    # The hyperparameters we're going to tune
    hyperparameter_ranges = {
        # inspired by https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        #'rl.training.config.clip_param': ContinuousParameter(0.1, 0.4),
        #'rl.training.config.kl_target': ContinuousParameter(0.003, 0.03),
        #'rl.training.config.vf_loss_coeff': ContinuousParameter(0.5, 1.0),
        #'rl.training.config.entropy_coeff': ContinuousParameter(0.0, 0.01),
        'rl.training.config.kl_coeff': ContinuousParameter(0.5, 1.0),
        'rl.training.config.num_sgd_iter': IntegerParameter(3, 50),
    }
    
    # The hyperparameters that are the same for all jobs
    static_hyperparameters = {
        "rl.training.stop.time_total_s": 600,  # Tell each training job to stop after 10 minutes
        #'rl.training.config.num_sgd_iter': 7,
        #'rl.training.config.sgd_minibatch_size': 1000,
        #'rl.training.config.train_batch_size': 25000,
    }

Prepare to launch the tuning job.
---------------------------------

First we create an estimator like we would if we were launching a single
training job. This will be used to create the ``tuner`` object.

.. code:: ipython3

    metric_definitions = RLEstimator.default_metric_definitions(RLToolkit.RAY)
    estimator = RLEstimator(entry_point="train-%s.py" % roboschool_problem,
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            image_name=custom_image_name,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            metric_definitions=metric_definitions,
                            hyperparameters=static_hyperparameters,
                        )
    
    tuner = HyperparameterTuner(estimator,
                                objective_metric_name='episode_reward_mean',
                                objective_type='Maximize',
                                hyperparameter_ranges=hyperparameter_ranges,
                                metric_definitions=metric_definitions,
                                max_jobs=max_jobs,
                                max_parallel_jobs=max_parallel_jobs,
                                base_tuning_job_name=job_name_prefix,
                               )
    tuner.fit()

Monitor progress
----------------

To see how your tuning job is doing, jump over to the SageMaker console.
Under the **Training** section, you’ll see Hyperparameter tuning jobs,
where you’ll see the newly created job. It will launch a series of
TrainingJobs in your account, each of which will behave like a regular
training job. They will show up in the list, and when each job is
completed, you’ll see the final value they achieved for mean reward.
Each job will also emit algorithm metrics to cloudwatch, which you can
see plotted in CloudWatch metrics. To see these, click on the training
job to det to its detail page, and then look for the link “View
algorithm metrics” which will let you see a chart of how that job is
progressing. By changing the search criteria in the CloudWatch console,
you can overlay the metrics for all the jobs in this tuning job.

