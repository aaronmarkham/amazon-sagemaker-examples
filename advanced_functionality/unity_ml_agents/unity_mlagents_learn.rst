Running mlagents-learn on SageMaker
===================================

Introduction
------------

`The Unity Machine Learning Agents Toolkit
(ML-Agents) <https://github.com/Unity-Technologies/ml-agents>`__ is an
open-source project that enables games and simulations to serve as
environments for training intelligent agents. This notebook contains
instructions for setting up SageMaker for using a command-line utility
mlagents-learn.

Pre-requisites
--------------

Build an Environment Executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need a built environment to use mlagents-learn on SageMaker. Please
follow the instruction
`here <https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Executable.md>`__
to create a built environment for Linux, and upload the executable file
and the dependency folder to s3.

.. code:: ipython3

    data_location = "<s3 prefix of your executable file and the dependency folder>"

Imports
~~~~~~~

We’ll begin with some necessary imports, and get an Amazon SageMaker
session to help perform certain tasks, as well as an IAM role with the
necessary permissions.

.. code:: ipython3

    import sagemaker
    import boto3
    import re
    import os
    import numpy as np
    import pandas as pd
    from docker_utils import build_and_push_docker_image
    
    role = sagemaker.get_execution_role()

Setup S3 bucket
~~~~~~~~~~~~~~~

Set up the linkage and authentication to the S3 bucket that you want to
use for checkpoint and the metadata.

.. code:: ipython3

    sage_session = sagemaker.session.Session()
    local_session = sagemaker.local.LocalSession()
    s3_bucket = sage_session.default_bucket()  
    s3_output_path = 's3://{}/'.format(s3_bucket)
    print("S3 bucket path: {}".format(s3_output_path))

Configure where training happens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can train your training jobs using the SageMaker notebook instance
or local notebook instance. In both of these scenarios, you can run the
following in either local or SageMaker modes. The local mode uses the
SageMaker Python SDK to run your code in a local container before
deploying to SageMaker. This can speed up iterative testing and
debugging while using the same familiar Python SDK interface. You just
need to set local_mode = True.

.. code:: ipython3

    # run in local_mode on this machine, or as a SageMaker TrainingJob?
    local_mode = True
    
    if local_mode:
        instance_type = 'local'
        sess = local_session
        !/bin/bash ./setup.sh
    else:
        # If on SageMaker, pick the instance type
        instance_type = "ml.c5.2xlarge"
        sess = sage_session

Create an estimator and fit the model
-------------------------------------

We also need a trainer config file which defines the training
hyperparameters for each Behavior in the scene, and the set-ups for the
environment parameters. Please refer `this
doc <https://github.com/Unity-Technologies/ml-agents/blob/2d0eb6147c031c082522eb683e569dd99b4d65fb/docs/Training-ML-Agents.md#training-configurations>`__
for detailed info. As a sample, we are providing src/train.yaml which is
`the config file to train the 3D Balance
Ball <https://github.com/Unity-Technologies/ml-agents/blob/eedc3f9c052295d89bed0ac40a8e82a8fd17fead/config/ppo/3DBall.yaml>`__
in the `Getting Started
guide <https://github.com/Unity-Technologies/ml-agents/blob/2d0eb6147c031c082522eb683e569dd99b4d65fb/docs/Getting-Started.md>`__.
Because the config file is version specific. Please replace the file if
you want to use another version. You can change the version of ml-agents
specifying the version on src/requirements.txt. Tensorflow version can
be specified changing the framework_version of the TensorFlow estimator.

Note: set env_name as your game execute file e.g. 3DBall.x86_64

.. code:: ipython3

    from sagemaker.tensorflow import TensorFlow
    
    metric_definitions = [
        {'Name': 'train:mean reward', 'Regex': 'Mean Reward: ([0-9]*.[0-9]*)'},
        {'Name': 'train:std of reward', 'Regex': 'Std of Reward: ([0-9]*.[0-9]*)'},
        {'Name': 'train:step', 'Regex': 'Step: ([0-9]*.)'}
    ]
    
    estimator = TensorFlow(entry_point='train.py',
                           source_dir='src',
                           train_instance_type=instance_type,
                           train_instance_count=1,
                           hyperparameters={'env_name': '<your env name>','yaml_file':'train.yaml'},
                           role=role,
                           framework_version='1.15.2',
                           py_version='py37',
                           metric_definitions=metric_definitions,
                           script_mode=True)
    
    estimator.fit({"train":data_location})

Plot metrics for training job
-----------------------------

.. code:: ipython3

    %matplotlib inline
    from sagemaker.analytics import TrainingJobAnalytics
    
    job_name = estimator.latest_training_job.job_name
    
    if not local_mode:
        try:
            df = TrainingJobAnalytics(job_name, ['train:mean reward']).dataframe()
        except KeyError:
            print("Training job '{}' is not ready, please check later.".format(job_name))
        num_metrics = len(df)
        if num_metrics == 0:
            print("No algorithm metrics found in CloudWatch")
        else:
            plt = df.plot(x='timestamp', y='value', figsize=(12,5), legend=True, style='b-')
            plt.set_ylabel('Mean reward')
            plt.set_xlabel('Training time (s)')
    else:
        print("Can't plot metrics in local mode.")
