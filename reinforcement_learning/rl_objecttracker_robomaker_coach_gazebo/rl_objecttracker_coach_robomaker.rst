Distributed Object Tracker RL training with Amazon SageMaker RL and RoboMaker
=============================================================================

How it works?
-------------

The reinforcement learning agent (i.e. Waffle) learns to track and
follow Burger by interacting with its environment, e.g., visual world
around it, by taking an action in a given state to maximize the expected
reward. The agent learns the optimal plan of actions in training by
trial-and-error through multiple episodes.

This notebook shows an example of distributed RL training across
SageMaker and two RoboMaker simulation envrionments that perform the
**rollouts** - execute a fixed number of episodes using the current
model or policy. The rollouts collect agent experiences
(state-transition tuples) and share this data with SageMaker for
training. SageMaker updates the model policy which is then used to
execute the next sequence of rollouts. This training loop continues
until the model converges, i.e. the car learns to drive and stops going
off-track. More formally, we can define the problem in terms of the
following:

1. **Objective**: Learn to drive toward and reach the Burger.
2. **Environment**: A simulator with Burger hosted on AWS RoboMaker.
3. **State**: The driving POV image captured by the Waffle’s head
   camera.
4. **Action**: Six discrete steering wheel positions at different angles
   (configurable)
5. **Reward**: Reward is inversely proportional to distance from Burger.
   Waffle gets more reward as it get closer to the Burger. It gets a
   reward of 0 if the action takes it away from Burger.

--------------

Prequisites
-----------

Imports
~~~~~~~

To get started, we’ll import the Python libraries we need, set up the
environment with a few prerequisites for permissions and configurations.

You can run this notebook from your local host or from a SageMaker
notebook instance. In both of these scenarios, you can run the following
to launch a training job on ``SageMaker`` and a simulation job on
``RoboMaker``.

.. code:: ipython3

    import sagemaker
    import boto3
    import sys
    import os
    import glob
    import re
    import subprocess
    from IPython.display import Markdown
    import time
    from time import gmtime, strftime
    sys.path.append("common")
    from misc import get_execution_role
    from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
    from markdown_helper import *

Setup S3 bucket
~~~~~~~~~~~~~~~

.. code:: ipython3

    # S3 bucket
    sage_session = sagemaker.session.Session()
    s3_bucket = sage_session.default_bucket()
    s3_output_path = 's3://{}/'.format(s3_bucket) # SDK appends the job name and output folder
    print("S3 bucket path: {}".format(s3_output_path))

Define Variables
~~~~~~~~~~~~~~~~

We define variables such as the job prefix for the training jobs and
s3_prefix for storing metadata required for synchronization between the
training and simulation jobs

.. code:: ipython3

    # create unique job name 
    job_name_prefix = 'rl-object-tracker'
    
    # create unique job name
    job_name = s3_prefix = job_name_prefix + "-sagemaker-" + strftime("%y%m%d-%H%M%S", gmtime())
    
    # Duration of job in seconds (5 hours)
    job_duration_in_seconds = 3600 * 5
    
    aws_region = sage_session.boto_region_name
    print("S3 bucket path: {}{}".format(s3_output_path, job_name))
    
    
    if aws_region not in ["us-west-2", "us-east-1", "eu-west-1"]:
        raise Exception("This notebook uses RoboMaker which is available only in US East (N. Virginia), US West (Oregon) and EU (Ireland). Please switch to one of these regions.")
    print("Model checkpoints and other metadata will be stored at: {}{}".format(s3_output_path, job_name))

Create an IAM role
~~~~~~~~~~~~~~~~~~

Either get the execution role when running from a SageMaker notebook
``role = sagemaker.get_execution_role()`` or, when running from local
machine, use utils method ``role = get_execution_role('role_name')`` to
create an execution role.

.. code:: ipython3

    try:
        role = sagemaker.get_execution_role()
    except:
        role = get_execution_role('sagemaker')
    
    print("Using IAM role arn: {}".format(role))

Permission setup for invoking AWS RoboMaker from this notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to enable this notebook to be able to execute AWS RoboMaker
jobs, we need to add one trust relationship to the default execution
role of this notebook.

.. code:: ipython3

    display(Markdown(generate_help_for_robomaker_trust_relationship(role)))

Configure VPC
-------------

| Since SageMaker and RoboMaker have to communicate with each other over
  the network, both of these services need to run in VPC mode. This can
  be done by supplying subnets and security groups to the job launching
  scripts.
| We will use the default VPC configuration for this example.

.. code:: ipython3

    ec2 = boto3.client('ec2')
    default_vpc = [vpc['VpcId'] for vpc in ec2.describe_vpcs()['Vpcs'] if vpc["IsDefault"] == True][0]
    
    default_security_groups = [group["GroupId"] for group in ec2.describe_security_groups()['SecurityGroups'] \
                       if group["GroupName"] == "default" and group["VpcId"] == default_vpc]
    
    default_subnets = [subnet["SubnetId"] for subnet in ec2.describe_subnets()["Subnets"] \
                      if subnet["VpcId"] == default_vpc and subnet['DefaultForAz']==True]
    
    print("Using default VPC:", default_vpc)
    print("Using default security group:", default_security_groups)
    print("Using default subnets:", default_subnets)

A SageMaker job running in VPC mode cannot access S3 resourcs. So, we
need to create a VPC S3 endpoint to allow S3 access from SageMaker
container. To learn more about the VPC mode, please visit `this
link. <https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html>`__

   The cell below should be executed to create the VPC S3 endpoint only
   if your are running this example for the first time. If the execution
   fails due to insufficient premissions or some other reasons, please
   create a VPC S3 endpoint manually by following
   `create-s3-endpoint.md <create-s3-endpoint.md>`__ (can be found in
   the same folder as this notebook).

.. code:: ipython3

    try:
        route_tables = [route_table["RouteTableId"] for route_table in ec2.describe_route_tables()['RouteTables']\
                    if route_table['VpcId'] == default_vpc]
    except Exception as e:
        if "UnauthorizedOperation" in str(e):
            display(Markdown(generate_help_for_s3_endpoint_permissions(role)))
        else:
            display(Markdown(create_s3_endpoint_manually(aws_region, default_vpc)))
        raise e
    
    print("Trying to attach S3 endpoints to the following route tables:", route_tables)
    
    assert len(route_tables) >= 1, "No route tables were found. Please follow the VPC S3 endpoint creation "\
                                  "guide by clicking the above link."
    
    try:
        ec2.create_vpc_endpoint(DryRun=False,
                               VpcEndpointType="Gateway",
                               VpcId=default_vpc,
                               ServiceName="com.amazonaws.{}.s3".format(aws_region),
                               RouteTableIds=route_tables)
        print("S3 endpoint created successfully!")
    except Exception as e:
        if "RouteAlreadyExists" in str(e):
            print("S3 endpoint already exists.")
        elif "UnauthorizedOperation" in str(e):
            display(Markdown(generate_help_for_s3_endpoint_permissions(role)))
            raise e
        else:
            display(Markdown(create_s3_endpoint_manually(aws_region, default_vpc)))
            raise e

Setup the environment
---------------------

The environment is defined in a Python file called
“object_tracker_env.py” and the file can be found at
``src/robomaker/environments/``. This file implements the gym interface
for our Gazebo based RoboMakersimulator. This is a common environment
file used by both SageMaker and RoboMaker. The environment variable -
``NODE_TYPE`` defines which node the code is running on. So, the
expressions that have ``rospy`` dependencies are executed on RoboMaker
only.

We can experiment with different reward functions by modifying
``reward_function`` in this file. Action space and steering angles can
be changed by modifying the step method in
``TurtleBot3ObjectTrackerAndFollowerDiscreteEnv`` class.

Configure the preset for RL algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| The parameters that configure the RL training job are defined in
  ``src/robomaker/presets/object_tracker.py``. Using the preset file,
  you can define agent parameters to select the specific agent
  algorithm. We suggest using Clipped PPO for this example.
| You can edit this file to modify algorithm parameters like
  learning_rate, neural network structure, batch_size, discount factor
  etc.

.. code:: ipython3

    !pygmentize src/robomaker/presets/object_tracker.py

Training Entrypoint
~~~~~~~~~~~~~~~~~~~

The training code is written in the file “training_worker.py” which is
uploaded in the /src directory. At a high level, it does the following:
- Uploads SageMaker node’s IP address. - Starts a Redis server which
receives agent experiences sent by rollout worker[s] (RoboMaker
simulator). - Trains the model everytime after a certain number of
episodes are received. - Uploads the new model weights on S3. The
rollout workers then update their model to execute the next set of
episodes.

.. code:: ipython3

    # Uncomment the line below to see the training code
    #!pygmentize src/training_worker.py

Train the model using Python SDK/ script mode
---------------------------------------------

.. code:: ipython3

    s3_location = "s3://%s/%s" % (s3_bucket, s3_prefix)
    !aws s3 rm --recursive {s3_location}
    
    
    # Make any changes to the envrironment and preset files below and upload these files if you want to use custom environment and preset
    !aws s3 cp src/robomaker/environments/ {s3_location}/environments/ --recursive --exclude ".ipynb_checkpoints*"
    !aws s3 cp src/robomaker/presets/ {s3_location}/presets/ --recursive --exclude ".ipynb_checkpoints*"

First, we define the following algorithm metrics that we want to capture
from cloudwatch logs to monitor the training progress. These are
algorithm specific parameters and might change for different algorithm.
We use `Clipped
PPO <https://coach.nervanasys.com/algorithms/policy_optimization/cppo/index.html>`__
for this example.

.. code:: ipython3

    metric_definitions = [
        # Training> Name=main_level/agent, Worker=0, Episode=19, Total reward=-102.88, Steps=19019, Training iteration=1
        {'Name': 'reward-training',
         'Regex': '^Training>.*Total reward=(.*?),'},
        
        # Policy training> Surrogate loss=-0.32664725184440613, KL divergence=7.255815035023261e-06, Entropy=2.83156156539917, training epoch=0, learning_rate=0.00025
        {'Name': 'ppo-surrogate-loss',
         'Regex': '^Policy training>.*Surrogate loss=(.*?),'},
         {'Name': 'ppo-entropy',
         'Regex': '^Policy training>.*Entropy=(.*?),'},
       
        # Testing> Name=main_level/agent, Worker=0, Episode=19, Total reward=1359.12, Steps=20015, Training iteration=2
        {'Name': 'reward-testing',
         'Regex': '^Testing>.*Total reward=(.*?),'},
    ]

We use the RLEstimator for training RL jobs.

1. Specify the source directory where the environment, presets and
   training code is uploaded.
2. Specify the entry point as the training code
3. Specify the choice of RL toolkit and framework. This automatically
   resolves to the ECR path for the RL Container.
4. Define the training parameters such as the instance count, instance
   type, job name, s3_bucket and s3_prefix for storing model checkpoints
   and metadata. **Only 1 training instance is supported for now.**
5. Set the RLCOACH_PRESET as “object_tracker” for this example.
6. Define the metrics definitions that you are interested in capturing
   in your logs. These can also be visualized in CloudWatch and
   SageMaker Notebooks.

.. code:: ipython3

    RLCOACH_PRESET = "object_tracker"
    
    instance_type = "ml.c5.4xlarge"
        
    estimator = RLEstimator(entry_point="training_worker.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            toolkit=RLToolkit.COACH,
                            toolkit_version='0.11',
                            framework=RLFramework.TENSORFLOW,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            train_max_run=job_duration_in_seconds,
                            hyperparameters={"s3_bucket": s3_bucket,
                                             "s3_prefix": s3_prefix,
                                             "aws_region": aws_region,
                                             "RLCOACH_PRESET": RLCOACH_PRESET,
                                          },
                            metric_definitions = metric_definitions,
                            subnets=default_subnets,
                            security_group_ids=default_security_groups,
                        )
    
    estimator.fit(job_name=job_name, wait=False)

Start the Robomaker job
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from botocore.exceptions import UnknownServiceError
    
    robomaker = boto3.client("robomaker")

Create Simulation Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first create a RoboMaker simulation application using the
``object-tracker public bundle``. Please refer to `RoboMaker Sample
Application Github
Repository <https://github.com/aws-robotics/aws-robomaker-sample-application-objecttracker>`__
if you want to learn more about this bundle or modify it.

.. code:: ipython3

    bundle_s3_key = 'object-tracker/simulation_ws.tar.gz'
    bundle_source = {'s3Bucket': s3_bucket,
                     's3Key': bundle_s3_key,
                     'architecture': "X86_64"}
    simulation_software_suite={'name': 'Gazebo',
                               'version': '7'}
    robot_software_suite={'name': 'ROS',
                          'version': 'Kinetic'}
    rendering_engine={'name': 'OGRE',
                      'version': '1.x'}

.. code:: ipython3

    simulation_application_bundle_location = "https://s3-us-west-2.amazonaws.com/robomaker-applications-us-west-2-11d8d0439f6a/object-tracker/object-tracker-1.0.80.0.1.0.130.0/simulation_ws.tar.gz"
    
    !wget {simulation_application_bundle_location}
    !aws s3 cp simulation_ws.tar.gz s3://{s3_bucket}/{bundle_s3_key}
    !rm simulation_ws.tar.gz

.. code:: ipython3

    app_name = "object-tracker-sample-application" + strftime("%y%m%d-%H%M%S", gmtime())
    
    try:
        response = robomaker.create_simulation_application(name=app_name,
                                                       sources=[bundle_source],
                                                       simulationSoftwareSuite=simulation_software_suite,
                                                       robotSoftwareSuite=robot_software_suite,
                                                       renderingEngine=rendering_engine
                                                      )
        simulation_app_arn = response["arn"]
        print("Created a new simulation app with ARN:", simulation_app_arn)
    except Exception as e:
        if "AccessDeniedException" in str(e):
            display(Markdown(generate_help_for_robomaker_all_permissions(role)))
            raise e
        else:
            raise e

Launch the Simulation job on RoboMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We create `AWS
RoboMaker <https://console.aws.amazon.com/robomaker/home#welcome>`__
Simulation Jobs that simulates the environment and shares this data with
SageMaker for training.

.. code:: ipython3

    num_simulation_workers = 1
    
    envriron_vars = {
                     "MODEL_S3_BUCKET": s3_bucket,
                     "MODEL_S3_PREFIX": s3_prefix,
                     "ROS_AWS_REGION": aws_region,
                     "MARKOV_PRESET_FILE": "object_tracker.py",
                     "NUMBER_OF_ROLLOUT_WORKERS": str(num_simulation_workers)}
    
    simulation_application = {"application":simulation_app_arn,
                              "launchConfig": {"packageName": "object_tracker_simulation",
                                               "launchFile": "distributed_training.launch",
                                               "environmentVariables": envriron_vars}
                             }
                                
    vpcConfig = {"subnets": default_subnets,
                 "securityGroups": default_security_groups,
                 "assignPublicIp": True}
    
    responses = []
    for job_no in range(num_simulation_workers):
        response =  robomaker.create_simulation_job(iamRole=role,
                                                clientRequestToken=strftime("%Y-%m-%d-%H-%M-%S", gmtime()),
                                                maxJobDurationInSeconds=job_duration_in_seconds,
                                                failureBehavior="Continue",
                                                simulationApplications=[simulation_application],
                                                vpcConfig=vpcConfig
                                                )
        responses.append(response)
    
    print("Created the following jobs:")
    job_arns = [response["arn"] for response in responses]
    for job_arn in job_arns:
        print("Job ARN", job_arn) 

Visualizing the simulations in RoboMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can visit the RoboMaker console to visualize the simulations or run
the following cell to generate the hyperlinks.

.. code:: ipython3

    display(Markdown(generate_robomaker_links(job_arns, aws_region)))

Clean Up
~~~~~~~~

Execute the cells below if you want to kill RoboMaker and SageMaker job.
It also removes RoboMaker resources created during the run.

.. code:: ipython3

    for job_arn in job_arns:
        robomaker.cancel_simulation_job(job=job_arn)

.. code:: ipython3

    sage_session.sagemaker_client.stop_training_job(TrainingJobName=estimator._current_job_name)

Evaluation
~~~~~~~~~~

.. code:: ipython3

    envriron_vars = {"MODEL_S3_BUCKET": s3_bucket,
                     "MODEL_S3_PREFIX": s3_prefix,
                     "ROS_AWS_REGION": aws_region,
                     "NUMBER_OF_TRIALS": str(20),
                     "MARKOV_PRESET_FILE": "%s.py" % RLCOACH_PRESET
                     }
    
    simulation_application = {"application":simulation_app_arn,
                              "launchConfig": {"packageName": "object_tracker_simulation",
                                               "launchFile": "evaluation.launch",
                                               "environmentVariables": envriron_vars}
                             }
                                
    vpcConfig = {"subnets": default_subnets,
                 "securityGroups": default_security_groups,
                 "assignPublicIp": True}
    
    
    
    response =  robomaker.create_simulation_job(iamRole=role,
                                            clientRequestToken=strftime("%Y-%m-%d-%H-%M-%S", gmtime()),
                                            maxJobDurationInSeconds=job_duration_in_seconds,
                                            failureBehavior="Continue",
                                            simulationApplications=[simulation_application],
                                            vpcConfig=vpcConfig
                                            )
    print("Created the following job:")
    print("Job ARN", response["arn"])

Clean Up Simulation Application Resource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    robomaker.delete_simulation_application(application=simulation_app_arn)
