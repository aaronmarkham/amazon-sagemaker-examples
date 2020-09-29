Game servers autopilot
======================

Multiplayer game publishers often need to either over-provision
resources or manually manage compute resource allocation when launching
a large-scale worldwide game, to avoid the long player-wait in the game
lobby. Game publishers need to develop, config, and deploy tools that
helped them to monitor and control the compute allocation.

This notebook demonstrates Game server autopilot, a new machine
learning-based example tool that makes it easy for game publishers to
reduce the time players wait for compute to spawn, while still avoiding
compute over-provisioning. It also eliminates manual configuration
decisions and changes publishers need to make and reduces the
opportunity for human errors.

We heard from customers that optimizing compute resource allocation is
not trivial. This is because it often takes substantial time to allocate
and prepare EC2 instances. The time needed to spin up an EC2 instance
and install game binaries and other assets must be learned and accounted
for in the allocation algorithm. Ever-changing usage patterns require a
model that is adaptive to emerging player habits. Finally, the system
also performs scale down in concert with new server allocation as
needed.

We describe a reinforcement learning-based system that learns to
allocate resources in response to player usage patterns. The hosted
model directly predicts the required number of game-servers so as to
allow EKS the time to allocate instances to reduce player wait time. The
training process integrates with the game eco-system, and requires
minimal manual configuration.

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


.. parsed-literal::

    S3 bucket path: s3://sagemaker-us-west-2-356566070122/


Parameters
~~~~~~~~~~

Adding new parameters for the job require update in the training section
that invokes the RLEstimator.

.. code:: ipython3

    job_name_prefix = 'rl-game-server-autopilot'
    job_duration_in_seconds = 60 * 60 * 24 * 5
    train_instance_count = 1
    cloudwatch_namespace = 'rl-game-server-autopilot'
    min_servers=10
    max_servers=100
    # over provisionning factor. use 5 for optimal. 
    over_prov_factor=5
    #gamma is the discount factor
    gamma=0.9
    # if local inference is set gs_inventory_url=local and populate learning_freq
    gs_inventory_url = 'https://4bfiebw6ui.execute-api.us-west-2.amazonaws.com/api/currsine1h/'
    #gs_inventory_url = 'local'
    # sleep time in seconds between step() executions
    learning_freq = 65
    # actions are normelized between 0 and 1, action factor the number of game servers needed e.g. 100 will be 100*action and clipped to the min and max servers parameters above
    action_factor = 100

.. code:: ipython3

    
    # Pick the instance type
    instance_type = "ml.c5.xlarge" #4 cpus
    #     instance_type = "ml.c5.4xlarge" #16 cpus
    #      instance_type = "ml.c5.2xlarge" #8 cpus
    #      instance_type = "ml.c4.4xlarge"
    #     instance_type = "ml.p2.8xlarge" #32 cpus
    #     instance_type = "ml.p3.2xlarge" #8 cpus
    #    instance_type = "ml.p3.8xlarge" #32 cpus
    #     instance_type = "ml.p3.16xlarge" #96 cpus
    #     instance_type = "ml.c5.18xlarge" #72 cpus
    
    num_cpus_per_instance = 4

Create an IAM role
~~~~~~~~~~~~~~~~~~

Either get the execution role when running from a SageMaker notebook
instance ``role = sagemaker.get_execution_role()`` or, when running from
local notebook instance, use utils method
``role = get_execution_role()`` to create an execution role. In this
example, the env thru the training job, publishes cloudwatch custom
metrics as well as put values in DynamoDB table. Therefore, an
appropriate role is required to be set to the role arn below.

.. code:: ipython3

    try:
        role = sagemaker.get_execution_role()
    except:
        role = get_execution_role()
    
    print("Using IAM role arn: {}".format(role))


.. parsed-literal::

    Using IAM role arn: arn:aws:iam::356566070122:role/service-role/AmazonSageMaker-ExecutionRole-20181024T210472


Set up the environment
======================

The environment is defined in a Python file called gameserver_env.py and
the file is uploaded on /src directory. The environment also implements
the init(), step() and reset() functions that describe how the
environment behaves. This is consistent with Open AI Gym interfaces for
defining an environment. It also implements help functions for custom
CloudWatch metrics (populate_cloudwatch_metric()) and a simple sine
demand simulator (get_curr_sine1h())

1. init() - initialize the environment in a pre-defined state
2. step() - take an action on the environment
3. reset()- restart the environment on a new episode
4. get_curr_sine1h() - return the sine value based on the current
   second.
5. populate_cloudwatch_metric(namespace,metric_value,metric_name) -
   populate the metric_name with metric_value in namespace.

.. code:: ipython3

    !pygmentize src/gameserver_env.py


.. parsed-literal::

    [34mimport[39;49;00m [04m[36mtime[39;49;00m
    [34mimport[39;49;00m [04m[36mboto3[39;49;00m
    [34mimport[39;49;00m [04m[36mrequests[39;49;00m
    [34mimport[39;49;00m [04m[36mgym[39;49;00m
    [34mimport[39;49;00m [04m[36mnumpy[39;49;00m [34mas[39;49;00m [04m[36mnp[39;49;00m
    [34mfrom[39;49;00m [04m[36mtime[39;49;00m [34mimport[39;49;00m gmtime,strftime
    [34mfrom[39;49;00m [04m[36mgym.spaces[39;49;00m [34mimport[39;49;00m Discrete, Box
    
    cloudwatch_cli = boto3.client([33m'[39;49;00m[33mcloudwatch[39;49;00m[33m'[39;49;00m,region_name=[33m'[39;49;00m[33mus-west-2[39;49;00m[33m'[39;49;00m)
     
    [34mclass[39;49;00m [04m[32mGameServerEnv[39;49;00m(gym.Env):
    
        [34mdef[39;49;00m [32m__init__[39;49;00m([36mself[39;49;00m, env_config={}):
            [34mprint[39;49;00m ([33m"[39;49;00m[33min __init__[39;49;00m[33m"[39;49;00m)
            [34mprint[39;49;00m ([33m"[39;49;00m[33menv_config {}[39;49;00m[33m"[39;49;00m.format(env_config))
            [36mself[39;49;00m.namespace = env_config[[33m'[39;49;00m[33mcloudwatch_namespace[39;49;00m[33m'[39;49;00m]
            [36mself[39;49;00m.gs_inventory_url = env_config[[33m'[39;49;00m[33mgs_inventory_url[39;49;00m[33m'[39;49;00m]
            [36mself[39;49;00m.learning_freq = env_config[[33m'[39;49;00m[33mlearning_freq[39;49;00m[33m'[39;49;00m]
            [36mself[39;49;00m.min_servers = [36mint[39;49;00m(env_config[[33m'[39;49;00m[33mmin_servers[39;49;00m[33m'[39;49;00m])
            [36mself[39;49;00m.max_servers = [36mint[39;49;00m(env_config[[33m'[39;49;00m[33mmax_servers[39;49;00m[33m'[39;49;00m])
            [36mself[39;49;00m.action_factor = [36mint[39;49;00m(env_config[[33m'[39;49;00m[33maction_factor[39;49;00m[33m'[39;49;00m])
            [36mself[39;49;00m.over_prov_factor = [36mint[39;49;00m(env_config[[33m'[39;49;00m[33mover_prov_factor[39;49;00m[33m'[39;49;00m])
            [36mself[39;49;00m.num_steps = [34m0[39;49;00m
            [36mself[39;49;00m.max_num_steps = [34m301[39;49;00m
            [36mself[39;49;00m.history_len = [34m5[39;49;00m
            [36mself[39;49;00m.total_num_of_obs = [34m1[39;49;00m
            [37m# we have two observation array, allocation and demand. allocation is alloc_observation, demand is observation hence *2[39;49;00m
            [36mself[39;49;00m.observation_space = Box(low=np.array([[36mself[39;49;00m.min_servers]*[36mself[39;49;00m.history_len*[34m2[39;49;00m),
                                               high=np.array([[36mself[39;49;00m.max_servers]*[36mself[39;49;00m.history_len*[34m2[39;49;00m),
                                               dtype=np.uint32)
            
            [37m# How many servers should the agent spin up at each time step [39;49;00m
            [36mself[39;49;00m.action_space = Box(low=np.array([[34m0[39;49;00m]),
                                         high=np.array([[34m1[39;49;00m]),
                                         dtype=np.float32)
    
        [34mdef[39;49;00m [32mreset[39;49;00m([36mself[39;49;00m):
            [34mprint[39;49;00m ([33m"[39;49;00m[33min reset[39;49;00m[33m"[39;49;00m)
            [37m#self.populate_cloudwatch_metric(self.namespace,1,'reset')[39;49;00m
            [36mself[39;49;00m.num_steps = [34m0[39;49;00m
            [36mself[39;49;00m.current_min = [34m0[39;49;00m
            [36mself[39;49;00m.demand_observation = np.array([[36mself[39;49;00m.min_servers]*[36mself[39;49;00m.history_len)
            [36mself[39;49;00m.alloc_observation = np.array([[36mself[39;49;00m.min_servers]*[36mself[39;49;00m.history_len)
            [37m#self.action_observation = np.array([self.min_servers]*self.history_len)[39;49;00m
            
            [34mprint[39;49;00m ([33m'[39;49;00m[33mself.demand_observation [39;49;00m[33m'[39;49;00m+[36mstr[39;49;00m([36mself[39;49;00m.demand_observation))
            [34mprint[39;49;00m ([33m'[39;49;00m[33mself.alloc_observation [39;49;00m[33m'[39;49;00m+[36mstr[39;49;00m([36mself[39;49;00m.alloc_observation))
            [37m#return np.concatenate((self.demand_observation, self.alloc_observation,self.action_observation))[39;49;00m
            [34mreturn[39;49;00m np.concatenate(([36mself[39;49;00m.demand_observation, [36mself[39;49;00m.alloc_observation))
    
       
    
        [34mdef[39;49;00m [32mstep[39;49;00m([36mself[39;49;00m, action):
            [34mprint[39;49;00m ([33m'[39;49;00m[33min step - action recieved from model[39;49;00m[33m'[39;49;00m+[36mstr[39;49;00m(action))
            [36mself[39;49;00m.num_steps+=[34m1[39;49;00m
            [36mself[39;49;00m.total_num_of_obs+=[34m1[39;49;00m
            [34mprint[39;49;00m([33m'[39;49;00m[33mtotal_num_of_obs={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.total_num_of_obs))
    
            raw_action=[36mfloat[39;49;00m(action)
            [36mself[39;49;00m.curr_action = raw_action*[36mself[39;49;00m.action_factor
            [36mself[39;49;00m.curr_action = np.clip([36mself[39;49;00m.curr_action, [36mself[39;49;00m.min_servers, [36mself[39;49;00m.max_servers)
            [34mprint[39;49;00m([33m'[39;49;00m[33mself.curr_action={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.curr_action))
            
                   
            [34mif[39;49;00m ([36mself[39;49;00m.gs_inventory_url!=[33m'[39;49;00m[33mlocal[39;49;00m[33m'[39;49;00m):
              [37m#get the demand from the matchmaking service[39;49;00m
              [34mprint[39;49;00m([33m'[39;49;00m[33mquering matchmaking service for current demand, curr_demand[39;49;00m[33m'[39;49;00m)
              [34mtry[39;49;00m:
               gs_url=[36mself[39;49;00m.gs_inventory_url
               req=requests.get(url=gs_url)
               data=req.json()
               [36mself[39;49;00m.curr_demand = [36mfloat[39;49;00m(data[[33m'[39;49;00m[33mPrediction[39;49;00m[33m'[39;49;00m][[33m'[39;49;00m[33mnum_of_gameservers[39;49;00m[33m'[39;49;00m])            
                
              [34mexcept[39;49;00m requests.exceptions.RequestException [34mas[39;49;00m e:
               [34mprint[39;49;00m(e)
               [34mprint[39;49;00m([33m'[39;49;00m[33mif matchmaking did not respond just randomized curr_demand between limit, reward will correct[39;49;00m[33m'[39;49;00m)
               [36mself[39;49;00m.curr_demand = [36mfloat[39;49;00m(np.random.randint([36mself[39;49;00m.min_servers,[36mself[39;49;00m.max_servers))
            [34mif[39;49;00m ([36mself[39;49;00m.gs_inventory_url==[33m'[39;49;00m[33mlocal[39;49;00m[33m'[39;49;00m):
              [34mprint[39;49;00m([33m'[39;49;00m[33mlocal matchmaking service for current demand, curr_demand[39;49;00m[33m'[39;49;00m)
              data=[36mself[39;49;00m.get_curr_sine1h()
              [36mself[39;49;00m.curr_demand = [36mfloat[39;49;00m(data[[33m'[39;49;00m[33mPrediction[39;49;00m[33m'[39;49;00m][[33m'[39;49;00m[33mnum_of_gameservers[39;49;00m[33m'[39;49;00m])       
            [37m# clip the demand to the allowed range[39;49;00m
            [36mself[39;49;00m.curr_demand = np.clip([36mself[39;49;00m.curr_demand, [36mself[39;49;00m.min_servers, [36mself[39;49;00m.max_servers)
            [34mprint[39;49;00m([33m'[39;49;00m[33mself.curr_demand={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.curr_demand)) 
    
            [36mself[39;49;00m.curr_alloc = [36mself[39;49;00m.alloc_observation[[34m0[39;49;00m]
            [34mprint[39;49;00m([33m'[39;49;00m[33mself.curr_alloc={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.curr_alloc)) 
                
            [37m# Assumes it takes history_len time steps to create or delete [39;49;00m
            [37m# the game server from allocation[39;49;00m
            [37m# self.action_observation = self.action_observation[1:][39;49;00m
            [37m# self.action_observation = np.append(self.action_observation, self.curr_action)[39;49;00m
            [37m# print('self.action_observation={}'.format(self.action_observation))[39;49;00m
            
            [37m# store the current demand in the history array demand_observation[39;49;00m
            [36mself[39;49;00m.demand_observation = [36mself[39;49;00m.demand_observation[[34m1[39;49;00m:] [37m# shift the observation by one to remove one history point[39;49;00m
            [36mself[39;49;00m.demand_observation=np.append([36mself[39;49;00m.demand_observation,[36mself[39;49;00m.curr_demand)
            [34mprint[39;49;00m([33m'[39;49;00m[33mself.demand_observation={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.demand_observation))
            
            [37m# store the current demand in the history array demand_observation[39;49;00m
            [36mself[39;49;00m.alloc_observation = [36mself[39;49;00m.alloc_observation[[34m1[39;49;00m:] 
            [36mself[39;49;00m.alloc_observation=np.append([36mself[39;49;00m.alloc_observation,[36mself[39;49;00m.curr_action)
            [34mprint[39;49;00m([33m'[39;49;00m[33mself.alloc_observation={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.alloc_observation))
     
            
            [37m#reward calculation - in case of over provision just 1-ratio. under provision is more severe so 500% more negative reward[39;49;00m
            [34mprint[39;49;00m([33m'[39;49;00m[33mcalculate the reward, calculate the ratio between allocation and demand, we use the first allocation in the series of history of five, first_alloc/curr_demand[39;49;00m[33m'[39;49;00m)
            [34mprint[39;49;00m([33m'[39;49;00m[33mhistory of previous predictions made by the model ={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.alloc_observation))
            
            ratio=[36mself[39;49;00m.curr_alloc/[36mself[39;49;00m.curr_demand
            [34mprint[39;49;00m([33m'[39;49;00m[33mratio={}[39;49;00m[33m'[39;49;00m.format(ratio))
            [34mif[39;49;00m (ratio>[34m1[39;49;00m):
               [37m#reward=1-ratio[39;49;00m
               reward = -[34m1[39;49;00m * ([36mself[39;49;00m.curr_alloc - [36mself[39;49;00m.curr_demand)
               [34mprint[39;49;00m([33m'[39;49;00m[33mover provision - ratio>1 - {}[39;49;00m[33m'[39;49;00m.format(reward))
            [34mif[39;49;00m (ratio<[34m1[39;49;00m):
               [37m#reward=-50*ratio[39;49;00m
               reward = -[34m5[39;49;00m * ([36mself[39;49;00m.curr_demand - [36mself[39;49;00m.curr_alloc) 
               [34mprint[39;49;00m([33m'[39;49;00m[33munder provision - ratio<1 - {}[39;49;00m[33m'[39;49;00m.format(reward))
            [34mif[39;49;00m (ratio==[34m1[39;49;00m):
               reward=[34m1[39;49;00m
               [34mprint[39;49;00m([33m'[39;49;00m[33mratio=1[39;49;00m[33m'[39;49;00m)
            reward -= ([36mself[39;49;00m.curr_demand - [36mself[39;49;00m.curr_alloc)*[36mself[39;49;00m.over_prov_factor
            [34mprint[39;49;00m([33m'[39;49;00m[33mratio={}[39;49;00m[33m'[39;49;00m.format(ratio))
            [34mprint[39;49;00m([33m'[39;49;00m[33mreward={}[39;49;00m[33m'[39;49;00m.format(reward))
                    
             
            [37m#Instrumnet the supply and demand in cloudwatch[39;49;00m
            [34mprint[39;49;00m([33m'[39;49;00m[33mpopulating cloudwatch - self.curr_demand={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.curr_demand))
            [36mself[39;49;00m.populate_cloudwatch_metric([36mself[39;49;00m.namespace,[36mself[39;49;00m.curr_demand,[33m'[39;49;00m[33mcurr_demand[39;49;00m[33m'[39;49;00m)
            [34mprint[39;49;00m([33m'[39;49;00m[33mpopulating cloudwatch - self.curr_alloc={}[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.curr_action))
            [36mself[39;49;00m.populate_cloudwatch_metric([36mself[39;49;00m.namespace,[36mself[39;49;00m.curr_action,[33m'[39;49;00m[33mcurr_alloc[39;49;00m[33m'[39;49;00m)
            [34mprint[39;49;00m([33m'[39;49;00m[33mpopulating cloudwatch - reward={}[39;49;00m[33m'[39;49;00m.format(reward))
            [36mself[39;49;00m.populate_cloudwatch_metric([36mself[39;49;00m.namespace,reward,[33m'[39;49;00m[33mreward[39;49;00m[33m'[39;49;00m)
            
            [34mif[39;49;00m ([36mself[39;49;00m.num_steps >= [36mself[39;49;00m.max_num_steps):
              done = [36mTrue[39;49;00m
              [34mprint[39;49;00m ([33m"[39;49;00m[33mself.num_steps [39;49;00m[33m"[39;49;00m+[36mstr[39;49;00m([36mself[39;49;00m.num_steps))
              [34mprint[39;49;00m ([33m"[39;49;00m[33mself.max_num_steps [39;49;00m[33m"[39;49;00m+[36mstr[39;49;00m([36mself[39;49;00m.max_num_steps))
            [34melse[39;49;00m:
              done = [36mFalse[39;49;00m
            
            [34mprint[39;49;00m ([33m'[39;49;00m[33mtime.sleep() for {} before next iteration[39;49;00m[33m'[39;49;00m.format([36mself[39;49;00m.learning_freq))
            time.sleep([36mint[39;49;00m([36mself[39;49;00m.learning_freq)) 
            
            extra_info = {}
            [37m#the next state includes the demand and allocation history. [39;49;00m
            [37m#next_state=np.concatenate((self.demand_observation,self.alloc_observation,self.action_observation))[39;49;00m
            next_state=np.concatenate(([36mself[39;49;00m.demand_observation,[36mself[39;49;00m.alloc_observation))
            [34mprint[39;49;00m ([33m'[39;49;00m[33mnext_state={}[39;49;00m[33m'[39;49;00m.format(next_state))
            [34mreturn[39;49;00m next_state, reward, done, extra_info
    
        [34mdef[39;49;00m [32mrender[39;49;00m([36mself[39;49;00m, mode):
            [34mprint[39;49;00m([33m"[39;49;00m[33min render[39;49;00m[33m"[39;49;00m)
            [34mpass[39;49;00m
    
        [34mdef[39;49;00m [32mpopulate_cloudwatch_metric[39;49;00m([36mself[39;49;00m,namespace,metric_value,metric_name):
            [34mprint[39;49;00m([33m"[39;49;00m[33min populate_cloudwatch_metric metric_value=[39;49;00m[33m"[39;49;00m+[36mstr[39;49;00m(metric_value)+[33m"[39;49;00m[33m metric_name=[39;49;00m[33m"[39;49;00m+metric_name)
            response = cloudwatch_cli.put_metric_data(
        	Namespace=namespace,
        	MetricData=[
               {
                  [33m'[39;49;00m[33mMetricName[39;49;00m[33m'[39;49;00m: metric_name,
                  [33m'[39;49;00m[33mUnit[39;49;00m[33m'[39;49;00m: [33m'[39;49;00m[33mNone[39;49;00m[33m'[39;49;00m,
                  [33m'[39;49;00m[33mValue[39;49;00m[33m'[39;49;00m: metric_value,
               },
            ]
            )
            [34mprint[39;49;00m([33m'[39;49;00m[33mresponse from cloud watch[39;49;00m[33m'[39;49;00m+[36mstr[39;49;00m(response))
            
        [34mdef[39;49;00m [32mget_curr_sine1h[39;49;00m([36mself[39;49;00m):
            max_servers=[36mself[39;49;00m.max_servers*[34m0.9[39;49;00m
            [34mprint[39;49;00m ([33m'[39;49;00m[33min get_curr_sine1h[39;49;00m[33m'[39;49;00m)
            cycle_arr=np.linspace([34m0.2[39;49;00m,[34m3.1[39;49;00m,[34m61[39;49;00m)
            [36mself[39;49;00m.current_min = ([36mself[39;49;00m.current_min + [34m1[39;49;00m) % [34m60[39;49;00m
            current_min = [36mself[39;49;00m.current_min
            [34mprint[39;49;00m([33m'[39;49;00m[33mcurrent_min={}[39;49;00m[33m'[39;49;00m.format(current_min))
            current_point=cycle_arr[[36mint[39;49;00m(current_min)]
            sine=max_servers*np.sin(current_point)
            [34mprint[39;49;00m([33m'[39;49;00m[33msine({})={}[39;49;00m[33m'[39;49;00m.format(current_point,sine))
            [34mreturn[39;49;00m {[33m"[39;49;00m[33mPrediction[39;49;00m[33m"[39;49;00m:{[33m"[39;49;00m[33mnum_of_gameservers[39;49;00m[33m"[39;49;00m: sine}}


Configure the presets for RL algorithm
--------------------------------------

The presets that configure the RL training jobs are defined in the
train_gameserver_ppo.py file which is also uploaded on the /src
directory. Using the preset file, you can define agent parameters to
select the specific agent algorithm. You can also set the environment
parameters, define the schedule and visualization parameters, and define
the graph manager. The schedule presets will define the number of heat
up steps, periodic evaluation steps, training steps between evaluations.
It can be used to define custom hyperparameters.

.. code:: ipython3

    !pygmentize src/train_gameserver_ppo.py


.. parsed-literal::

    [34mimport[39;49;00m [04m[36mjson[39;49;00m
    [34mimport[39;49;00m [04m[36mos[39;49;00m
    [34mimport[39;49;00m [04m[36msys[39;49;00m
    [34mimport[39;49;00m [04m[36mgym[39;49;00m
    [34mimport[39;49;00m [04m[36mray[39;49;00m
    [34mfrom[39;49;00m [04m[36mray.tune[39;49;00m [34mimport[39;49;00m run_experiments
    [34mfrom[39;49;00m [04m[36mray.tune.registry[39;49;00m [34mimport[39;49;00m register_env
    
    [34mfrom[39;49;00m [04m[36msagemaker_rl.ray_launcher[39;49;00m [34mimport[39;49;00m SageMakerRayLauncher
    
    env_config={}
    
    [34mclass[39;49;00m [04m[32mMyLauncher[39;49;00m(SageMakerRayLauncher):
    
        [34mdef[39;49;00m [32mregister_env_creator[39;49;00m([36mself[39;49;00m):
            [34mfrom[39;49;00m [04m[36mgameserver_env[39;49;00m [34mimport[39;49;00m GameServerEnv
            register_env([33m"[39;49;00m[33mGameServers[39;49;00m[33m"[39;49;00m, [34mlambda[39;49;00m env_config: GameServerEnv(env_config))
            
        [34mdef[39;49;00m [32m_save_tf_model[39;49;00m([36mself[39;49;00m):
            [34mprint[39;49;00m([33m"[39;49;00m[33min _save_tf_model[39;49;00m[33m"[39;49;00m)
            ckpt_dir = [33m'[39;49;00m[33m/opt/ml/output/data/checkpoint[39;49;00m[33m'[39;49;00m
            model_dir = [33m'[39;49;00m[33m/opt/ml/model[39;49;00m[33m'[39;49;00m
    
            [37m# Re-Initialize from the checkpoint so that you will have the latest models up.[39;49;00m
            tf.train.init_from_checkpoint(ckpt_dir,
                                          {[33m'[39;49;00m[33mmain_level/agent/online/network_0/[39;49;00m[33m'[39;49;00m: [33m'[39;49;00m[33mmain_level/agent/online/network_0[39;49;00m[33m'[39;49;00m})
            tf.train.init_from_checkpoint(ckpt_dir,
                                          {[33m'[39;49;00m[33mmain_level/agent/online/network_1/[39;49;00m[33m'[39;49;00m: [33m'[39;49;00m[33mmain_level/agent/online/network_1[39;49;00m[33m'[39;49;00m})
    
            [37m# Create a new session with a new tf graph.[39;49;00m
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=[36mTrue[39;49;00m))
            sess.run(tf.global_variables_initializer())  [37m# initialize the checkpoint.[39;49;00m
    
            [37m# This is the node that will accept the input.[39;49;00m
            input_nodes = tf.get_default_graph().get_tensor_by_name([33m'[39;49;00m[33mmain_level/agent/main/online/[39;49;00m[33m'[39;49;00m + \
                                                                    [33m'[39;49;00m[33mnetwork_0/observation/observation:0[39;49;00m[33m'[39;49;00m)
            [37m# This is the node that will produce the output.[39;49;00m
            output_nodes = tf.get_default_graph().get_operation_by_name([33m'[39;49;00m[33mmain_level/agent/main/online/[39;49;00m[33m'[39;49;00m + \
                                                                        [33m'[39;49;00m[33mnetwork_1/ppo_head_0/policy_mean/BiasAdd[39;49;00m[33m'[39;49;00m)
            [37m# Save the model as a servable model.[39;49;00m
            tf.saved_model.simple_save(session=sess,
                                       export_dir=[33m'[39;49;00m[33mmodel[39;49;00m[33m'[39;49;00m,
                                       inputs={[33m"[39;49;00m[33mobservation[39;49;00m[33m"[39;49;00m: input_nodes},
                                       outputs={[33m"[39;49;00m[33mpolicy[39;49;00m[33m"[39;49;00m: output_nodes.outputs[[34m0[39;49;00m]})
            [37m# Move to the appropriate folder. [39;49;00m
            shutil.move([33m'[39;49;00m[33mmodel/[39;49;00m[33m'[39;49;00m, model_dir + [33m'[39;49;00m[33m/model/tf-model/00000001/[39;49;00m[33m'[39;49;00m)
            [37m# SageMaker will pick it up and upload to the right path.[39;49;00m
            [34mprint[39;49;00m([33m"[39;49;00m[33min _save_tf_model Success[39;49;00m[33m"[39;49;00m)
    
        [34mdef[39;49;00m [32mget_experiment_config[39;49;00m([36mself[39;49;00m):
            [34mprint[39;49;00m([33m'[39;49;00m[33mget_experiment_config[39;49;00m[33m'[39;49;00m)       
            [34mprint[39;49;00m(env_config)
            [37m# allowing 1600 seconds to the job toto stop and save the model[39;49;00m
            time_total_s=[36mfloat[39;49;00m(env_config[[33m"[39;49;00m[33mtime_total_s[39;49;00m[33m"[39;49;00m])-[34m4600[39;49;00m
            [34mprint[39;49;00m([33m"[39;49;00m[33mtime_total_s=[39;49;00m[33m"[39;49;00m+[36mstr[39;49;00m(time_total_s))
            [34mreturn[39;49;00m {
              [33m"[39;49;00m[33mtraining[39;49;00m[33m"[39;49;00m: {
                [33m"[39;49;00m[33menv[39;49;00m[33m"[39;49;00m: [33m"[39;49;00m[33mGameServers[39;49;00m[33m"[39;49;00m,
                [33m"[39;49;00m[33mrun[39;49;00m[33m"[39;49;00m: [33m"[39;49;00m[33mPPO[39;49;00m[33m"[39;49;00m,
                 [33m"[39;49;00m[33mstop[39;49;00m[33m"[39;49;00m: {
                   [33m"[39;49;00m[33mtime_total_s[39;49;00m[33m"[39;49;00m: time_total_s
                 },
                [33m"[39;49;00m[33mconfig[39;49;00m[33m"[39;49;00m: {
                   [33m"[39;49;00m[33mignore_worker_failures[39;49;00m[33m"[39;49;00m: [36mTrue[39;49;00m,
                   [33m"[39;49;00m[33mgamma[39;49;00m[33m"[39;49;00m: [34m0[39;49;00m,
                   [33m"[39;49;00m[33mkl_coeff[39;49;00m[33m"[39;49;00m: [34m1.0[39;49;00m,
                   [33m"[39;49;00m[33mnum_sgd_iter[39;49;00m[33m"[39;49;00m: [34m10[39;49;00m,
                   [33m"[39;49;00m[33mlr[39;49;00m[33m"[39;49;00m: [34m0.0001[39;49;00m,
                   [33m"[39;49;00m[33msgd_minibatch_size[39;49;00m[33m"[39;49;00m: [34m32[39;49;00m, 
                   [33m"[39;49;00m[33mtrain_batch_size[39;49;00m[33m"[39;49;00m: [34m128[39;49;00m,
                   [33m"[39;49;00m[33mmodel[39;49;00m[33m"[39;49;00m: {
    [37m#                 "free_log_std": True,[39;49;00m
    [37m#                  "fcnet_hiddens": [512, 512],[39;49;00m
                    },
                   [33m"[39;49;00m[33muse_gae[39;49;00m[33m"[39;49;00m: [36mTrue[39;49;00m,
                   [37m#"num_workers": (self.num_cpus-1),[39;49;00m
                   [33m"[39;49;00m[33mnum_gpus[39;49;00m[33m"[39;49;00m: [36mself[39;49;00m.num_gpus,
                   [37m#"batch_mode": "complete_episodes",[39;49;00m
                   [33m"[39;49;00m[33mnum_workers[39;49;00m[33m"[39;49;00m: [34m1[39;49;00m,
                    [33m"[39;49;00m[33menv_config[39;49;00m[33m"[39;49;00m: env_config,
                   [37m#'observation_filter': 'MeanStdFilter',[39;49;00m
                }
              }
            }
    
    [34mif[39;49;00m [31m__name__[39;49;00m == [33m"[39;49;00m[33m__main__[39;49;00m[33m"[39;49;00m:
        [34mfor[39;49;00m i [35min[39;49;00m [36mrange[39;49;00m([36mlen[39;49;00m(sys.argv)):
          [34mif[39;49;00m i==[34m0[39;49;00m:
             [34mcontinue[39;49;00m
          [34mif[39;49;00m i % [34m2[39;49;00m > [34m0[39;49;00m:
             env_config[sys.argv[i].split([33m'[39;49;00m[33m--[39;49;00m[33m'[39;49;00m,[34m1[39;49;00m)[[34m1[39;49;00m]]=sys.argv[i+[34m1[39;49;00m]
        MyLauncher().train_main()


Train the RL model using the Python SDK Script mode
---------------------------------------------------

The RLEstimator is used for training RL jobs.

1. The entry_point value indicates the script that invokes the
   GameServer RL environment.
2. source_dir indicates the location of environment code which currently
   includes train-gameserver-ppo.py and game_server_env.py.
3. Specify the choice of RL toolkit and framework. This automatically
   resolves to the ECR path for the RL Container.
4. Define the training parameters such as the instance count, job name,
   S3 path for output and job name.
5. Specify the hyperparameters for the RL agent algorithm. The
   RLCOACH_PRESET or the RLRAY_PRESET can be used to specify the RL
   agent algorithm you want to use.
6. Define the metrics definitions that you are interested in capturing
   in your logs. These can also be visualized in CloudWatch and
   SageMaker Notebooks.

.. code:: ipython3

    metric_definitions = [{'Name': 'episode_reward_mean',
      'Regex': 'episode_reward_mean: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'episode_reward_max',
      'Regex': 'episode_reward_max: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'episode_len_mean',
      'Regex': 'episode_len_mean: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'entropy',
      'Regex': 'entropy: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'episode_reward_min',
      'Regex': 'episode_reward_min: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'vf_loss',
      'Regex': 'vf_loss: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'policy_loss',
      'Regex': 'policy_loss: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},                                            
    ]
    
    metric_definitions




.. parsed-literal::

    [{'Name': 'episode_reward_mean',
      'Regex': 'episode_reward_mean: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'episode_reward_max',
      'Regex': 'episode_reward_max: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'episode_len_mean',
      'Regex': 'episode_len_mean: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'entropy',
      'Regex': 'entropy: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'episode_reward_min',
      'Regex': 'episode_reward_min: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'vf_loss',
      'Regex': 'vf_loss: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'},
     {'Name': 'policy_loss',
      'Regex': 'policy_loss: ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'}]



.. code:: ipython3

    %%time
    #metric_definitions = RLEstimator.default_metric_definitions(RLToolkit.RAY)
        
    estimator = RLEstimator(
                            entry_point="train_gameserver_ppo.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            toolkit=RLToolkit.RAY,
                            toolkit_version='0.6.5',
                            framework=RLFramework.TENSORFLOW,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=train_instance_count,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            metric_definitions=metric_definitions,
                            train_max_run=job_duration_in_seconds,
                            hyperparameters={
                               "cloudwatch_namespace":cloudwatch_namespace,
                              "gs_inventory_url":gs_inventory_url,
                              "learning_freq":learning_freq,
                              "time_total_s":job_duration_in_seconds,
                              "min_servers":min_servers,
                              "max_servers":max_servers,
                              "gamma":gamma,
                              "action_factor":action_factor,
                              "over_prov_factor":over_prov_factor,
                              "save_model": 1
                            }
                        )
    
    estimator.fit(wait=False)
    job_name = estimator.latest_training_job.job_name
    print("Training job: %s" % job_name)


.. parsed-literal::

    Training job: rl-game-server-autopilot-2019-12-25-06-03-34-742
    CPU times: user 118 ms, sys: 0 ns, total: 118 ms
    Wall time: 315 ms


.. code:: ipython3

    import sagemaker
    sagemaker.__version__




.. parsed-literal::

    '1.45.0.dev0'



Store intermediate training output and model checkpoints
--------------------------------------------------------

The output from the training job above is stored in a S3.

.. code:: ipython3

    %%time
    
    job_name=estimator._current_job_name
    print("Job name: {}".format(job_name))
    
    s3_url = "s3://{}/{}".format(s3_bucket,job_name)
    
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

Evaluation of RL models
-----------------------

We use the latest checkpointed model to run evaluation for the RL Agent.

Load checkpointed model
~~~~~~~~~~~~~~~~~~~~~~~

Checkpointed data from the previously trained models will be passed on
for evaluation / inference in the checkpoint channel. Since TensorFlow
stores ckeckpoint file containes absolute paths from when they were
generated (see issue), we need to replace the absolute paths to relative
paths. This is implemented within evaluate-game-server.py

.. code:: ipython3

    %%time
    
    wait_for_s3_object(s3_bucket, output_tar_key, tmp_dir)  
    
    if not os.path.isfile("{}/output.tar.gz".format(tmp_dir)):
        raise FileNotFoundError("File output.tar.gz not found")
    os.system("tar -xvzf {}/output.tar.gz -C {}".format(tmp_dir, tmp_dir))
    
    checkpoint_dir = "{}/checkpoint".format(tmp_dir)
    
    print("Checkpoint directory {}".format(checkpoint_dir))

.. code:: ipython3

    %%time
    checkpoint_path = "s3://{}/{}/checkpoint/".format(s3_bucket, job_name)
    if not os.listdir(checkpoint_dir):
         raise FileNotFoundError("Checkpoint files not found under the path")
    os.system("aws s3 cp --recursive {} {}".format(checkpoint_dir, checkpoint_path))
    print("S3 checkpoint file path: {}".format(checkpoint_path))

Run the evaluation step
-----------------------

Use the checkpointed model to run the evaluation step.

.. code:: ipython3

    %%time
    job_name = "5obs-local-sine-2019-08-18-21-13-45-314"
    print("job_name: %s" % job_name)
    estimator_eval = RLEstimator(entry_point="evaluate_gameserver_ppo.py",
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            role=role,
                            toolkit=RLToolkit.RAY,
                            toolkit_version='0.6.5',
                            framework=RLFramework.TENSORFLOW,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            base_job_name=job_name_prefix + "-evaluation",
                            hyperparameters={
                              "cloudwatch_namespace":cloudwatch_namespace,
                              "gs_inventory_url":gs_inventory_url,
                              "learning_freq":learning_freq,
                              "time_total_s":job_duration_in_seconds,
                              "min_servers":min_servers,
                              "max_servers":max_servers,
                              "gamma":gamma,
                              "action_factor":action_factor,
                              "over_prov_factor":over_prov_factor,
                              "save_model": 1
                            }     
                        )
    estimator_eval.fit({'model': checkpoint_path})
    job_name = estimator_eval.latest_training_job.job_name
    print("Evaluation job: %s" % job_name)


.. parsed-literal::

    job_name: 5obs-local-sine-2019-08-18-21-13-45-314
    [31min __init__[0m
    [31menv_config[0m
    [31m{'cloudwatch_namespace': '5obs-local-sine', 'gs_inventory_url': 'https://4bfiebw6ui.execute-api.us-west-2.amazonaws.com/api/currsine1h/', 'learning_freq': '5', 'max_servers': '100', 'min_servers': '10', 'save_model': '1', 'time_total_s': '32400'}[0m
    [31mself.curr_demand=63.138143498979936[0m
    [31mcalculate the reward, calculate the ratio between allocation and demand, curr_alloc/curr_demand[0m
    [31minterm ratio=1.0151651067081289[0m
    [31mover provision - ratio>1 - -0.9574966835151812[0m
    [31mhttps://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/tensorflow#adapting-your-local-tensorflow-script[0m
    [31m2019-08-19 06:24:38,987 sagemaker-containers INFO     Reporting training SUCCESS[0m
    
    2019-08-19 06:24:43 Uploading - Uploading generated training model
    2019-08-19 06:24:43 Completed - Training job completed
    Billable seconds: 60
    Evaluation job: 5obs-local-sine-evaluation-2019-08-19-06-21-59-623
    CPU times: user 1.65 s, sys: 129 ms, total: 1.78 s
    Wall time: 3min 14s


Hosting
-------

Once the training is done, we can deploy the trained model as an Amazon
SageMaker real-time hosted endpoint. This will allow us to make
predictions (or inference) from the model. Note that we don’t have to
host on the same insantance (or type of instance) that we used to train.
The endpoint deployment can be accomplished as follows:

Model deployment
~~~~~~~~~~~~~~~~

Now let us deploy the RL policy so that we can get the optimal action,
given an environment observation. In case the notebook restarted and
lost its previous estimator object, populate the estimator.model_data
with the full s3 link to the model.tar.gz. e.g.,
s3://sagemaker-us-west-2-356566070122/rl-gameserver-autopilot-2019-07-19-19-36-32-926/output/model.tar.gz

.. code:: ipython3

    from sagemaker.tensorflow.serving import Model
    print ("model name: %s" % estimator.model_data)
    model_data='s3://sagemaker-us-west-2-356566070122/rl-gs-training-2019-09-23-15-41-40-260/output/model.tar.gz'
    model = Model(model_data=model_data,
                  role=role)
    
    predictor = model.deploy(initial_instance_count=1, instance_type=instance_type)


.. parsed-literal::

    model name: s3://sagemaker-us-west-2-356566070122/rl-gs-training-2019-09-23-15-41-40-260/output/model.tar.gz
    -------------------------------------------------------------------------!

Inference
~~~~~~~~~

Now that the trained model is deployed at an endpoint that is
up-and-running, we can use this endpoint for inference. The format of
input should match that of observation_space in the defined environment.
In this example, the observation space is a 15 dimensional vector
formulated from previous and current observations. For the sake of
space, this demo doesn’t include the non-trivial construction process.
Instead, we provide a dummy input below. For more details, please check
src/gameserver_env.py.

.. code:: ipython3

    sagemaker_region = 'us-west-2'
    sagemaker_client = boto3.client('sagemaker-runtime',region_name=sagemaker_region)
    #populate the correct endpoint_name
    endpoint_name ="sagemaker-tensorflow-serving-2019-09-23-20-53-20-237"
    content_type = "application/json"
    accept = "Accept"
    last_observations = np.arange(1, 16)
    
    response = sagemaker_client.invoke_endpoint(
          EndpointName=endpoint_name,
          ContentType=content_type,
          Accept=accept,
          Body=last_observations
        )
    response['Body'].read()

Delete the Endpoint
~~~~~~~~~~~~~~~~~~~

Having an endpoint running will incur some costs. Therefore as a
clean-up job, we should delete the endpoint.

.. code:: ipython3

    predictor.delete_endpoint()
