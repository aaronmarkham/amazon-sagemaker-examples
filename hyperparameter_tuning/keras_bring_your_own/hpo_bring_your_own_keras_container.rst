Hyperparameter Tuning using Your Own Keras/Tensorflow Container
===============================================================

This notebook shows how to build your own Keras(Tensorflow) container,
test it locally using SageMaker Python SDK local mode, and bring it to
SageMaker for training, leveraging hyperparameter tuning.

The model used for this notebook is a ResNet model, trainer with the
CIFAR-10 dataset. The example is based on
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

Set up the notebook instance to support local mode
--------------------------------------------------

Currently you need to install docker-compose in order to use local mode
(i.e., testing the container in the notebook instance without pushing it
to ECR).

.. code:: ipython3

    !/bin/bash setup.sh

Permissions
-----------

Running this notebook requires permissions in addition to the normal
``SageMakerFullAccess`` permissions. This is because it creates new
repositories in Amazon ECR. The easiest way to add these permissions is
simply to add the managed policy
``AmazonEC2ContainerRegistryFullAccess`` to the role that you used to
start your notebook instance. There’s no need to restart your notebook
instance when you do this, the new permissions will be available
immediately.

Set up the environment
----------------------

We will set up a few things before starting the workflow.

1. get the execution role which will be passed to sagemaker for
   accessing your resources such as s3 bucket
2. specify the s3 bucket and prefix where training data set and model
   artifacts are stored

.. code:: ipython3

    import os
    import numpy as np
    import tempfile
    
    import tensorflow as tf
    
    import sagemaker
    import boto3
    from sagemaker.estimator import Estimator
    
    region = boto3.Session().region_name
    
    sagemaker_session = sagemaker.Session()
    smclient = boto3.client('sagemaker')
    
    bucket = sagemaker.Session().default_bucket()  # s3 bucket name, must be in the same region as the one specified above
    prefix = 'sagemaker/DEMO-hpo-keras-cifar10'
    
    role = sagemaker.get_execution_role()
    
    NUM_CLASSES = 10   # the data set has 10 categories of images

Complete source code
--------------------

-  `trainer/start.py <trainer/start.py>`__: Keras model
-  `trainer/environment.py <trainer/environment.py>`__: Contain
   information about the SageMaker environment

Building the image
------------------

We will build the docker image using the Tensorflow versions on
dockerhub. The full list of Tensorflow versions can be found at
https://hub.docker.com/r/tensorflow/tensorflow/tags/

.. code:: ipython3

    import shlex
    import subprocess
    
    def get_image_name(ecr_repository, tensorflow_version_tag):
        return '%s:tensorflow-%s' % (ecr_repository, tensorflow_version_tag)
    
    def build_image(name, version):
        cmd = 'docker build -t %s --build-arg VERSION=%s -f Dockerfile .' % (name, version)
        subprocess.check_call(shlex.split(cmd))
    
    #version tag can be found at https://hub.docker.com/r/tensorflow/tensorflow/tags/ 
    #e.g., latest cpu version is 'latest', while latest gpu version is 'latest-gpu'
    tensorflow_version_tag = '1.10.1'   
    
    account = boto3.client('sts').get_caller_identity()['Account']
    
    domain = 'amazonaws.com'
    if (region == 'cn-north-1' or region == 'cn-northwest-1'):
        domain = 'amazonaws.com.cn'
    
    ecr_repository="%s.dkr.ecr.%s.%s/test" %(account,region,domain) # your ECR repository, which you should have been created before running the notebook
    
    image_name = get_image_name(ecr_repository, tensorflow_version_tag)
    
    print('building image:'+image_name)
    build_image(image_name, tensorflow_version_tag)

Prepare the data
----------------

.. code:: ipython3

    def upload_channel(channel_name, x, y):
        y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    
        file_path = tempfile.mkdtemp()
        np.savez_compressed(os.path.join(file_path, 'cifar-10-npz-compressed.npz'), x=x, y=y)
    
        return sagemaker_session.upload_data(path=file_path, bucket=bucket, key_prefix='data/DEMO-keras-cifar10/%s' % channel_name)
    
    
    def upload_training_data():
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
        train_data_location = upload_channel('train', x_train, y_train)
        test_data_location = upload_channel('test', x_test, y_test)
    
        return {'train': train_data_location, 'test': test_data_location}
    
    channels = upload_training_data()


Testing the container locally (optional)
----------------------------------------

You can test the container locally using local mode of SageMaker Python
SDK. A training container will be created in the notebook instance based
on the docker image you built. Note that we have not pushed the docker
image to ECR yet since we are only running local mode here. You can skip
to the tuning step if you want but testing the container locally can
help you find issues quickly before kicking off the tuning job.

Setting the hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    hyperparameters = dict(batch_size=32, data_augmentation=True, learning_rate=.0001, 
                           width_shift_range=.1, height_shift_range=.1, epochs=1)
    hyperparameters

Create a training job using local mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    
    output_location = "s3://{}/{}/output".format(bucket,prefix)
    
    estimator = Estimator(image_name, role=role, output_path=output_location,
                          train_instance_count=1, 
                          train_instance_type='local', hyperparameters=hyperparameters)
    estimator.fit(channels)

Pushing the container to ECR
----------------------------

Now that we’ve tested the container locally and it works fine, we can
move on to run the hyperparmeter tuning. Before kicking off the tuning
job, you need to push the docker image to ECR first.

The cell below will create the ECR repository, if it does not exist yet,
and push the image to ECR.

.. code:: ipython3

    # The name of our algorithm
    algorithm_name = 'test'
    
    # If the repository doesn't exist in ECR, create it.
    exist_repo = !aws ecr describe-repositories --repository-names {algorithm_name} > /dev/null 2>&1
    
    if not exist_repo:
        !aws ecr create-repository --repository-name {algorithm_name} > /dev/null
    
    # Get the login command from ECR and execute it directly
    !$(aws ecr get-login --region {region} --no-include-email)
    
    !docker push {image_name}

Specify hyperparameter tuning job configuration
-----------------------------------------------

*Note, with the default setting below, the hyperparameter tuning job can
take 20~30 minutes to complete. You can customize the code in order to
get better result, such as increasing the total number of training jobs,
epochs, etc., with the understanding that the tuning time will be
increased accordingly as well.*

Now you configure the tuning job by defining a JSON object that you pass
as the value of the TuningJobConfig parameter to the create_tuning_job
call. In this JSON object, you specify: \* The ranges of hyperparameters
you want to tune \* The limits of the resource the tuning job can
consume \* The objective metric for the tuning job

.. code:: ipython3

    import json
    from time import gmtime, strftime
    
    tuning_job_name = 'BYO-keras-tuningjob-' + strftime("%d-%H-%M-%S", gmtime())
    
    print(tuning_job_name)
    
    tuning_job_config = {
        "ParameterRanges": {
          "CategoricalParameterRanges": [],
          "ContinuousParameterRanges": [
            {
              "MaxValue": "0.001",
              "MinValue": "0.0001",
              "Name": "learning_rate",          
            }
          ],
          "IntegerParameterRanges": []
        },
        "ResourceLimits": {
          "MaxNumberOfTrainingJobs": 9,
          "MaxParallelTrainingJobs": 3
        },
        "Strategy": "Bayesian",
        "HyperParameterTuningJobObjective": {
          "MetricName": "loss",
          "Type": "Minimize"
        }
      }


Specify training job configuration
----------------------------------

Now you configure the training jobs the tuning job launches by defining
a JSON object that you pass as the value of the TrainingJobDefinition
parameter to the create_tuning_job call. In this JSON object, you
specify: \* Metrics that the training jobs emit \* The container image
for the algorithm to train \* The input configuration for your training
and test data \* Configuration for the output of the algorithm \* The
values of any algorithm hyperparameters that are not tuned in the tuning
job \* The type of instance to use for the training jobs \* The stopping
condition for the training jobs

This example defines one metric that Tensorflow container emits: loss.

.. code:: ipython3

    training_image = image_name
    
    print('training artifacts will be uploaded to: {}'.format(output_location))
    
    training_job_definition = {
        "AlgorithmSpecification": {
          "MetricDefinitions": [
            {
              "Name": "loss",
              "Regex": "loss: ([0-9\\.]+)"
            }
          ],
          "TrainingImage": training_image,
          "TrainingInputMode": "File"
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": channels['train'],
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None"
            },
            {
                "ChannelName": "test",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": channels['test'],
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },            
                "CompressionType": "None",
                "RecordWrapperType": "None"            
            }
        ],
        "OutputDataConfig": {
          "S3OutputPath": "s3://{}/{}/output".format(bucket,prefix)
        },
        "ResourceConfig": {
          "InstanceCount": 1,
          "InstanceType": "ml.m4.xlarge",
          "VolumeSizeInGB": 50
        },
        "RoleArn": role,
        "StaticHyperParameters": {
            "batch_size":"32",
            "data_augmentation":"True",
            "height_shift_range":"0.1",
            "width_shift_range":"0.1",
            "epochs":'1'
        },
        "StoppingCondition": {
          "MaxRuntimeInSeconds": 43200
        }
    }


Create and launch a hyperparameter tuning job
---------------------------------------------

Now you can launch a hyperparameter tuning job by calling
create_tuning_job API. Pass the name and JSON objects you created in
previous steps as the values of the parameters. After the tuning job is
created, you should be able to describe the tuning job to see its
progress in the next step, and you can go to SageMaker console->Jobs to
check out the progress of each training job that has been created.

.. code:: ipython3

    smclient.create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name,
                                                   HyperParameterTuningJobConfig = tuning_job_config,
                                                   TrainingJobDefinition = training_job_definition)

Let’s just run a quick check of the hyperparameter tuning jobs status to
make sure it started successfully and is ``InProgress``.

.. code:: ipython3

    smclient.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name)['HyperParameterTuningJobStatus']

Analyze tuning job results - after tuning job is completed
----------------------------------------------------------

Please refer to “HPO_Analyze_TuningJob_Results.ipynb” to see example
code to analyze the tuning job results.

Deploy the best model
---------------------

Now that we have got the best model, we can deploy it to an endpoint.
Please refer to other SageMaker sample notebooks or SageMaker
documentation to see how to deploy a model.
