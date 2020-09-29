Bring Your Own Pipe-mode Algorithm
==================================

**Create a Docker container for training SageMaker algorithms using
Pipe-mode**

--------------

--------------

Contents
--------

1. `Overview <#Overview>`__
2. `Preparation <#Preparation>`__
3. `Permissions <#Permissions>`__
4. `Code <#Code>`__
5. `train.py <#train.py>`__
6. `Dockerfile <#Dockerfile>`__
7. `Publish <#Publish>`__
8. `Train <#Train>`__
9. `Conclusion <#Conclusion>`__

Preparation
-----------

*This notebook was created and tested on an ml.t2.medium notebook
instance.*

Let’s start by specifying:

-  S3 URIs ``s3_training_input`` and ``s3_model_output`` that you want
   to use for training input and model data respectively. These should
   be within the same region as the Notebook Instance, training, and
   hosting. Since the “algorithm” we’re building here doesn’t really
   have any specific data-format, feel free to point
   ``s3_training_input`` to any s3 dataset you have, the bigger the
   dataset the better to test the raw IO throughput performance.
-  The ``training_instance_type`` to use for training. More powerful
   instance types have more CPU and bandwidth which would result in
   higher throughput.
-  The IAM role arn used to give training access to your data.

Permissions
~~~~~~~~~~~

Running this notebook requires permissions in addition to the normal
``SageMakerFullAccess`` permissions. This is because we’ll be creating a
new repository in Amazon ECR. The easiest way to add these permissions
is simply to add the managed policy
``AmazonEC2ContainerRegistryFullAccess`` to the role that you used to
start your notebook instance. There’s no need to restart your notebook
instance when you do this, the new permissions will be available
immediately.

.. code:: ipython2

    s3_training_input = 's3://<your_s3_bucket_name_here>/<training_data_prefix>/'
    s3_model_output = 's3://<your_s3_bucket_name_here/<model_output_prefix>/'
    # We're using a cheaper instance here, switch to a higher-end ml.c5.18xlarge
    # to achieve much higher throughput performance:
    training_instance_type = "ml.m4.xlarge"
    
    
    # Define IAM role
    import boto3
    import re
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    role = get_execution_role()

--------------

Code
----

For the purposes of this demo we’re going to write an extremely simple
“training” algorithm in Python. In essence it will conform to the
specifications required by SageMaker Training and will read data in
Pipe-mode but will do nothing with the data, simply reading it and
throwing it away. We’re doing it this way to be able to illustrate only
exactly what’s needed to support Pipe-mode without complicating the code
with a real training algorithm.

In Pipe-mode, data is pre-fetched from S3 at high-concurrency and
throughput and streamed into Unix Named Pipes (aka FIFOs) - one FIFO per
Channel per epoch. The algorithm must open the FIFO for reading and read
through to (or optionally abort mid-stream) and close its end of the
file descriptor when done. It can then optionally wait for the next
epoch’s FIFO to get created and commence reading, iterating through
epochs until it has achieved its completion criteria.

For this example, we’ll need two supporting files:

train.py
~~~~~~~~

``train.py`` simply iterates through 5 epochs on the ``training``
Channel. Each epoch involves reading the training data stream from a
FIFO named ``/opt/ml/input/data/training_${epoch}``. At the end of the
epoch the code simply iterates to the next epoch, waits for the new
epoch’s FIFO to get created and continues on.

A lot of the code in ``train.py`` is merely boilerplate code, dealing
with printing log messages, trapping termination signals etc. The main
code that iterates through reading each epoch’s data through its
corresponding FIFO is the following:

.. code:: python

   # we're allocating a byte array here to read data into, a real algo
   # may opt to prefetch the data into a memory buffer and train in
   # in parallel so that both IO and training happen simultaneously
   data = bytearray(16777216)
   total_read = 0
   total_duration = 0
   for epoch in range(num_epochs):
       check_termination()
       epoch_bytes_read = 0
       # As per SageMaker Training spec, the FIFO's path will be based on
       # the channel name and the current epoch:
       fifo_path = '{0}/{1}_{2}'.format(data_dir, channel_name, epoch)

       # Usually the fifo will already exist by the time we get here, but
       # to be safe we should wait to confirm:
       wait_till_fifo_exists(fifo_path)
       with open(fifo_path, 'rb', buffering=0) as fifo:
           print('opened fifo: %s' % fifo_path)
           # Now simply iterate reading from the file until EOF. Again, a
           # real algorithm will actually do something with the data
           # rather than simply reading and immediately discarding like we
           # are doing here
           start = time.time()
           bytes_read = fifo.readinto(data)
           total_read += bytes_read
           epoch_bytes_read += bytes_read
           while bytes_read > 0 and not terminated:
               bytes_read = fifo.readinto(data)
               total_read += bytes_read
               epoch_bytes_read += bytes_read

           duration = time.time() - start
           total_duration += duration
           epoch_throughput = epoch_bytes_read / duration / 1000000
           print('Completed epoch %s; read %s bytes; time: %.2fs, throughput: %.2f MB/s'
                 % (epoch, epoch_bytes_read, duration, epoch_throughput))

Dockerfile
~~~~~~~~~~

Smaller containers are preferred for Amazon SageMaker as they lead to
faster spin up times in training and endpoint creation, so this
container is kept minimal. It simply starts with Alpine (a minimal Linux
install) with python then adds ``train.py``, and finally runs
``train.py`` when the entrypoint is launched.

.. code:: dockerfile

   # use minimal alpine base image as we only need python and nothing else here
   FROM python:2-alpine3.6

   MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

   COPY train.py /train.py

   ENTRYPOINT ["python2.7", "-u", "/train.py"]

--------------

Publish
-------

Now, to publish this container to ECR, we’ll run the comands below.

This command will take several minutes to run the first time.

.. code:: sh

    %%sh
    
    # The name of our algorithm
    algorithm_name=sagemaker-pipe-demo
    
    set -eu # stop if anything fails
    
    account=$(aws sts get-caller-identity --query Account --output text)
    
    # Get the region defined in the current configuration (default to us-west-2 if none defined)
    region=$(aws configure get region)
    region=${region:-us-west-2}
    
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
    
    # If the repository doesn't exist in ECR, create it.
    
    aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
    
    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
    fi
    
    # Get the login command from ECR and execute it directly
    $(aws ecr get-login --region ${region} --no-include-email)
    
    # Build the docker image locally with the image name and then push it to ECR
    # with the full name.
    docker build  -t ${algorithm_name} .
    docker tag ${algorithm_name} ${fullname}
    
    docker push ${fullname}

--------------

Train
-----

Now, let’s setup the information needed to run the training container in
SageMaker.

First, we’ll get our region and account information so that we can point
to the ECR container we just created.

.. code:: ipython2

    region = boto3.Session().region_name
    account = boto3.client('sts').get_caller_identity().get('Account')

-  Specify the role to use
-  Give the training job a name
-  Point the algorithm to the container we created
-  Specify training instance resources
-  Point to the S3 location of our input data and the ``training``
   channel expected by our algorithm
-  Point to the S3 location for output
-  Maximum run time

.. code:: ipython2

    import time
    import json
    import os
    
    pipe_job = 'DEMO-pipe-byo-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    
    print("Training job", pipe_job)
    
    training_params = {
        "RoleArn": role,
        "TrainingJobName": pipe_job,
        "AlgorithmSpecification": {
            "TrainingImage": '{}.dkr.ecr.{}.amazonaws.com/sagemaker-pipe-demo:latest'.format(account, region),
            "TrainingInputMode": "Pipe"
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "{}".format(training_instance_type),
            "VolumeSizeInGB": 1
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "{}".format(s3_training_input),
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None"
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": "{}".format(s3_model_output)
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 60 * 60
        }
    }

Now let’s kick off our training job on Amazon SageMaker Training using
the parameters we just created. Because training is managed (AWS takes
care of spinning up and spinning down the hardware), we don’t have to
wait for our job to finish to continue, but for this case, let’s setup a
waiter so we can monitor the status of our training.

.. code:: ipython2

    %%time
    
    sm_session = Session()
    sm = boto3.client('sagemaker')
    sm.create_training_job(**training_params)
    
    status = sm.describe_training_job(TrainingJobName=pipe_job)['TrainingJobStatus']
    print(status)
    sm_session.logs_for_job(job_name=pipe_job, wait=True)
    sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=pipe_job)
    status = sm.describe_training_job(TrainingJobName=pipe_job)['TrainingJobStatus']
    print("Training job ended with status: " + status)
    if status == 'Failed':
        message = sm.describe_training_job(TrainingJobName=pipe_job)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')

Note the throughput logged by the training logs above. By way of
comparison a File-mode algorithm will achieve at most approximately
150MB/s on a high-end ``ml.c5.18xlarge`` and approximately 75MB/s on a
``ml.m4.xlarge``.

--------------

Conclusion
----------

There are a few situations where Pipe-mode may not be the optimum choice
for training in which case you should stick to using File-mode:

-  If your algorithm needs to backtrack or skip ahead within an epoch.
   This is simply not possible in Pipe-mode since the underlying FIFO
   cannot not support ``lseek()`` operations.
-  If your training dataset is small enough to fit in memory and you
   need to run multiple epochs. In this case may be quicker and easier
   just to load it all into memory and iterate.
-  Your training dataset is not easily parse-able from a streaming
   source.

In all other scenarios, if you have an IO-bound training algorithm,
switching to Pipe-mode may give you a significant throughput-boost and
will reduce the size of the disk volume required. This should result in
both saving you time and reducing training costs.

You can read more about building your own training algorithms in the
`SageMaker Training
documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html>`__.
