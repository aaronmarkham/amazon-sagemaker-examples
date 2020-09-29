Horovod Distributed Training with SageMaker TensorFlow script mode.
===================================================================

Horovod is a distributed training framework based on Message Passing
Interfae (MPI). For information about Horovod, see `Horovod
README <https://github.com/uber/horovod>`__.

You can perform distributed training with Horovod on SageMaker by using
the SageMaker Tensorflow container. If MPI is enabled when you create
the training job, SageMaker creates the MPI environment and executes the
``mpirun`` command to execute the training script. Details on how to
configure mpi settings in training job are described later in this
example.

In this example notebook, we create a Horovod training job that uses the
MNIST data set.

Set up the environment
----------------------

We get the ``IAM`` role that this notebook is running as and pass that
role to the TensorFlow estimator that SageMaker uses to get data and
perform training.

.. code:: ipython3

    import sagemaker
    import os
    from sagemaker.utils import sagemaker_timestamp
    from sagemaker.tensorflow import TensorFlow
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    
    default_s3_bucket = sagemaker_session.default_bucket()
    sagemaker_iam_role = get_execution_role()
    
    train_script = "mnist_hvd.py"
    instance_count = 2

Prepare Data for training
-------------------------

Now we download the MNIST dataset to the local ``/tmp/data/`` directory
and then upload it to an S3 bucket. After uploading the dataset to S3,
we delete the data from ``/tmp/data/``.

.. code:: ipython3

    import os
    import shutil
    
    import numpy as np
    
    import keras
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    s3_train_path = "s3://{}/mnist/train.npz".format(default_s3_bucket)
    s3_test_path = "s3://{}/mnist/test.npz".format(default_s3_bucket)
    
    # Create local directory
    ! mkdir -p /tmp/data/mnist_train
    ! mkdir -p /tmp/data/mnist_test
    
    # Save data locally
    np.savez('/tmp/data/mnist_train/train.npz', data=x_train, labels=y_train)
    np.savez('/tmp/data/mnist_test/test.npz', data=x_test, labels=y_test)
    
    # Upload the dataset to s3
    ! aws s3 cp /tmp/data/mnist_train/train.npz $s3_train_path
    ! aws s3 cp /tmp/data/mnist_test/test.npz $s3_test_path
    
    print('training data at ', s3_train_path)
    print('test data at ', s3_test_path)
    ! rm -rf /tmp/data

Write a script for horovod distributed training
-----------------------------------------------

This example is based on the `Keras MNIST horovod
example <https://github.com/uber/horovod/blob/master/examples/keras_mnist.py>`__
example in the horovod github repository.

To run this script we have to make following modifications:

1. Accept ``--model_dir`` as a command-line argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify the script to accept ``model_dir`` as a command-line argument
that defines the directory path (i.e. ``/opt/ml/model/``) where the
output model is saved. Because Sagemaker deletes the training cluster
when training completes, saving the model to ``/opt/ml/model/``
directory prevents the trained model from getting lost, because when the
training job completes, SageMaker writes the data stored in
``/opt/ml/model/`` to an S3 bucket.

This also allows the SageMaker training job to integrate with other
SageMaker services, such as hosted inference endpoints or batch
transform jobs. It also allows you to host the trained model outside of
SageMaker.

The following code adds ``model_dir`` as a command-line argument to the
script:

::

   parser = argparse.ArgumentParser()
   parser.add_argument('--model_dir', type=str)

More details can be found
`here <https://github.com/aws/sagemaker-containers/blob/master/README.rst>`__.

2. Load train and test data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can get local directory path where the ``train`` and ``test`` data
is downloaded by reading the environment variable ``SM_CHANNEL_TRAIN``
and ``SM_CHANNEL_TEST`` respectively. After you get the directory path,
load the data into memory.

Here is the code:

::

   x_train = np.load(os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'train.npz'))['data']
   y_train = np.load(os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'train.npz'))['labels']

   x_test = np.load(os.path.join(os.environ['SM_CHANNEL_TEST'], 'test.npz'))['data']
   y_test = np.load(os.path.join(os.environ['SM_CHANNEL_TEST'], 'test.npz'))['labels']

For a list of all environment variables set by SageMaker that are
accessible inside a training script, see `SageMaker
Containers <https://github.com/aws/sagemaker-containers/blob/master/README.rst>`__.

3. Save the model only at the master node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because in Horovod the training is distributed to multiple nodes, the
model should only be saved by the master node. The following code in the
script does this:

::

   # Horovod: Save model only on worker 0 (i.e. master)
   if hvd.rank() == 0:
       saved_model_path = tf.contrib.saved_model.save_keras_model(model, args.model_dir)

Training script
~~~~~~~~~~~~~~~

Here is the final training script.

.. code:: ipython3

    !cat 'mnist_hvd.py'

Test locally using SageMaker Python SDK TensorFlow Estimator
------------------------------------------------------------

You can use the SageMaker Python SDK TensorFlow estimator to easily
train locally and in SageMaker.

This notebook shows how to use the SageMaker Python SDK to run your code
in a local container before deploying to SageMaker’s managed training or
hosting environments. Just change your estimator’s
``train_instance_type`` to ``local`` or ``local_gpu``. For more
information, see:
https://github.com/aws/sagemaker-python-sdk#local-mode.

To use this feature, you need to install docker-compose (and
nvidia-docker if you are training with a GPU). Run the following script
to install docker-compose or nvidia-docker-compose, and configure the
notebook environment for you.

**Note**: You can only run a single local notebook at a time.

.. code:: ipython3

    !/bin/bash ./setup.sh

To train locally, set ``train_instance_type`` to ``local``:

.. code:: ipython3

    train_instance_type='local'

The MPI environment for Horovod can be configured by setting the
following flags in the ``mpi`` field of the ``distribution`` dictionary
that you pass to the TensorFlow estimator :

-  ``enabled (bool)``: If set to ``True``, the MPI setup is performed
   and ``mpirun`` command is executed.
-  ``processes_per_host (int) [Optional]``: Number of processes MPI
   should launch on each host. Note, this should not be greater than the
   available slots on the selected instance type. This flag should be
   set for the multi-cpu/gpu training.
-  ``custom_mpi_options (str) [Optional]``: Any mpirun flag(s) can be
   passed in this field that will be added to the mpirun command
   executed by SageMaker to launch distributed horovod training.

For more information about the ``distribution`` dictionary, see the
SageMaker Python SDK
`README <https://github.com/aws/sagemaker-python-sdk/blob/v1.17.3/src/sagemaker/tensorflow/README.rst>`__.

First, enable MPI:

.. code:: ipython3

    distributions = {'mpi': {'enabled': True}}

Now, we create the Tensorflow estimator passing the
``train_instance_type`` and ``distribution``

.. code:: ipython3

    estimator_local = TensorFlow(entry_point=train_script,
                           role=sagemaker_iam_role,
                           train_instance_count=instance_count,
                           train_instance_type=train_instance_type,
                           script_mode=True,
                           framework_version='1.15.2',
                           py_version='py3',
                           distributions=distributions,
                           base_job_name='hvd-mnist-local')

Call ``fit()`` to start the local training

.. code:: ipython3

    estimator_local.fit({"train":s3_train_path, "test":s3_test_path})

Train in SageMaker
------------------

After you test the training job locally, run it on SageMaker:

First, change the instance type from ``local`` to the valid EC2 instance
type. For example, ``ml.c4.xlarge``.

.. code:: ipython3

    train_instance_type='ml.c4.xlarge'

You can also provide your custom MPI options by passing in the
``custom_mpi_options`` field of ``distribution`` dictionary that will be
added to the ``mpirun`` command executed by SageMaker:

.. code:: ipython3

    distributions = {'mpi': {'enabled': True, "custom_mpi_options": "-verbose --NCCL_DEBUG=INFO"}}

Now, we create the Tensorflow estimator passing the
``train_instance_type`` and ``distribution`` to launch the training job
in sagemaker.

.. code:: ipython3

    estimator = TensorFlow(entry_point=train_script,
                           role=sagemaker_iam_role,
                           train_instance_count=instance_count,
                           train_instance_type=train_instance_type,
                           script_mode=True,
                           framework_version='1.15.2',
                           py_version='py3',
                           distributions=distributions,
                           base_job_name='hvd-mnist')

Call ``fit()`` to start the training

.. code:: ipython3

    estimator.fit({"train":s3_train_path, "test":s3_test_path})

Horovod training in SageMaker using multiple CPU/GPU
----------------------------------------------------

To enable mulitiple CPUs or GPUs for horovod training, set the
``processes_per_host`` field in the ``mpi`` section of the
``distribution`` dictionary to the desired value of processes that will
be executed per instance.

.. code:: ipython3

    distributions = {'mpi': {'enabled': True, "processes_per_host": 2}}

Now, we create the Tensorflow estimator passing the
``train_instance_type`` and ``distribution``

.. code:: ipython3

    estimator = TensorFlow(entry_point=train_script,
                           role=sagemaker_iam_role,
                           train_instance_count=instance_count,
                           train_instance_type=train_instance_type,
                           script_mode=True,
                           framework_version='1.15.2',
                           py_version='py3',
                           distributions=distributions,
                           base_job_name='hvd-mnist-multi-cpu')

Call ``fit()`` to start the training

.. code:: ipython3

    estimator.fit({"train":s3_train_path, "test":s3_test_path})

Improving horovod training performance on SageMaker
---------------------------------------------------

Performing Horovod training inside a VPC improves the network latency
between nodes, leading to higher performance and stability of Horovod
training jobs.

For a detailed explanation of how to configure a VPC for SageMaker
training, see `Secure Training and Inference with
VPC <https://github.com/aws/sagemaker-python-sdk#secure-training-and-inference-with-vpc>`__.

Setup VPC infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~

We will setup following resources as part of VPC stack: \* ``VPC``: AWS
Virtual private cloud with CIDR block. \* ``Subnets``: Two subnets with
the CIDR blocks ``10.0.0.0/24`` and ``10.0.1.0/24`` \*
``Security Group``: Defining the open ingress and egress ports, such as
TCP. \* ``VpcEndpoint``: S3 Vpc endpoint allowing sagemaker’s vpc
cluster to dosenload data from S3. \* ``Route Table``: Defining routes
and is tied to subnets and VPC.

Complete cloud formation template for setting up the VPC stack can be
seen `here <./vpc_infra_cfn.json>`__.

.. code:: ipython3

    import boto3
    from botocore.exceptions import ClientError
    from time import sleep
    
    def create_vpn_infra(stack_name="hvdvpcstack"):
        cfn = boto3.client("cloudformation")
    
        cfn_template = open("vpc_infra_cfn.json", "r").read()
        
        try:
            vpn_stack = cfn.create_stack(StackName=(stack_name),
                                         TemplateBody=cfn_template)
        except ClientError as e:
            if e.response['Error']['Code'] == 'AlreadyExistsException':
                print("Stack: {} already exists, so skipping stack creation.".format(stack_name))
            else:
                print("Unexpected error: %s" % e)
                raise e
    
        describe_stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    
        while describe_stack["StackStatus"] == "CREATE_IN_PROGRESS":
            describe_stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
            sleep(0.5)
    
        if describe_stack["StackStatus"] != "CREATE_COMPLETE":
            raise ValueError("Stack creation failed in state: {}".format(describe_stack["StackStatus"]))
    
        print("Stack: {} created successfully with status: {}".format(stack_name, describe_stack["StackStatus"]))
    
        subnets = []
        security_groups = []
    
        for output_field in describe_stack["Outputs"]:
    
            if output_field["OutputKey"] == "SecurityGroupId":
                security_groups.append(output_field["OutputValue"])
            if output_field["OutputKey"] == "Subnet1Id" or output_field["OutputKey"] == "Subnet2Id":
                subnets.append(output_field["OutputValue"])
    
        return subnets, security_groups
    
    
    subnets, security_groups = create_vpn_infra()
    print("Subnets: {}".format(subnets))
    print("Security Groups: {}".format(security_groups))

VPC training in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we create the Tensorflow estimator, passing the
``train_instance_type`` and ``distribution``.

.. code:: ipython3

    estimator = TensorFlow(entry_point=train_script,
                           role=sagemaker_iam_role,
                           train_instance_count=instance_count,
                           train_instance_type=train_instance_type,
                           script_mode=True,
                           framework_version='1.15.2',
                           py_version='py3',
                           distributions=distributions,
                           security_group_ids=security_groups,
                           subnets=subnets,
                           base_job_name='hvd-mnist-vpc')

Call ``fit()`` to start the training

.. code:: ipython3

    estimator.fit({"train":s3_train_path, "test":s3_test_path})

After training is completed, you can host the saved model by using
TensorFlow Serving on SageMaker. For an example that uses TensorFlow
Serving, see
`(https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_serving_container/tensorflow_serving_container.ipynb <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_serving_container/tensorflow_serving_container.ipynb>`__.

Reference Links:
----------------

-  `SageMaker Container MPI
   Support. <https://github.com/aws/sagemaker-containers/blob/master/src/sagemaker_containers/_mpi.py>`__
-  `Horovod Official Documentation <https://github.com/uber/horovod>`__
-  `SageMaker Tensorflow script mode
   example. <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_script_mode_quickstart/tensorflow_script_mode_quickstart.ipynb>`__
