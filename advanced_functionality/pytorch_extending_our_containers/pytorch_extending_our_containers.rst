Extending our PyTorch containers
================================

With Amazon SageMaker, you can package your own algorithms that can then
be trained and deployed in the SageMaker environment. This notebook
guides you through an example on how to extend one of our existing and
predefined SageMaker deep learning framework containers.

By packaging an algorithm in a container, you can bring almost any code
to the Amazon SageMaker environment, regardless of programming language,
environment, framework, or dependencies.

1.  `Extending our PyTorch
    containers <#Extending-our-pytorch-containers>`__
2.  `When should I extend a SageMaker
    container? <#When-should-I-extend-a-SageMaker-container?>`__
3.  `Permissions <#Permissions>`__
4.  `The example <#The-example>`__
5.  `The presentation <#The-presentation>`__
6.  `Part 1: Packaging and Uploading your Algorithm for use with Amazon
    SageMaker <#Part-1:-Packaging-and-Uploading-your-Algorithm-for-use-with-Amazon-SageMaker>`__

    1. `An overview of Docker <#An-overview-of-Docker>`__
    2. `How Amazon SageMaker runs your Docker
       container <#How-Amazon-SageMaker-runs-your-Docker-container>`__
    3. `Running your container during
       training <#Running-your-container-during-training>`__ 1. `The
       input <#The-input>`__ 1. `The output <#The-output>`__
    4. `Running your container during
       hosting <#Running-your-container-during-hosting>`__
    5. `The parts of the sample
       container <#The-parts-of-the-sample-container>`__
    6. `The Dockerfile <#The-Dockerfile>`__
    7. `Building and registering the
       container <#Building-and-registering-the-container>`__

7.  `Testing your algorithm on your local
    machine <#Testing-your-algorithm-on-your-local-machine>`__
8.  `Download the CIFAR-10 dataset <#Download-the-CIFAR-10-dataset>`__
9.  `SageMaker Python SDK Local
    Training <#SageMaker-Python-SDK-Local-Training>`__
10. `Fit, Deploy, Predict <#Fit,-Deploy,-Predict>`__
11. `Making predictions using Python
    SDK <#Making-predictions-using-Python-SDK>`__
12. `Part 2: Training and Hosting your Algorithm in Amazon
    SageMaker <#Part-2:-Training-and-Hosting-your-Algorithm-in-Amazon-SageMaker>`__
13. `Set up the environment <#Set-up-the-environment>`__
14. `Create the session <#Create-the-session>`__
15. `Upload the data for training <#Upload-the-data-for-training>`__
16. `Training On SageMaker <#Training-on-SageMaker>`__
17. `Optional cleanup <#Optional-cleanup>`__
18. `Reference <#Reference>`__

*or* I’m impatient, just `let me see the code <#The-Dockerfile>`__!

When should I extend a SageMaker container?
-------------------------------------------

You may not need to create a container to bring your own code to Amazon
SageMaker. When you are using a framework such as
`TensorFlow <https://github.com/aws/sagemaker-tensorflow-container>`__,
`MXNet <https://github.com/aws/sagemaker-mxnet-container>`__,
`PyTorch <https://github.com/aws/sagemaker-pytorch-container>`__ or
`Chainer <https://github.com/aws/sagemaker-chainer-container>`__ that
has direct support in SageMaker, you can simply supply the Python code
that implements your algorithm using the SDK entry points for that
framework.

Even if there is direct SDK support for your environment or framework,
you may want to add additional functionality or configure your container
environment differently while utilizing our container to use on
SageMaker.

**Some of the reasons to extend a SageMaker deep learning framework
container are:** 1. Install additional dependencies. (E.g. I want to
install a specific Python library, that the current SageMaker containers
don’t install.) 2. Configure your environment. (E.g. I want to add an
environment variable to my container.)

**Although it is possible to extend any of our framework containers as a
parent image, the example this notebook covers is currently only
intended to work with our PyTorch (0.4.0+) and Chainer (4.1.0+)
containers.**

This walkthrough shows that it is quite straightforward to extend one of
our containers to build your own custom container for PyTorch or
Chainer.

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

The example
-----------

In this example we show how to package a PyTorch container, extending
the SageMaker PyTorch container, with a Python example which works with
the CIFAR-10 dataset. By extending the SageMaker PyTorch container we
can utilize the existing training and hosting solution made to work on
SageMaker. By comparison, if one were to build their own custom
framework container from scratch, they would need to implement a
training and hosting solution in order to use SageMaker. Here is an
example showing `how to create a SageMaker TensorFlow container from
scratch <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb>`__.

In this example, we use a single image to support training and hosting.
This simplifies the procedure because we only need to manage one image
for both tasks. Sometimes you may want separate images for training and
hosting because they have different requirements. In this case, separate
the parts discussed below into separate Dockerfiles and build two
images. Choosing whether to use a single image or two images is a matter
of what is most convenient for you to develop and manage.

If you’re only using Amazon SageMaker for training or hosting, but not
both, only the functionality used needs to be built into your container.

The presentation
----------------

This presentation is divided into two parts: *building* the container
and *using* the container.

Part 1: Packaging and Uploading your Algorithm for use with Amazon SageMaker
============================================================================

An overview of Docker
~~~~~~~~~~~~~~~~~~~~~

If you’re familiar with Docker already, you can skip ahead to the next
section.

For many data scientists, Docker containers are a new technology. But
they are not difficult and can significantly simplify the deployment of
your software packages.

Docker provides a simple way to package arbitrary code into an *image*
that is totally self-contained. Once you have an image, you can use
Docker to run a *container* based on that image. Running a container is
just like running a program on the machine except that the container
creates a fully self-contained environment for the program to run.
Containers are isolated from each other and from the host environment,
so the way your program is set up is the way it runs, no matter where
you run it.

Docker is more powerful than environment managers like conda or
virtualenv because (a) it is completely language independent and (b) it
comprises your whole operating environment, including startup commands,
and environment variable.

A Docker container is like a virtual machine, but it is much lighter
weight. For example, a program running in a container can start in less
than a second and many containers can run simultaneously on the same
physical or virtual machine instance.

Docker uses a simple file called a ``Dockerfile`` to specify how the
image is assembled. An example is provided below. You can build your
Docker images based on Docker images built by yourself or by others,
which can simplify things quite a bit.

Docker has become very popular in programming and devops communities due
to its flexibility and its well-defined specification of how code can be
run in its containers. It is the underpinning of many services built in
the past few years, such as `Amazon
ECS <https://aws.amazon.com/ecs/>`__.

Amazon SageMaker uses Docker to allow users to train and deploy
arbitrary algorithms.

In Amazon SageMaker, Docker containers are invoked in a one way for
training and another, slightly different, way for hosting. The following
sections outline how to build containers for the SageMaker environment.

Some helpful links:

-  `Docker home page <http://www.docker.com>`__
-  `Getting started with
   Docker <https://docs.docker.com/get-started/>`__
-  `Dockerfile
   reference <https://docs.docker.com/engine/reference/builder/>`__
-  ```docker run``
   reference <https://docs.docker.com/engine/reference/run/>`__

How Amazon SageMaker runs your Docker container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because you can run the same image in training or hosting, Amazon
SageMaker runs your container with the argument ``train`` or ``serve``.
How your container processes this argument depends on the container. All
SageMaker deep learning framework containers already cover this
requirement and will trigger your defined training algorithm and
inference code.

-  If you specify a program as an ``ENTRYPOINT`` in the Dockerfile, that
   program will be run at startup and its first argument will be
   ``train`` or ``serve``. The program can then look at that argument
   and decide what to do. The original ``ENTRYPOINT`` specified within
   the SageMaker PyTorch is
   `here <https://github.com/aws/deep-learning-containers/blob/master/pytorch/training/docker/1.5.1/py3/Dockerfile.cpu#L123>`__.

Running your container during training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, our SageMaker PyTorch container utilizes
`console_scripts <http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point>`__
to make use of the ``train`` command issued at training time. The line
that gets invoked during ``train`` is defined within the setup.py file
inside `SageMaker
Containers <https://github.com/aws/sagemaker-containers/blob/master/setup.py#L48>`__,
our common SageMaker deep learning container framework. When this
command is run, it will invoke the `trainer
class <https://github.com/aws/sagemaker-containers/blob/master/src/sagemaker_containers/cli/train.py>`__
to run, which will finally invoke our `PyTorch container
code <https://github.com/aws/sagemaker-pytorch-container/blob/master/src/sagemaker_pytorch_container/training.py>`__
to run your Python file.

A number of files are laid out for your use, under the ``/opt/ml``
directory:

::

   /opt/ml
   |-- input
   |   |-- config
   |   |   |-- hyperparameters.json
   |   |   `-- resourceConfig.json
   |   `-- data
   |       `-- <channel_name>
   |           `-- <input data>
   |-- model
   |   `-- <model files>
   `-- output
       `-- failure

The input
'''''''''

-  ``/opt/ml/input/config`` contains information to control how your
   program runs. ``hyperparameters.json`` is a JSON-formatted dictionary
   of hyperparameter names to values. These values are always strings,
   so you may need to convert them. ``resourceConfig.json`` is a
   JSON-formatted file that describes the network layout used for
   distributed training.
-  ``/opt/ml/input/data/<channel_name>/`` (for File mode) contains the
   input data for that channel. The channels are created based on the
   call to CreateTrainingJob but it’s generally important that channels
   match algorithm expectations. The files for each channel are copied
   from S3 to this directory, preserving the tree structure indicated by
   the S3 key structure.
-  ``/opt/ml/input/data/<channel_name>_<epoch_number>`` (for Pipe mode)
   is the pipe for a given epoch. Epochs start at zero and go up by one
   each time you read them. There is no limit to the number of epochs
   that you can run, but you must close each pipe before reading the
   next epoch.

The output
''''''''''

-  ``/opt/ml/model/`` is the directory where you write the model that
   your algorithm generates. Your model can be in any format that you
   want. It can be a single file or a whole directory tree. SageMaker
   packages any files in this directory into a compressed tar archive
   file. This file is made available at the S3 location returned in the
   ``DescribeTrainingJob`` result.
-  ``/opt/ml/output`` is a directory where the algorithm can write a
   file ``failure`` that describes why the job failed. The contents of
   this file are returned in the ``FailureReason`` field of the
   ``DescribeTrainingJob`` result. For jobs that succeed, there is no
   reason to write this file as it is ignored.

Running your container during hosting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hosting has a very different model than training because hosting is
reponding to inference requests that come in via HTTP. Currently, the
SageMaker PyTorch containers
`uses <https://github.com/aws/sagemaker-pytorch-container/blob/master/src/sagemaker_pytorch_container/serving.py#L103>`__
our `recommended Python serving
stack <https://github.com/aws/sagemaker-containers/blob/master/src/sagemaker_containers/_server.py#L44>`__
to provide robust and scalable serving of inference requests:

.. figure:: stack.png
   :alt: Request serving stack

   Request serving stack

Amazon SageMaker uses two URLs in the container:

-  ``/ping`` receives ``GET`` requests from the infrastructure. Your
   program returns 200 if the container is up and accepting requests.
-  ``/invocations`` is the endpoint that receives client inference
   ``POST`` requests. The format of the request and the response is up
   to the algorithm. If the client supplied ``ContentType`` and
   ``Accept`` headers, these are passed in as well.

The container has the model files in the same place that they were
written to during training:

::

   /opt/ml
   `-- model
       `-- <model files>

The parts of the sample container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``container`` directory has all the components you need to extend
the SageMaker PyTorch container to use as an sample algorithm:

::

   .
   |-- Dockerfile
   |-- build_and_push.sh
   `-- cifar10
       `-- cifar10.py

Let’s discuss each of these in turn:

-  **``Dockerfile``** describes how to build your Docker container
   image. More details are provided below.
-  **``build_and_push.sh``** is a script that uses the Dockerfile to
   build your container images and then pushes it to ECR. We invoke the
   commands directly later in this notebook, but you can just copy and
   run the script for your own algorithms.
-  **``cifar10``** is the directory which contains our user code to be
   invoked.

In this simple application, we install only one file in the container.
You may only need that many, but if you have many supporting routines,
you may wish to install more.

The files that we put in the container are:

-  **``cifar10.py``** is the program that implements our training
   algorithm and handles loading our model for inferences.

The Dockerfile
~~~~~~~~~~~~~~

The Dockerfile describes the image that we want to build. You can think
of it as describing the complete operating system installation of the
system that you want to run. A Docker container running is quite a bit
lighter than a full operating system, however, because it takes
advantage of Linux on the host machine for the basic operations.

We start from the SageMaker PyTorch image as the base. The base image is
an ECR image, so it will have the following pattern. \*
{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-{framework}:{framework_version}-{processor_type}-{python_version}

Here is an explanation of each field. 1. account - AWS account ID the
ECR image belongs to. Our public deep learning framework images are all
under the 520713654638 account. 2. region - The region the ECR image
belongs to. `Available
regions <https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/>`__.
3. framework - The deep learning framework. 4. framework_version - The
version of the deep learning framework. 5. processor_type - CPU or GPU.
6. python_version - The supported version of Python.

So the SageMaker PyTorch ECR image would be:
520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch:0.4.0-cpu-py3

Information on supported frameworks and versions can be found in this
`README <https://github.com/aws/sagemaker-python-sdk>`__.

Next, we add the code that implements our specific algorithm to the
container and set up the right environment for it to run under.

**DISCLAIMER: As of now, the support for the two environment variables
below are only supported for the SageMaker Chainer (4.1.0+) and PyTorch
(0.4.0+) containers.**

Finally, we need to specify two environment variables. 1.
SAGEMAKER_SUBMIT_DIRECTORY - the directory within the container
containing our Python script for training and inference. 2.
SAGEMAKER_PROGRAM - the Python script that should be invoked for
training and inference.

Let’s look at the Dockerfile for this example.

.. code:: ipython3

    !cat container/Dockerfile


.. parsed-literal::

    # Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License"). You
    # may not use this file except in compliance with the License. A copy of
    # the License is located at
    #
    #     http://aws.amazon.com/apache2.0/
    #
    # or in the "license" file accompanying this file. This file is
    # distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
    # ANY KIND, either express or implied. See the License for the specific
    # language governing permissions and limitations under the License.
    
    # For more information on creating a Dockerfile
    # https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
    # https://github.com/awslabs/amazon-sagemaker-examples/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb
    # SageMaker PyTorch image
    FROM 520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch:0.4.0-cpu-py3
    
    ENV PATH="/opt/ml/code:${PATH}"
    
    # /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
    COPY /cifar10 /opt/ml/code
    
    # this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
    ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
    
    # this environment variable is used by the SageMaker PyTorch container to determine our program entry point
    # for training and serving.
    # For more information: https://github.com/aws/sagemaker-pytorch-container
    ENV SAGEMAKER_PROGRAM cifar10.py

Building and registering the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following shell code shows how to build the container image using
``docker build`` and push the container image to ECR using
``docker push``. This code is also available as the shell script
``container/build-and-push.sh``, which you can run as
``build-and-push.sh pytorch-extending-our-containers-cifar10-example``
to build the image ``pytorch-extending-our-containers-cifar10-example``.

This code looks for an ECR repository in the account you’re using and
the current default region (if you’re using a SageMaker notebook
instance, this is the region where the notebook instance was created).
If the repository doesn’t exist, the script will create it. In addition,
since we are using the SageMaker PyTorch image as the base, we will need
to retrieve ECR credentials to pull this public image.

.. code:: sh

    %%sh
    
    # The name of our algorithm
    algorithm_name=pytorch-extending-our-containers-cifar10-example
    
    cd container
    
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
    
    # Get the login command from ECR in order to pull down the SageMaker PyTorch image
    $(aws ecr get-login --registry-ids 520713654638 --region ${region} --no-include-email)
    
    # Build the docker image locally with the image name and then push it to ECR
    # with the full name.
    
    docker build  -t ${algorithm_name} . --build-arg REGION=${region}
    docker tag ${algorithm_name} ${fullname}
    
    docker push ${fullname}

Testing your algorithm on your local machine
--------------------------------------------

When you’re packaging your first algorithm to use with Amazon SageMaker,
you probably want to test it yourself to make sure it’s working
correctly. We use the `SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__ to test both
locally and on SageMaker. For more examples with the SageMaker Python
SDK, see `Amazon SageMaker
Examples <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk>`__.
In order to test our algorithm, we need our dataset.

Download the CIFAR-10 dataset
-----------------------------

We will be utilizing the CIFAR10 dataset loader provided within PyTorch
to download and load our data for training.

.. code:: ipython3

    from utils.utils_cifar import get_train_data_loader, get_test_data_loader, imshow, classes
    
    trainloader = get_train_data_loader('/tmp/pytorch-example/cifar-10-data')
    testloader = get_test_data_loader('/tmp/pytorch-example/cifar-10-data')

.. code:: ipython3

    # There should be the original tar file along with the extracted data directory. (cifar-10-python.tar.gz, cifar-10-batches-py)
    ! ls /tmp/pytorch-example/cifar-10-data

SageMaker Python SDK Local Training
-----------------------------------

To represent our training, we use the Estimator class, which needs to be
configured in five steps. 1. IAM role - our AWS execution role 2.
train_instance_count - number of instances to use for training. 3.
train_instance_type - type of instance to use for training. For training
locally, we specify ``local`` or ``local_gpu``. 4. image_name - our
custom PyTorch Docker image we created. 5. hyperparameters -
hyperparameters we want to pass.

Let’s start with setting up our IAM role. We make use of a helper
function within the Python SDK. This function throw an exception if run
outside of a SageMaker notebook instance, as it gets metadata from the
notebook instance. If running outside, you must provide an IAM role with
proper access stated above in `Permissions <#Permissions>`__.

.. code:: ipython3

    from sagemaker import get_execution_role
    
    role = get_execution_role()

Fit, Deploy, Predict
--------------------

Now that the rest of our estimator is configured, we can call ``fit()``
with the path to our local CIFAR10 dataset prefixed with ``file://``.
This invokes our PyTorch container with ‘train’ and passes in our
hyperparameters and other metadata as json files in /opt/ml/input/config
within the container to our program entry point defined in the
Dockerfile.

After our training has succeeded, our training algorithm outputs our
trained model within the /opt/ml/model directory, which is used to
handle predictions.

We can then call ``deploy()`` with an instance_count and instance_type,
which is 1 and ``local``. This invokes our PyTorch container with
‘serve’, which setups our container to handle prediction requests as
defined
`here <https://github.com/aws/sagemaker-pytorch-container/blob/master/src/sagemaker_pytorch_container/serving.py#L103>`__.
What is returned is a predictor, which is used to make inferences
against our trained model.

After our prediction, we can delete our endpoint.

We recommend testing and training your training algorithm locally first,
as it provides quicker iterations and better debuggability.

.. code:: ipython3

    # Lets set up our SageMaker notebook instance for local mode.
    !/bin/bash ./utils/setup.sh

.. code:: ipython3

    import os
    import subprocess
    
    instance_type = 'local'
    
    if subprocess.call('nvidia-smi') == 0:
        ## Set type to GPU if one is present
        instance_type = 'local_gpu'
        
    print("Instance type = " + instance_type)

.. code:: ipython3

    from sagemaker.estimator import Estimator
    
    hyperparameters = {'epochs': 1}
    
    estimator = Estimator(role=role,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          image_name='pytorch-extending-our-containers-cifar10-example:latest',
                          hyperparameters=hyperparameters)
    
    estimator.fit('file:///tmp/pytorch-example/cifar-10-data')
    
    predictor = estimator.deploy(1, instance_type)

Making predictions using Python SDK
-----------------------------------

To make predictions, we will use a few images, from the test loader,
converted into a json format to send as an inference request.

The reponse will be tensors containing the probabilities of each image
belonging to one of the 10 classes. Based on the highest probability we
will map that index to the corresponding class in our output. The
classes can be referenced from the `CIFAR-10
website <https://www.cs.toronto.edu/~kriz/cifar.html>`__. Since we
didn’t train the model for that long, we aren’t expecting very accurate
results.

.. code:: ipython3

    import torchvision, torch
    import numpy as np
    
    from sagemaker.predictor import json_serializer, json_deserializer
    
    # get some test images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%4s' % classes[labels[j]] for j in range(4)))
    
    predictor.accept = 'application/json'
    predictor.content_type = 'application/json'
    
    predictor.serializer = json_serializer
    predictor.deserializer = json_deserializer
    
    outputs = predictor.predict(images.numpy())
    
    _, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)
    
    print('Predicted: ', ' '.join('%4s' % classes[predicted[j]]
                                  for j in range(4)))

.. code:: ipython3

    predictor.delete_endpoint()

Part 2: Training and Hosting your Algorithm in Amazon SageMaker
===============================================================

Once you have your container packaged, you can use it to train and serve
models. Let’s do that with the algorithm we made above.

Set up the environment
----------------------

Here we specify the bucket to use and the role that is used for working
with SageMaker.

.. code:: ipython3

    # S3 prefix
    prefix = 'DEMO-pytorch-cifar10'

Create the session
------------------

The session remembers our connection parameters to SageMaker. We use it
to perform all of our SageMaker operations.

.. code:: ipython3

    import sagemaker as sage
    
    sess = sage.Session()

Upload the data for training
----------------------------

We will use the tools provided by the SageMaker Python SDK to upload the
data to a default bucket.

.. code:: ipython3

    WORK_DIRECTORY = '/tmp/pytorch-example/cifar-10-data'
    
    data_location = sess.upload_data(WORK_DIRECTORY, key_prefix=prefix)

Training on SageMaker
---------------------

Training a model on SageMaker with the Python SDK is done in a way that
is similar to the way we trained it locally. This is done by changing
our train_instance_type from ``local`` to one of our `supported EC2
instance
types <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__.

In addition, we must now specify the ECR image URL, which we just pushed
above.

Finally, our local training dataset has to be in Amazon S3 and the S3
URL to our dataset is passed into the ``fit()`` call.

Let’s first fetch our ECR image url that corresponds to the image we
just built and pushed.

.. code:: ipython3

    import boto3
    
    client = boto3.client('sts')
    account = client.get_caller_identity()['Account']
    
    my_session = boto3.session.Session()
    region = my_session.region_name
    
    algorithm_name = 'pytorch-extending-our-containers-cifar10-example'
    
    ecr_image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)
    
    print(ecr_image)

.. code:: ipython3

    from sagemaker.estimator import Estimator
    
    hyperparameters = {'epochs': 1}
    
    instance_type = 'ml.m4.xlarge'
    
    estimator = Estimator(role=role,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          image_name=ecr_image,
                          hyperparameters=hyperparameters)
    
    estimator.fit(data_location)
    
    predictor = estimator.deploy(1, instance_type)

.. code:: ipython3

    # get some test images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%4s' % classes[labels[j]] for j in range(4)))
    
    predictor.accept = 'application/json'
    predictor.content_type = 'application/json'
    
    predictor.serializer = json_serializer
    predictor.deserializer = json_deserializer
    
    outputs = predictor.predict(images.numpy())
    
    _, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)
    
    print('Predicted: ', ' '.join('%4s' % classes[predicted[j]]
                                  for j in range(4)))

Optional cleanup
----------------

When you’re done with the endpoint, you should clean it up.

All of the training jobs, models and endpoints we created can be viewed
through the SageMaker console of your AWS account.

.. code:: ipython3

    predictor.delete_endpoint()

Reference
=========

-  `How Amazon SageMaker interacts with your Docker container for
   training <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html>`__
-  `How Amazon SageMaker interacts with your Docker container for
   inference <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html>`__
-  `CIFAR-10 Dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`__
-  `SageMaker Python
   SDK <https://github.com/aws/sagemaker-python-sdk>`__
-  `Dockerfile <https://docs.docker.com/engine/reference/builder/>`__
-  `scikit-bring-your-own <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb>`__
-  `SageMaker PyTorch
   container <https://github.com/aws/sagemaker-pytorch-container>`__
