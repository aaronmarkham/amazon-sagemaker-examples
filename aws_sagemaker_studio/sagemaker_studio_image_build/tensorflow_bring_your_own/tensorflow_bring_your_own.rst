Building your own TensorFlow container from Amazon SageMaker Studio
===================================================================

**STUDIO KERNEL NOTE:** If you are prompted for Kernel, choose ’Python 3
(TensorFlow CPU Optimized)

With Amazon SageMaker, you can package your own algorithms that can then
be trained and deployed in the SageMaker environment. This notebook
guides you through an example using TensorFlow that shows you how to
build a Docker container for SageMaker and use it for training and
inference.

This notebook contains a modified version of the existing `Tensorflow
Bring Your
Own <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/tensorflow_bring_your_own>`__
notebook created to run on Amazon SageMaker Notebook Instances. Because
the underlying architecture between Amazon SageMaker Notebook Instances
and Amazon SageMaker Studio Notebooks is different, this notebook is
created specifically to illustrate a bring-your-own scenario within
Amazon SageMaker Studio using the `SageMaker Studio Image Build
CLI <https://github.com/aws-samples/sagemaker-studio-image-build-cli/blob/master/README.md>`__

By packaging an algorithm in a container, you can bring almost any code
to the Amazon SageMaker environment, regardless of programming language,
environment, framework, or dependencies.

1.  `Building your own TensorFlow
    container <#Building-your-own-tensorflow-container>`__
2.  `When should I build my own algorithm
    container? <#When-should-I-build-my-own-algorithm-container?>`__
3.  `Permissions <#Permissions>`__
4.  `The example <#The-example>`__
5.  `The workflow <#The-workflow>`__
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
    7. `Building and registering the container using the
       sagemaker-docker CLI <#Building-and-registering-the-container>`__

7.  `Testing your algorithm on your local
    machine <#Testing-your-algorithm-on-your-local-machine>`__
8.  `Part 2: Training and Hosting your Algorithm in Amazon
    SageMaker <#Part-2:-Training-and-Hosting-your-Algorithm-in-Amazon-SageMaker>`__
9.  `Set up the environment <#Set-up-the-environment>`__
10. `Create the session <#Create-the-session>`__
11. `Upload the data for training <#Upload-the-data-for-training>`__
12. `Training On SageMaker <#Training-on-SageMaker>`__
13. `Optional cleanup <#Optional-cleanup>`__
14. `Reference <#Reference>`__

*or* I’m impatient, just `let me see the code <#The-Dockerfile>`__!

When should I build my own algorithm container?
-----------------------------------------------

You may not need to create a container to bring your own code to Amazon
SageMaker. When you are using a framework such as Apache MXNet or
TensorFlow that has direct support in SageMaker, you can simply supply
the Python code that implements your algorithm using the SDK entry
points for that framework. This set of supported frameworks is regularly
added to, so you should check the current list to determine whether your
algorithm is written in one of these common machine learning
environments.

Even if there is direct SDK support for your environment or framework,
you may find it more effective to build your own container. If the code
that implements your algorithm is quite complex or you need special
additions to the framework, building your own container may be the right
choice.

Some of the reasons to build an already supported framework container
are: 1. A specific version isn’t supported. 2. Configure and install
your dependencies and environment. 3. Use a different training/hosting
solution than provided.

This walkthrough shows that it is quite straightforward to build your
own container. So you can still use SageMaker even if your use case is
not covered by the deep learning containers that we’ve built for you.

Permissions
-----------

Running this notebook requires permissions in addition to the normal
``SageMakerFullAccess`` execution role permissions. This is because it:

1. Creates a new repository and pushes built images to `Amazon Elastic
   Container Registry <https://aws.amazon.com/ecr/>`__
2. Utilizes `AWS Code Build <https://aws.amazon.com/codebuild/>`__ to
   build new docker images

The example
-----------

In this example we show how to package a custom TensorFlow container
with Amazon SageMaker studio with a Python example which works with the
CIFAR-10 dataset and uses TensorFlow Serving for inference. However,
different inference solutions other than TensorFlow Serving can be used
by modifying the docker container.

In this example, we use a single image to support training and hosting.
This simplifies the procedure because we only need to manage one image
for both tasks. Sometimes you may want separate images for training and
hosting because they have different requirements. In this case, separate
the parts discussed below into separate Dockerfiles and build two
images. Choosing whether to use a single image or two images is a matter
of what is most convenient for you to develop and manage.

If you’re only using Amazon SageMaker for training or hosting, but not
both, only the functionality used needs to be built into your container.

The workflow
------------

This notebook is divided into two parts: *building* the container and
*using* the container.

Part 1: Packaging and Uploading your Algorithm for use with Amazon SageMaker
============================================================================

An overview of Docker
~~~~~~~~~~~~~~~~~~~~~

If you’re familiar with Docker already, you can skip ahead to the next
section.

For many data scientists, Docker containers are a new technology. But
they are not difficult and can significantly simply the deployment of
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
and environment variables.

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
How your container processes this argument depends on the container.

-  In this example, we don’t define an ``ENTRYPOINT`` in the Dockerfile
   so Docker runs the command ```train`` at training
   time <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html>`__
   and ```serve`` at serving
   time <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html>`__.
   In this example, we define these as executable Python scripts, but
   they could be any program that we want to start in that environment.
-  If you specify a program as an ``ENTRYPOINT`` in the Dockerfile, that
   program will be run at startup and its first argument will be
   ``train`` or ``serve``. The program can then look at that argument
   and decide what to do.
-  If you are building separate containers for training and hosting (or
   building only for one or the other), you can define a program as an
   ``ENTRYPOINT`` in the Dockerfile and ignore (or verify) the first
   argument passed in.

Running your container during training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When Amazon SageMaker runs training, your ``train`` script is run, as in
a regular Python program. A number of files are laid out for your use,
under the ``/opt/ml`` directory:

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
reponding to inference requests that come in via HTTP. In this example,
we use `TensorFlow Serving <https://www.tensorflow.org/serving/>`__,
however the hosting solution can be customized. One example is the
`Python serving stack within the scikit learn
example <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb>`__.

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

This directory has all the components you need to package the sample
algorithm for Amazon SageMager:

::

   .
   |-- Dockerfile
   `-- cifar10
       |-- cifar10.py
       |-- resnet_model.py
       |-- nginx.conf
       |-- serve
       `-- train

Let’s discuss each of these in turn:

-  **``Dockerfile``** describes how to build your Docker container
   image. More details are provided below.
-  **``cifar10``** is the directory which contains the files that are
   installed in the container.

In this simple application, we install only five files in the container.
You may only need that many, but if you have many supporting routines,
you may wish to install more. These five files show the standard
structure of our Python containers, although you are free to choose a
different toolset and therefore could have a different layout. If you’re
writing in a different programming language, you will have a different
layout depending on the frameworks and tools you choose.

The files that we put in the container are:

-  **``cifar10.py``** is the program that implements our training
   algorithm.
-  **``resnet_model.py``** is the program that contains our Resnet
   model.
-  **``nginx.conf``** is the configuration file for the nginx front-end.
   Generally, you should be able to take this file as-is.
-  **``serve``** is the program started when the container is started
   for hosting. It simply launches nginx and loads your exported model
   with TensorFlow Serving.
-  **``train``** is the program that is invoked when the container is
   run for training. Our implementation of this script invokes
   cifar10.py with our our hyperparameter values retrieved from
   /opt/ml/input/config/hyperparameters.json. The goal for doing this is
   to avoid having to modify our training algorithm program.

In summary, the two files you probably want to change for your
application are ``train`` and ``serve``.

The Dockerfile
~~~~~~~~~~~~~~

The Dockerfile describes the image that we want to build. You can think
of it as describing the complete operating system installation of the
system that you want to run. A Docker container running is quite a bit
lighter than a full operating system, however, because it takes
advantage of Linux on the host machine for the basic operations.

For the Python science stack, we start from an official TensorFlow
docker image and run the normal tools to install TensorFlow Serving.
Then we add the code that implements our specific algorithm to the
container and set up the right environment for it to run under.

Let’s look at the `Dockerfile <./container/Dockerfile>`__ for this
example.

Building and registering the container using the SageMaker Studio Image Build CLI
---------------------------------------------------------------------------------

There are two ways to build and push docker images to ECR from within an
Amazon SageMaker Studio Notebook.

1. **Setup Your Own Integrations** Build the necessary integrations and
   workflow into your Studio environment that allow you to use a build
   service such as AWS Code Build to build your docker images as well as
   setup your ECR repository and pushes image to that respository.

2. **Utilize the SageMaker Studio Image Build CLI convenience package**
   This is the preferred approach as it removes the heavy lift of
   setting up your own workflows and docker build capabilities. The CLI
   provides an abstraction of those underlying integrations and
   workflows allowing you to easily build and push docker images using
   simple CLI commands.

Using the SageMaker Studio Image Build CLI Convenience Package
--------------------------------------------------------------

There are just a few steps to get started using the new convenience
package.

Step 1: Install the CLI
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install sagemaker_studio_image_build

Step 2: Ensure IAM Role has access to necessary services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker Studio Image Build CLI uses Amazon Elastic Container
Registry and AWS CodeBuild so we need to ensure that the role we provide
as input to our CLI commands has the necessary policies and permissions
attached.

Two scenarios are supported including:

1. **Add IAM Permissions to SageMaker Execution Role**

This scenario includes updating the Execution Role attached to this
notebook instance with the required permissions. In this scenario, you
need to get the current execution role and ensure the trust policy and
additional permissions are associated with the role.

2. **Create/Utilize a secondary role with appropriate permissions
   attached**

This scenario includes using a secondary role setup with the permissions
below and identified in the –role argument when invoking the CLI
(Example: *sm-docker build . –role build-cli-role*)

For this example, we are going to **Add IAM Permissions to the current
SageMaker Execution Role**.

Let’s first grab the current execution role…

.. code:: ipython3

    import sagemaker
    import boto3
    
    try:
        role = sagemaker.get_execution_role()
    except:
        role = get_execution_role()
    
    print("Using IAM role arn: {}".format(role))

Now we need to add the permissions below for the role identified above.

| **Update Trust Policy for CodeBuild** \* Open
  `IAM <https://console.aws.amazon.com/iam/home#/roles>`__ and search
  for the role listed above. \* Select the Role and click on the **Trust
  relationships** tab.
| \* Update the trust relationship using the JSON to establish a trust
  relationship with CodeBuild

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    },
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "codebuild.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}

-  Once you’ve added the trust relationship above, click **Update Trust
   Policy**

We also need to add some additional permissions to the execution role to
be able to build the image with CodeBuild and push the image to ECR. You
can update the existing execution policy attached to the role or create
a new policy and attach it to the existing execution role. Whichever
option you choose, ensure the policy has the correct permissions set for
intended S3 bucket access. The sample policy in the `CLI
README <https://github.com/aws-samples/sagemaker-studio-image-build-cli/tree/b0b8d337dba4f1ecc88f33f81e815fb44c4c9915>`__
assumes access to the default session bucket so this may need to be
modified for your use casee. For this example, we are going to create a
new policy and attach it to the existing role.

**Create policy allowing access to supporting services**

-  Open `Policies <https://console.aws.amazon.com/iam/home#/policies>`__
   in IAM
-  Click **Create policy**
-  Select the JSON tab and copy/paste the policy below

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "codebuild:DeleteProject",
                "codebuild:CreateProject",
                "codebuild:BatchGetBuilds",
                "codebuild:StartBuild"
            ],
            "Resource": "arn:aws:codebuild:*:*:project/sagemaker-studio*"
        },
        {
            "Effect": "Allow",
            "Action": "logs:CreateLogStream",
            "Resource": "arn:aws:logs:*:*:log-group:/aws/codebuild/sagemaker-studio*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:GetLogEvents",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:log-group:/aws/codebuild/sagemaker-studio*:log-stream:*"
        },
        {
            "Effect": "Allow",
            "Action": "logs:CreateLogGroup",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:CreateRepository",
                "ecr:BatchGetImage",
                "ecr:CompleteLayerUpload",
                "ecr:DescribeImages",
                "ecr:DescribeRepositories",
                "ecr:UploadLayerPart",
                "ecr:ListImages",
                "ecr:InitiateLayerUpload",
                "ecr:BatchCheckLayerAvailability",
                "ecr:PutImage"
            ],
            "Resource": "arn:aws:ecr:*:*:repository/sagemaker-studio*"
        },
        {
            "Effect": "Allow",
            "Action": "ecr:GetAuthorizationToken",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
              "s3:GetObject",
              "s3:DeleteObject",
              "s3:PutObject"
              ],
            "Resource": "arn:aws:s3:::sagemaker-*/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket"
            ],
            "Resource": "arn:aws:s3:::sagemaker*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:GetRole",
                "iam:ListRoles"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::*:role/*",
            "Condition": {
                "StringLikeIfExists": {
                    "iam:PassedToService": "codebuild.amazonaws.com"
                }
            }
        }
    ]
}

-  Click **Review policy**
-  Give the policy a name such as ``Studio-Image-Build-Policy``
-  Click **Create policy**

We now need to attach our policy to the Execution Role attached to this
notebook environment.

-  Go back to `Roles <https://console.aws.amazon.com/iam/home#/roles>`__
   in IAM
-  Select the SageMaker Execution Role from abovee
-  On the **Permissions** tab, click **Attach policies**
-  Search for the Policy we created above ``Studio-Image-Build-Policy``
-  Select the policy and click **Attach policy**

Step 3: Building and registering the container using the SageMaker Studio Image Build CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will now create our training container image, using the SageMaker
Studio Image Build CLI

To do this we need to navigate to the directory containing our
Dockerfile and simply execute the build command:

::

                 sm-docker build .
                 

The build command can optionally take additional arguments depending on
your needs:

::

                 sm-docker build . --file /path/to/Dockerfile --build-arg foo=bar
                 

**TIP** If you receive a permissions error below, please ensure you have
completed **both** permission setup items above: (1) Update Trust Policy
(2) Create new policy & Attach it to the existing SageMaker Execution
Role

.. code:: ipython3

    !sm-docker build .

**NOTE** The Image URI output above will be used as the input training
image for our training job

--------------

Download the CIFAR-10 dataset
-----------------------------

Our training algorithm is expecting our training data to be in the file
format of `TFRecords <https://www.tensorflow.org/guide/datasets>`__,
which is a simple record-oriented binary format that many TensorFlow
applications use for training data.

Below is a Python script adapted from the `official TensorFlow CIFAR-10
example <https://github.com/tensorflow/models/blob/451906e4e82f19712455066c1b27e2a6ba71b1dd/research/slim/datasets/download_and_convert_cifar10.py>`__,
which downloads the CIFAR-10 dataset and converts them into TFRecords.

The adapted script has a dependency on ipywidgets so we will first need
to install that dependencies in our notebook prior to executing the
script.

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install ipywidgets

.. code:: ipython3

    ! python utils/generate_cifar10_tfrecords.py --data-dir=/tmp/cifar-10-data

.. code:: ipython3

    # There should be three tfrecords. (eval, train, validation)
    ! ls /tmp/cifar-10-data

Part 2: Training and Hosting your Algorithm in Amazon SageMaker
===============================================================

Once you have your container packaged, you can use it to train and serve
models. Let’s do that with the algorithm we made above.

Set up the environment
----------------------

Here we specify the bucket to use

.. code:: ipython3

    # S3 prefix
    prefix = 'DEMO-tensorflow-cifar10'

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

    WORK_DIRECTORY = '/tmp/cifar-10-data'
    
    data_location = sess.upload_data(WORK_DIRECTORY, key_prefix=prefix)

Train & Deploy on SageMaker
---------------------------

Next, we will perform our training using SageMaker Training Instances.
Because are bringing our own training image, we need to specify our ECR
image URL. This is the Image URI that was output from our SageMaker
Studio Image Build CLI that we executed above. Make sure you update the
ECR Image value with that output value as indicated below.

Finally, our local training dataset has to be in Amazon S3 and the S3
URL to our dataset is passed into the ``fit()`` call. After our model is
trained, we will then use the ``deploy()`` call to deploy our model to a
persistent endpoint using SageMaker Hosting.

Let’s first fetch our ECR image url that corresponds to the image we
just built and pushed.

.. code:: ipython3

    import boto3
    
    sm = boto3.client('sagemaker')
    ecr = boto3.client('ecr')
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = boto3.session.Session().region_name
    
    domain_id = 'sagemaker-studio-{}'.format(sm.list_apps()['Apps'][0]['DomainId'])
    image_tag = ecr.list_images(repositoryName=domain_id, filter={'tagStatus':'TAGGED'})['imageIds'][0]['imageTag']
    ecr_image = '{}.dkr.ecr.{}.amazonaws.com/{}:{}'.format(account, region, domain_id, image_tag)
    
    print(ecr_image)

Train our model using SageMaker Training Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from sagemaker.estimator import Estimator
    
    hyperparameters = {'train-steps': 100}
    
    instance_type = 'ml.m4.xlarge'
    
    estimator = Estimator(role=role,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          image_name=ecr_image,
                          hyperparameters=hyperparameters)
    
    estimator.fit(data_location)

Host our model using SageMaker Hosting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    predictor = estimator.deploy(1, instance_type)

Test Endpoint - Making predictions using Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make predictions, we use an image that is converted using Imageio
into a json format to send as an inference request. We need to install
Imageio to deserialize the image that is used to make predictions.

The JSON reponse will be the probabilities of the image belonging to one
of the 10 classes along with the most likely class the picture belongs
to. The classes can be referenced from the `CIFAR-10
website <https://www.cs.toronto.edu/~kriz/cifar.html>`__.

**NOTE**: Since we didn’t train the model for that long, we aren’t
expecting very accurate results. To improve results, consider
experimennting with additional training optimizations.

**Import Imageio**

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install imageio

**View our sample image**

.. code:: ipython3

    import os
    
    from IPython.display import Image, display
    
    images = []
    for entry in os.scandir('data'):
        if entry.is_file() and entry.name.endswith("png"):
            images.append('data/' + entry.name)
    
    for image in images:
        display(Image(image))

**Format Image for prediction**

.. code:: ipython3

    import imageio as imageio
    import numpy
    
    from sagemaker.predictor import json_serializer, json_deserializer
    
    image = imageio.imread("data/cat.png")
    print(image.shape)
    
    data = {'instances': numpy.asarray(image).astype(float).tolist()}

**Send image to endpoint for prediction**

.. code:: ipython3

    # The request and response format is JSON for TensorFlow Serving.
    # For more information: https://www.tensorflow.org/serving/api_rest#predict_api
    predictor.accept = 'application/json'
    predictor.content_type = 'application/json'
    
    predictor.serializer = json_serializer
    predictor.deserializer = json_deserializer
    
    # For more information on the predictor class.
    # https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/predictor.py
    predictor.predict(data)

As we mentioned above, we don’t expect our model to perform well as we
did not train it for very long. You can increase you experiements
through additional training cycles to continue to improve your model.

Optional cleanup
----------------

When you’re done with the endpoint, you should clean it up.

All of the training jobs, models and endpoints we created can be viewed
through the SageMaker console of your AWS account.

.. code:: ipython3

    predictor.delete_endpoint()

Reference
=========

-  `SageMaker Studio Image Build
   CLI <https://github.com/aws-samples/sagemaker-studio-image-build-cli/README.md>`__
-  `How Amazon SageMaker interacts with your Docker container for
   training <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html>`__
-  `How Amazon SageMaker interacts with your Docker container for
   inference <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html>`__
-  `CIFAR-10 Dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`__
-  `SageMaker Python
   SDK <https://github.com/aws/sagemaker-python-sdk>`__
-  `Dockerfile <https://docs.docker.com/engine/reference/builder/>`__
