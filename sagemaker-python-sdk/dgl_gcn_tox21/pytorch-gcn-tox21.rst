Training Amazon SageMaker models for molecular property prediction by using DGL with PyTorch backend
====================================================================================================

The **Amazon SageMaker Python SDK** makes it easy to train Deep Graph
Library (DGL) models. In this example, you train a simple graph neural
network for molecular toxicity prediction by using
`DGL <https://github.com/dmlc/dgl>`__ and the Tox21 dataset.

The dataset contains qualitative toxicity measurements for 8,014
compounds on 12 different targets, including nuclear receptors and
stress-response pathways. Each target yields a binary classification
problem. You can model the problem as a graph classification problem.

Setup
-----

Define a few variables that you need later in the example.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # Setup session
    sess = sagemaker.Session()
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = sess.default_bucket()
    
    # Location to put your custom code.
    custom_code_upload_location = 'customcode'
    
    # IAM execution role that gives Amazon SageMaker access to resources in your AWS account.
    # You can use the Amazon SageMaker Python SDK to get the role from the notebook environment. 
    role = get_execution_role()

Training Script
---------------

``main.py`` provides all the code you need for training a molecular
property prediction model by using Amazon SageMaker.

.. code:: ipython3

    !cat main.py

Bring Your Own Image for Amazon SageMaker
-----------------------------------------

In this example, you need rdkit library to handle the tox21 dataset. The
DGL CPU and GPU Docker has the rdkit library pre-installed at Dockerhub
under dgllib registry (namely,
dgllib/dgl-sagemaker-cpu:dgl_0.4_pytorch_1.2.0_rdkit for CPU and
dgllib/dgl-sagemaker-gpu:dgl_0.4_pytorch_1.2.0_rdkit for GPU). You can
pull the image yourself according to your requirement and push it into
your AWS ECR. Following script helps you to do so. You can skip this
step if you have already prepared your DGL Docker image in your Amazon
Elastic Container Registry (Amazon ECR).

.. code:: sh

    %%sh
    # For CPU default_docker_name="dgllib/dgl-sagemaker-cpu:dgl_0.4_pytorch_1.2.0_rdkit"
    default_docker_name="dgllib/dgl-sagemaker-gpu:dgl_0.4_pytorch_1.2.0_rdkit"
    docker pull $default_docker_name
    
    docker_name=sagemaker-dgl-pytorch-gcn-tox21
    
    # For CPU docker build -t $docker_name -f gcn_tox21_cpu.Dockerfile .
    docker build -t $docker_name -f gcn_tox21_gpu.Dockerfile .
    
    account=$(aws sts get-caller-identity --query Account --output text)
    echo $account
    region=$(aws configure get region)
    
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${docker_name}:latest"
    # If the repository doesn't exist in ECR, create it.
    aws ecr describe-repositories --repository-names "${docker_name}" > /dev/null 2>&1
    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "${docker_name}" > /dev/null
    fi
    
    # Get the login command from ECR and execute it directly
    $(aws ecr get-login --region ${region} --no-include-email)
    
    docker tag ${docker_name} ${fullname}
    docker push ${fullname}

The Amazon SageMaker Estimator class
------------------------------------

The Amazon SageMaker Estimator allows you to run a single machine in
Amazon SageMaker, using CPU or GPU-based instances.

When you create the estimator, pass in the file name of the training
script and the name of the IAM execution role. Also provide a few other
parameters. ``train_instance_count`` and ``train_instance_type``
determine the number and type of SageMaker instances that will be used
for the training job. The hyperparameters can be passed to the training
script via a dict of values. See ``main.py`` for how they are handled.

The entrypoint of Amazon SageMaker Docker (e.g.,
dgllib/dgl-sagemaker-gpu:dgl_0.4_pytorch_1.2.0_rdkit) is a train script
under /usr/bin/. The train script inside dgl docker image provided above
will try to get the real entrypoint from the hyperparameters (with the
key ‘entrypoint’) and run the real entrypoint under ‘training-code’ data
channel (/opt/ml/input/data/training-code/) .

For this example, choose one ml.p3.2xlarge instance. You can also use a
CPU instance such as ml.c4.2xlarge for the CPU image. You can also add a
task_tag with value ‘DGL’ to help tracking the task.

.. code:: ipython3

    import boto3
    
    # Set target dgl-docker name
    docker_name='sagemaker-dgl-pytorch-gcn-tox21'
    
    CODE_PATH = 'main.py'
    code_location = sess.upload_data(CODE_PATH, bucket=bucket, key_prefix=custom_code_upload_location)
    
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, docker_name)
    print(image)
    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    estimator = sagemaker.estimator.Estimator(image,
                            role, 
                            train_instance_count=1, 
                            train_instance_type= 'ml.p3.2xlarge', #'ml.c4.2xlarge'
                            hyperparameters={'entrypoint': CODE_PATH},
                            tags=task_tags,
                            sagemaker_session=sess)

Running the Training Job
------------------------

After you construct an Estimator object, fit it by using Amazon
SageMaker.

.. code:: ipython3

    estimator.fit({'training-code': code_location})

Output
------

You can get the model training output from the Amazon Sagemaker console
by searching for the training task and looking for the address of ‘S3
model artifact’
