.. raw:: html

   <h1>

Script-mode Custom Training Container (2)

.. raw:: html

   </h1>

This notebook demonstrates how to build and use a custom Docker
container for training with Amazon SageMaker that leverages on the
Script Mode execution that is implemented by the
sagemaker-training-toolkit library. Reference documentation is available
at https://github.com/aws/sagemaker-training-toolkit.

The difference from the first example is that we are not copying the
training code during the Docker build process, and we are loading them
dynamically from Amazon S3 (this feature is implemented through the
sagemaker-training-toolkit).

We start by defining some variables like the current execution role, the
ECR repository that we are going to use for pushing the custom Docker
container and a default Amazon S3 bucket to be used by Amazon SageMaker.

.. code:: ipython3

    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    
    ecr_namespace = 'sagemaker-training-containers/'
    prefix = 'script-mode-container-2'
    
    ecr_repository_name = ecr_namespace + prefix
    role = get_execution_role()
    account_id = role.split(':')[4]
    region = boto3.Session().region_name
    sagemaker_session = sagemaker.session.Session()
    bucket = sagemaker_session.default_bucket()
    
    print(account_id)
    print(region)
    print(role)
    print(bucket)

Let’s take a look at the Dockerfile which defines the statements for
building our script-mode custom training container:

.. code:: ipython3

    ! pygmentize ../docker/Dockerfile


.. parsed-literal::

    [37m# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.[39;49;00m
    [37m# https://github.com/aws/sagemaker-mxnet-container[39;49;00m
    
    [34mFROM[39;49;00m[33m ubuntu:16.04[39;49;00m
    
    LABEL [31mmaintainer[39;49;00m=[33m"Giuseppe A. Porcelli"[39;49;00m
    
    [37m# Defining some variables used at build time to install Python3[39;49;00m
    ARG [31mPYTHON[39;49;00m=python3
    ARG [31mPYTHON_PIP[39;49;00m=python3-pip
    ARG [31mPIP[39;49;00m=pip3
    ARG [31mPYTHON_VERSION[39;49;00m=[34m3[39;49;00m.6.6
    
    [37m# Install some handful libraries like curl, wget, git, build-essential, zlib[39;49;00m
    [34mRUN[39;49;00m apt-get update && apt-get install -y --no-install-recommends software-properties-common && [33m\[39;49;00m
        add-apt-repository ppa:deadsnakes/ppa -y && [33m\[39;49;00m
        apt-get update && apt-get install -y --no-install-recommends [33m\[39;49;00m
            build-essential [33m\[39;49;00m
            ca-certificates [33m\[39;49;00m
            curl [33m\[39;49;00m
            wget [33m\[39;49;00m
            git [33m\[39;49;00m
            libopencv-dev [33m\[39;49;00m
            openssh-client [33m\[39;49;00m
            openssh-server [33m\[39;49;00m
            vim [33m\[39;49;00m
            zlib1g-dev && [33m\[39;49;00m
        rm -rf /var/lib/apt/lists/*
    
    [37m# Installing Python3[39;49;00m
    [34mRUN[39;49;00m wget https://www.python.org/ftp/python/[31m$PYTHON_VERSION[39;49;00m/Python-[31m$PYTHON_VERSION[39;49;00m.tgz && [33m\[39;49;00m
            tar -xvf Python-[31m$PYTHON_VERSION[39;49;00m.tgz && [36mcd[39;49;00m Python-[31m$PYTHON_VERSION[39;49;00m && [33m\[39;49;00m
            ./configure && make && make install && [33m\[39;49;00m
            apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && [33m\[39;49;00m
            make && make install && rm -rf ../Python-[31m$PYTHON_VERSION[39;49;00m* && [33m\[39;49;00m
            ln -s /usr/local/bin/pip3 /usr/bin/pip
    
    [37m# Upgrading pip and creating symbolic link for python3[39;49;00m
    [34mRUN[39;49;00m [33m${[39;49;00m[31mPIP[39;49;00m[33m}[39;49;00m --no-cache-dir install --upgrade pip
    [34mRUN[39;49;00m ln -s [34m$([39;49;00mwhich [33m${[39;49;00m[31mPYTHON[39;49;00m[33m}[39;49;00m[34m)[39;49;00m /usr/local/bin/python
    
    [34mWORKDIR[39;49;00m[33m /[39;49;00m
    
    [37m# Installing numpy, pandas, scikit-learn, scipy[39;49;00m
    [34mRUN[39;49;00m [33m${[39;49;00m[31mPIP[39;49;00m[33m}[39;49;00m install --no-cache --upgrade [33m\[39;49;00m
            [31mnumpy[39;49;00m==[34m1[39;49;00m.14.5 [33m\[39;49;00m
            [31mpandas[39;49;00m==[34m0[39;49;00m.24.1 [33m\[39;49;00m
            scikit-learn==[34m0[39;49;00m.20.3 [33m\[39;49;00m
            [31mrequests[39;49;00m==[34m2[39;49;00m.21.0 [33m\[39;49;00m
            [31mscipy[39;49;00m==[34m1[39;49;00m.2.2
    
    [37m# Setting some environment variables.[39;49;00m
    [34mENV[39;49;00m[33m PYTHONDONTWRITEBYTECODE=1 \[39;49;00m
        [31mPYTHONUNBUFFERED[39;49;00m=[34m1[39;49;00m [33m\[39;49;00m
        [31mLD_LIBRARY_PATH[39;49;00m=[33m"[39;49;00m[33m${[39;49;00m[31mLD_LIBRARY_PATH[39;49;00m[33m}[39;49;00m[33m:/usr/local/lib[39;49;00m[33m"[39;49;00m [33m\[39;49;00m
        [31mPYTHONIOENCODING[39;49;00m=UTF-8 [33m\[39;49;00m
        [31mLANG[39;49;00m=C.UTF-8 [33m\[39;49;00m
        [31mLC_ALL[39;49;00m=C.UTF-8
    
    [34mRUN[39;49;00m [33m${[39;49;00m[31mPIP[39;49;00m[33m}[39;49;00m install --no-cache --upgrade [33m\[39;49;00m
        sagemaker-training


At high-level the Dockerfile specifies the following operations for
building this container:

.. raw:: html

   <ul>

.. raw:: html

   <li>

Start from Ubuntu 16.04

.. raw:: html

   </li>

.. raw:: html

   <li>

Define some variables to be used at build time to install Python 3

.. raw:: html

   </li>

.. raw:: html

   <li>

Some handful libraries are installed with apt-get

.. raw:: html

   </li>

.. raw:: html

   <li>

We then install Python 3 and create a symbolic link

.. raw:: html

   </li>

.. raw:: html

   <li>

We install some Python libraries like numpy, pandas, ScikitLearn, etc.

.. raw:: html

   </li>

.. raw:: html

   <li>

We set e few environment variables, including PYTHONUNBUFFERED which is
used to avoid buffering Python standard output (useful for logging)

.. raw:: html

   </li>

.. raw:: html

   <li>

We install the sagemaker-training-toolkit library

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   <h3>

Build and push the container

.. raw:: html

   </h3>

We are now ready to build this container and push it to Amazon ECR. This
task is executed using a shell script stored in the ../script/ folder.
Let’s take a look at this script and then execute it.

.. code:: ipython3

    ! pygmentize ../scripts/build_and_push.sh

.. raw:: html

   <h3>

——————————————————————————————————————–

.. raw:: html

   </h3>

The script builds the Docker container, then creates the repository if
it does not exist, and finally pushes the container to the ECR
repository. The build task requires a few minutes to be executed the
first time, then Docker caches build outputs to be reused for the
subsequent build operations.

.. code:: ipython3

    %%capture
    ! ../scripts/build_and_push.sh $account_id $region $ecr_repository_name

.. raw:: html

   <h3>

Training with Amazon SageMaker

.. raw:: html

   </h3>

Once we have correctly pushed our container to Amazon ECR, we are ready
to start training with Amazon SageMaker, which requires the ECR path to
the Docker container used for training as parameter for starting a
training job.

.. code:: ipython3

    container_image_uri = '{0}.dkr.ecr.{1}.amazonaws.com/{2}:latest'.format(account_id, region, ecr_repository_name)
    print(container_image_uri)

Given the purpose of this example is explaining how to build custom
script-mode containers, we are not going to train a real model. The
script that will be executed does not define a specific training logic;
it just outputs the configurations injected by SageMaker and implements
a dummy training loop. Training data is also dummy. Let’s analyze the
script first:

.. code:: ipython3

    ! pygmentize source_dir/train.py

You can realize that the training code has been implemented as a
standard Python script, that will be invoked by the
sagemaker-training-toolkit library passing hyperparameters as arguments.
This way of invoking training script is indeed called Script Mode for
Amazon SageMaker containers.

Now, we upload some dummy data to Amazon S3, in order to define our
S3-based training channels.

.. code:: ipython3

    ! echo "val1, val2, val3" > dummy.csv
    print(sagemaker_session.upload_data('dummy.csv', bucket, prefix + '/train'))
    print(sagemaker_session.upload_data('dummy.csv', bucket, prefix + '/val'))
    ! rm dummy.csv

We want to dynamically run user-provided code loading it from Amazon S3,
so we need to:

.. raw:: html

   <ul>

.. raw:: html

   <li>

Package the source_dir folder in a tar.gz archive

.. raw:: html

   </li>

.. raw:: html

   <li>

Upload the archive to Amazon S3

.. raw:: html

   </li>

.. raw:: html

   <li>

Specify the path to the archive in Amazon S3 as one of the parameters of
the training job

.. raw:: html

   </li>

.. raw:: html

   </ul>

Note: these steps are executed automatically by the Amazon SageMaker
Python SDK when using framework estimators for MXNet, Tensorflow, etc.

.. code:: ipython3

    import tarfile
    import os
    
    def create_tar_file(source_files, target=None):
        if target:
            filename = target
        else:
            _, filename = tempfile.mkstemp()
    
        with tarfile.open(filename, mode="w:gz") as t:
            for sf in source_files:
                # Add all files from the directory into the root of the directory structure of the tar
                t.add(sf, arcname=os.path.basename(sf))
        return filename
    
    create_tar_file(["source_dir/train.py", "source_dir/utils.py"], "sourcedir.tar.gz")

.. code:: ipython3

    sources = sagemaker_session.upload_data('sourcedir.tar.gz', bucket, prefix + '/code')
    print(sources)
    ! rm sourcedir.tar.gz

When starting the training job, we need to let the
sagemaker-training-toolkit library know where the sources are stored in
Amazon S3 and what is the module to be invoked. These parameters are
specified through the following reserved hyperparameters (these reserved
hyperparameters are injected automatically when using framework
estimators of the Amazon SageMaker Python SDK):

.. raw:: html

   <ul>

.. raw:: html

   <li>

sagemaker_program

.. raw:: html

   </li>

.. raw:: html

   <li>

sagemaker_submit_directory

.. raw:: html

   </li>

.. raw:: html

   </ul>

Finally, we can execute the training job by calling the fit() method of
the generic Estimator object defined in the Amazon SageMaker Python SDK
(https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/estimator.py).
This corresponds to calling the CreateTrainingJob() API
(https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTrainingJob.html).

.. code:: ipython3

    import sagemaker
    import json
    
    # JSON encode hyperparameters.
    def json_encode_hyperparameters(hyperparameters):
        return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}
    
    hyperparameters = json_encode_hyperparameters({
        "sagemaker_program": "train.py",
        "sagemaker_submit_directory": sources,
        "hp1": "value1",
        "hp2": 300,
        "hp3": 0.001})
    
    est = sagemaker.estimator.Estimator(container_image_uri,
                                        role,
                                        train_instance_count=1, 
                                        train_instance_type='local',
                                        base_job_name=prefix,
                                        hyperparameters=hyperparameters)
    
    train_config = sagemaker.session.s3_input('s3://{0}/{1}/train/'.format(bucket, prefix), content_type='text/csv')
    val_config = sagemaker.session.s3_input('s3://{0}/{1}/val/'.format(bucket, prefix), content_type='text/csv')
    
    est.fit({'train': train_config, 'validation': val_config })

.. raw:: html

   <h3>

Training with a custom SDK framework estimator

.. raw:: html

   </h3>

As you have seen, in the previous steps we had to upload our code to
Amazon S3 and then inject reserved hyperparameters to execute training.
In order to facilitate this task, you can also try defining a custom
framework estimator using the Amazon SageMaker Python SDK and run
training with that class, which will take care of managing these tasks.

.. code:: ipython3

    from sagemaker.estimator import Framework
    
    class CustomFramework(Framework):
        def __init__(
            self,
            entry_point,
            source_dir=None,
            hyperparameters=None,
            py_version="py3",
            framework_version=None,
            image_name=None,
            distributions=None,
            **kwargs
        ):
            super(CustomFramework, self).__init__(
                entry_point, source_dir, hyperparameters, image_name=image_name, **kwargs
            )
        
        def _configure_distribution(self, distributions):
            return
        
        def create_model(
            self,
            model_server_workers=None,
            role=None,
            vpc_config_override=None,
            entry_point=None,
            source_dir=None,
            dependencies=None,
            image_name=None,
            **kwargs
        ):
            return None
            
    import sagemaker
    
    est = CustomFramework(image_name=container_image_uri,
                          role=role,
                          entry_point='train.py',
                          source_dir='source_dir/',
                          train_instance_count=1, 
                          train_instance_type='local', # we use local mode
                          #train_instance_type='ml.m5.xlarge',
                          base_job_name=prefix,
                          hyperparameters={
                              "hp1": "value1",
                              "hp2": "300",
                              "hp3": "0.001"
                          })
    
    train_config = sagemaker.session.s3_input('s3://{0}/{1}/train/'.format(bucket, prefix), content_type='text/csv')
    val_config = sagemaker.session.s3_input('s3://{0}/{1}/val/'.format(bucket, prefix), content_type='text/csv')
    
    est.fit({'train': train_config, 'validation': val_config })

