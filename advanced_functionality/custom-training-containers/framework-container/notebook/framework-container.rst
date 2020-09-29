.. raw:: html

   <h1>

Custom Framework Container

.. raw:: html

   </h1>

This notebook demonstrates how to build and use a simple custom Docker
container for training with Amazon SageMaker that leverages on the
sagemaker-training-toolkit library to define framework containers. A
framework container is similar to a script-mode container, but in
addition it loads a Python framework module that is used to configure
the framework and then run the user-provided module.

Reference documentation is available at
https://github.com/aws/sagemaker-training-toolkit

We start by defining some variables like the current execution role, the
ECR repository that we are going to use for pushing the custom Docker
container and a default Amazon S3 bucket to be used by Amazon SageMaker.

.. code:: ipython3

    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    
    ecr_namespace = 'sagemaker-training-containers/'
    prefix = 'framework-container'
    
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
building our custom framework container:

.. code:: ipython3

    !pygmentize ../docker/Dockerfile

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

We copy a .tar.gz package named custom_framework_training-1.0.0.tar.gz
in the WORKDIR

.. raw:: html

   </li>

.. raw:: html

   <li>

We then install some Python libraries like numpy, pandas, ScikitLearn
and the package we copied at the previous step

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

Finally, we set the value of the environment variable
SAGEMAKER_TRAINING_MODULE to a python module in the training package we
installed

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   <h2>

Training module

.. raw:: html

   </h2>

When looking at the Dockerfile above, you might be askiong yourself what
the custom_framework_training-1.0.0.tar.gz package is. When building a
framework container, sagemaker-training-toolkit allows you to specify a
framework module that will be run first, and then invoke a user-provided
module.

The advantage of using this approach is that you can use the framework
module to configure the framework of choice or apply any settings
related to the libraries installed in the environment, and then run the
user module (we will see shortly how).

Our framework module is part of a Python package - that you can find in
the folder ../package/ - distributed as a .tar.gz by the Python
setuptools library (https://setuptools.readthedocs.io/en/latest/).

Setuptools uses a setup.py file to build the package. Following is the
content of this file:

.. code:: ipython3

    !pygmentize ../package/setup.py

This build script looks at the packages under the local src/ path and
specifies the dependency on sagemaker-training. The training module
contains the following code:

.. code:: ipython3

    !pygmentize ../package/src/custom_framework_training/training.py

The idea here is that we will use the entry_point.run() function of the
sagemaker-training-toolkit library to execute the user-provided module.
You might want to set additional framework-level configurations
(e.g. parameter servers) before calling the user module.

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

First, the script runs the setup.py to create the training package,
which is copied under ../docker/code/.

Then it builds the Docker container, creates the repository if it does
not exist, and finally pushes the container to the ECR repository. The
build task requires a few minutes to be executed the first time, then
Docker caches build outputs to be reused for the subsequent build
operations.

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
framework containers, we are not going to train a real model. The script
that will be executed does not define a specific training logic; it just
outputs the configurations injected by SageMaker and implements a dummy
training loop. Training data is also dummy. Let’s analyze the script
first:

.. code:: ipython3

    ! pygmentize source_dir/train.py

You can realize that the training code has been implemented as a
standard Python script, that will be invoked as a module by the
framework container code, passing hyperparameters as arguments.

Now, we upload some dummy data to Amazon S3, in order to define our
S3-based training channels.

.. code:: ipython3

    ! echo "val1, val2, val3" > dummy.csv
    print(sagemaker_session.upload_data('dummy.csv', bucket, prefix + '/train'))
    print(sagemaker_session.upload_data('dummy.csv', bucket, prefix + '/val'))
    ! rm dummy.csv

Framework containers enable dynamically running user-provided code
loading it from Amazon S3, so we need to:

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

