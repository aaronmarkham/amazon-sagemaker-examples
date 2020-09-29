.. raw:: html

   <h1>

Script-mode Custom Training Container

.. raw:: html

   </h1>

This notebook demonstrates how to build and use a custom Docker
container for training with Amazon SageMaker that leverages on the
Script Mode execution that is implemented by the
sagemaker-training-toolkit library. Reference documentation is available
at https://github.com/aws/sagemaker-training-toolkit

We start by defining some variables like the current execution role, the
ECR repository that we are going to use for pushing the custom Docker
container and a default Amazon S3 bucket to be used by Amazon SageMaker.

.. code:: ipython3

    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    
    ecr_namespace = 'sagemaker-training-containers/'
    prefix = 'script-mode-container'
    
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

   <li>

Finally, we copy all contents in code/ (which is where our training code
is) under /opt/ml/code/ which is the path where
sagemaker-training-toolkit expects to find training code

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

    ! pygmentize ../docker/code/train.py

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

Finally, we can execute the training job by calling the fit() method of
the generic Estimator object defined in the Amazon SageMaker Python SDK
(https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/estimator.py).
This corresponds to calling the CreateTrainingJob() API
(https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTrainingJob.html).

.. code:: ipython3

    import sagemaker
    import json
    
    # JSON encode hyperparameters
    def json_encode_hyperparameters(hyperparameters):
        return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}
    
    hyperparameters = json_encode_hyperparameters({
        "hp1": "value1",
        "hp2": 300,
        "hp3": 0.001})
    
    est = sagemaker.estimator.Estimator(container_image_uri,
                                        role, 
                                        train_instance_count=1, 
                                        train_instance_type='local', # we use local mode
                                        #train_instance_type='ml.m5.xlarge',
                                        base_job_name=prefix,
                                        hyperparameters=hyperparameters)
    
    train_config = sagemaker.session.s3_input('s3://{0}/{1}/train/'.format(bucket, prefix), content_type='text/csv')
    val_config = sagemaker.session.s3_input('s3://{0}/{1}/val/'.format(bucket, prefix), content_type='text/csv')
    
    est.fit({'train': train_config, 'validation': val_config })

