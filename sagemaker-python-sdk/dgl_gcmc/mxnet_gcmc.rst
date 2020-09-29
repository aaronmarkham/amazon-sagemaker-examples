Training Graph Convolutional Matrix Completion by using the Deep Graph Library with MXNet backend on Amazon SageMaker
---------------------------------------------------------------------------------------------------------------------

The **Amazon SageMaker Python SDK** makes it easy to train Deep Graph
Library (DGL) models. In this example, you train `Graph Convolutional
Matrix Completion <https://arxiv.org/abs/1706.02263>`__ network using
the `DMLC DGL API <https://github.com/dmlc/dgl.git>`__ and the
`MovieLens dataset <https://grouplens.org/datasets/movielens/>`__. Three
datasets are supported: \* MovieLens 100K Dataset, MovieLens 100K movie
ratings. Stable benchmark dataset. 100,000 ratings from 1,000 users on
1,700 movies. \* MovieLens 1M Dataset, MovieLens 1M movie ratings.
Stable benchmark dataset. 1 million ratings from 6,000 users on 4,000
movies. \* MovieLens 10M Dataset, MovieLens 10M movie ratings. Stable
benchmark dataset. 10 million ratings and 100,000 tag applications
applied to 10,000 movies by 72,000 users.

Prerequisites
~~~~~~~~~~~~~

To get started, install necessary packages.

.. code:: ipython3

    !conda install -y boto3
    !conda install -c anaconda -y botocore

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # Setup session
    sess = sagemaker.Session()
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here.
    bucket = sess.default_bucket()
    
    # Location to put your custom code.
    custom_code_upload_location = 'customcode'
    
    # Location where results of model training are saved.
    model_artifacts_location = 's3://{}/artifacts'.format(bucket)
    
    # IAM role that gives Amazon SageMaker access to resources in your AWS account.
    # You can use the Amazon SageMaker Python SDK to get the role from your notebook environment. 
    role = get_execution_role()

The training script
~~~~~~~~~~~~~~~~~~~

The train.py script provides all the code you need for training an
Amazon SageMaker model.

.. code:: ipython3

    !cat train.py

Build GCMC Docker image
~~~~~~~~~~~~~~~~~~~~~~~

AWS provides basic Docker images in
https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html.
For both PyTorch 1.3 and MXNet 1.6, DGL is preinstalled. As this example
needs additional dependencies, you can download a Docker file to build a
new image. You should build a GCMC-specific Docker image and push it
into your Amazon Elastic Container Registry (Amazon ECR).

Note: Do change the GCMC.Dockerfile with the latest MXNet GPU deep
learning containers images name with py3 available in your region.

.. code:: sh

    %%sh
    account=$(aws sts get-caller-identity --query Account --output text)
    echo $account
    region=$(aws configure get region)
    
    docker_name=sagemaker-dgl-gcmc
    
    $(aws ecr get-login --no-include-email --region ${region} --registry-ids 763104351884)
    docker build -t $docker_name -f GCMC.Dockerfile .
    
    # Get the login command from ECR and execute it directly
    $(aws ecr get-login --region ${region} --no-include-email)
    
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${docker_name}:latest"
    # If the repository doesn't exist in ECR, create it.
    aws ecr describe-repositories --repository-names "${docker_name}" > /dev/null 2>&1
    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "${docker_name}" > /dev/null
    fi
    
    docker tag ${docker_name} ${fullname}
    
    docker push ${fullname}

Amazon SageMaker’s estimator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the Amazon SageMaker Estimator, you can run a single machine in
Amazon SageMaker, using CPU or GPU-based instances.

When you create the estimator, pass-in the file name of the training
script and the name of the IAM execution role. You can also use a few
other parameters. train_instance_count and train_instance_type determine
the number and type of Amazon SageMaker instances that will be used for
the training job. The hyperparameters parameter is a dictionary of
values that is passed to your training script as parameters so that you
can use argparse to parse them. You can see how to access these values
in the train.py script above.

In this example, you upload the whole code base (including train.py)
into an Amazon SageMaker container and run the GCMC training using the
MovieLens dataset.

You can also add a task_tag with value ‘DGL’ to help tracking the task.

.. code:: ipython3

    from sagemaker.mxnet.estimator import MXNet
    
    # Set target dgl-docker name
    docker_name='sagemaker-dgl-gcmc'
    
    CODE_PATH = '../dgl_gcmc'
    CODE_ENTRY = 'train.py'
    #code_location = sess.upload_data(CODE_PATH, bucket=bucket, key_prefix=custom_code_upload_location)
    
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    region = sess.boto_session.region_name
    image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, docker_name)
    print(image)
    
    params = {}
    params['data_name'] = 'ml-1m'
    # set output to SageMaker ML output
    params['save_dir'] = '/opt/ml/model/'
    task_tags = [{'Key':'ML Task', 'Value':'DGL'}]
    
    estimator = MXNet(entry_point=CODE_ENTRY,
                      source_dir=CODE_PATH,
                      role=role,
                      train_instance_count=1,
                      train_instance_type='ml.p3.2xlarge',
                      image_name=image,
                      hyperparameters=params,
                      tags=task_tags,
                      sagemaker_session=sess)

Running the Training Job
~~~~~~~~~~~~~~~~~~~~~~~~

After you construct the Estimator object, fit it using Amazon SageMaker.
The dataset is automatically downloaded.

.. code:: ipython3

    estimator.fit()

Output
------

You can get the model training output from the Amazon Sagemaker console
by searching for the training task and looking for the address of ‘S3
model artifact’
