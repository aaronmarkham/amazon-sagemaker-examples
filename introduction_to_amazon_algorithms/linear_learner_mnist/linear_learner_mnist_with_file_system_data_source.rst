Train Linear Learner model using File System Data Source
========================================================

This notebook example is similar to `An Introduction to Linear Learner
with
MNIST <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist.ipynb>`__.

`An Introduction to Linear Learner with
MNIST <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist.ipynb>`__
has been adapted to walk you through on using the AWS Elastic File
System (EFS) or AWS FSx for Lustre (FSxLustre) as an input datasource to
training jobs.

Please read the original notebook and try it out to gain an
understanding of the ML use-case and how it is being solved. We will not
delve into that here in this notebook.

Setup
-----

Again, we won’t go into detail explaining the code below, it has been
lifted verbatim from `An Introduction to Linear Learner with
MNIST <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist.ipynb>`__.

.. code:: ipython3

    !pip install -U --quiet "sagemaker>=1.14.2,<2"
    # Define IAM role
    import boto3
    import re
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    role = get_execution_role()
    
    # Specify training container
    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'linear-learner')
    
    # Specify S3 bucket and prefix that you want to use for model data
    # Feel free to specify a different bucket here if you wish.
    bucket = Session().default_bucket()
    prefix = 'sagemaker/DEMO-linear-mnist'
    
    # Setup an output S3 location for the model artifact
    output_location = 's3://{}/{}/output'.format(bucket, prefix)
    print('training artifacts will be uploaded to: {}'.format(output_location))

Prepare File System Input
-------------------------

Next, we specify the details of file system as an input to your training
job. Using file system as a data source eliminates the time your
training job spends downloading data with data streamed directly from
file system into your training algorithm.

.. code:: ipython3

    from sagemaker.inputs import FileSystemInput
    
    # Specify file system id.
    file_system_id = '<your_file_system_id>'
    
    # Specify directory path associated with the file system. You need to provide normalized and absolute path here.
    file_system_directory_path = '<your_file_system_directory_path>'
    
    # Specify the access mode of the mount of the directory associated with the file system. 
    # Directory can be mounted either in 'ro'(read-only) or 'rw' (read-write).
    file_system_access_mode = '<your_file_system_access_mode>'
    
    # Specify your file system type, "EFS" or "FSxLustre".
    file_system_type = '<your_file_system_type>'
    
    # Give Amazon SageMaker Training Jobs Access to FileSystem Resources in Your Amazon VPC.
    security_groups_ids = '<your_security_groups_ids>'
    subnets = '<your_subnets>'
    
    file_system_input = FileSystemInput(file_system_id=file_system_id,
                                        file_system_type=file_system_type,
                                        directory_path=file_system_directory_path,
                                        file_system_access_mode=file_system_access_mode)


Training the linear model
-------------------------

Once we have the file system provisioned and file system input ready for
training, the next step is to actually train the model.

.. code:: ipython3

    import boto3
    import sagemaker
    
    sess = sagemaker.Session()
    
    linear = sagemaker.estimator.Estimator(container,
                                           role, 
                                           subnets=subnets,
                                           security_group_ids=security_groups_ids,
                                           train_instance_count=1, 
                                           train_instance_type='ml.c4.xlarge',
                                           output_path=output_location,
                                           sagemaker_session=sess)
                                                               
    linear.set_hyperparameters(feature_dim=784,
                               predictor_type='binary_classifier',
                               mini_batch_size=200)
    
    linear.fit({'train': file_system_input})

Towards the end of the job you should see model artifact generated and
uploaded to ``output_location``.
