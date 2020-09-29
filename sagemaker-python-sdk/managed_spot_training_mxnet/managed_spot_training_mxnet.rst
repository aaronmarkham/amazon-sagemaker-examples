Training SageMaker Models using the Apache MXNet Module API on SageMaker Managed Spot Training
==============================================================================================

The example here is almost the same as `Training and hosting SageMaker
Models using the Apache MXNet Module
API <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb>`__.

This notebook tackles the exact same problem with the same solution, but
it has been modified to be able to run using SageMaker Managed Spot
infrastructure. SageMaker Managed Spot uses `EC2 Spot
Instances <https://aws.amazon.com/ec2/spot/>`__ to run Training at a
lower cost.

Please read the original notebook and try it out to gain an
understanding of the ML use-case and how it is being solved. We will not
delve into that here in this notebook.

First setup variables and define functions
------------------------------------------

Again, we won’t go into detail explaining the code below, it has been
lifted verbatim from `Training and hosting SageMaker Models using the
Apache MXNet Module
API <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb>`__

.. code:: ipython3

    !pip install -qU awscli boto3 sagemaker

.. code:: ipython3

    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = Session().default_bucket()
    
    # Location to save your custom code in tar.gz format.
    custom_code_upload_location = 's3://{}/customcode/mxnet'.format(bucket)
    
    # Location where results of model training are saved.
    model_artifacts_location = 's3://{}/artifacts'.format(bucket)
    
    # IAM execution role that gives SageMaker access to resources in your AWS account.
    # We can use the SageMaker Python SDK to get the role from our notebook environment. 
    role = get_execution_role()
    
    import boto3
    
    region = boto3.Session().region_name
    train_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)
    test_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)

Managed Spot Training with MXNet
================================

For Managed Spot Training using MXNet we need to configure three things:
1. Enable the ``train_use_spot_instances`` constructor arg - a simple
self-explanatory boolean. 2. Set the ``train_max_wait`` constructor arg
- this is an int arg representing the amount of time you are willing to
wait for Spot infrastructure to become available. Some instance types
are harder to get at Spot prices and you may have to wait longer. You
are not charged for time spent waiting for Spot infrastructure to become
available, you’re only charged for actual compute time spent once Spot
instances have been successfully procured. 3. Setup a
``checkpoint_s3_uri`` constructor arg. This arg will tell SageMaker an
S3 location where to save checkpoints (assuming your algorithm has been
modified to save checkpoints periodically). While not strictly necessary
checkpointing is highly recommended for Manage Spot Training jobs due to
the fact that Spot instances can be interrupted with short notice and
using checkpoints to resume from the last interruption ensures you don’t
lose any progress made before the interruption.

Feel free to toggle the ``train_use_spot_instances`` variable to see the
effect of running the same job using regular (a.k.a. “On Demand”)
infrastructure.

Note that ``train_max_wait`` can be set if and only if
``train_use_spot_instances`` is enabled and **must** be greater than or
equal to ``train_max_run``.

.. code:: ipython3

    train_use_spot_instances = True
    train_max_run=3600
    train_max_wait = 7200 if train_use_spot_instances else None
    import uuid
    checkpoint_suffix = str(uuid.uuid4())[:8]
    checkpoint_s3_uri = 's3://{}/artifacts/mxnet-checkpoint-{}/'.format(bucket, checkpoint_suffix) if train_use_spot_instances else None

.. code:: ipython3

    from sagemaker.mxnet import MXNet
    
    mnist_estimator = MXNet(entry_point='mnist.py',
                            role=role,
                            output_path=model_artifacts_location,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            train_instance_type='ml.m4.xlarge',
                            framework_version='1.6.0',
                            py_version='py3',
                            distributions={'parameter_server': {'enabled': True}},
                            hyperparameters={'learning-rate': 0.1},
                            train_use_spot_instances=train_use_spot_instances,
                            train_max_run=train_max_run,
                            train_max_wait=train_max_wait,
                            checkpoint_s3_uri=checkpoint_s3_uri)
    mnist_estimator.fit({'train': train_data_location, 'test': test_data_location})

Savings
=======

Towards the end of the job you should see two lines of output printed:

-  ``Training seconds: X`` : This is the actual compute-time your
   training job spent
-  ``Billable seconds: Y`` : This is the time you will be billed for
   after Spot discounting is applied.

If you enabled the ``train_use_spot_instances`` var then you should see
a notable difference between ``X`` and ``Y`` signifying the cost savings
you will get for having chosen Managed Spot Training. This should be
reflected in an additional line: -
``Managed Spot Training savings: (1-Y/X)*100 %``
