Enable Spot Training with Amazon SageMaker Debugger
===================================================

Amazon SageMaker Debugger is a new capability of Amazon SageMaker that
allows debugging machine learning training. It lets you go beyond just
looking at scalars like losses and accuracies during training and gives
you full visibility into all tensors ‘flowing through the graph’ during
training. Amazon SageMaker Debugger helps you to monitor your training
in near real time using rules and would provide you alerts, once it has
detected inconsistency in training flow.

Using Amazon SageMaker Debugger is a two step process: Saving tensors
and Analysis.

Saving tensors
~~~~~~~~~~~~~~

Tensors define the state of the training job at any particular instant
in its lifecycle. Debugger exposes a library which allows you to capture
these tensors and save them for analysis.

Analysis
~~~~~~~~

There are two ways to get to tensors and run analysis on them. One way
is to use concept called **Rules**. For more information about a
rules-based approach to analysis, see
`Rules <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#Rules>`__.
You can also perform interactive analysis in a notebook. Please refer to
our other notebooks on how to do that.

Spot Training
-------------

This notebook talks about how Amazon SageMaker Debugger feature can also
be used with Spot Training. For more information related to spot
training in Amazon SageMaker please see `Spot
Training <https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html>`__.

The examples uses a small gluon CNN model and trains it on the
FashionMNIST dataset. If during the training spot instance terminates,
the training and analysis of tensors will continue from the last saved
checkpoint.

.. code:: ipython3

    import sagemaker
    import boto3
    import os
    from sagemaker.mxnet import MXNet
    from sagemaker.debugger import Rule, rule_configs


Configuring the inputs for the training job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now call the Amazon SageMaker MXNet Estimator to kick off a training job
along with enabling Debugger functionality.

-  ``entrypoint_script`` points to the simple MXNet training script that
   is ran by training job
-  ``hyperparameters`` are the parameters that will be passed to the
   training script.

.. code:: ipython3

    # Set the SageMaker Session
    sagemaker_session = sagemaker.Session()
    
    # Define the entrypoint script
    entrypoint_script='mxnet_gluon_spot_training.py'
    hyperparameters = {'batch-size' : 100,  'epochs' : 5, 'checkpoint-path' : '/opt/ml/checkpoints' }


Training MXNet models in Amazon SageMaker with Amazon SageMaker Debugger
------------------------------------------------------------------------

Train a small MXNet CNN model with the FashonMNIST dataset in this
notebook, with Amazon SageMaker Debugger enabled. This is done using an
Amazon SageMaker MXNet 1.6.0 container with script mode. Amazon
SageMaker Debugger currently works with Python3, so be sure to set
``py_version='py3'`` when creating the Amazon SageMaker Estimator.

Enable Amazon SageMaker Debugger and Spot Training in Estimator object
----------------------------------------------------------------------

Enabling Amazon SageMaker Debugger in training job can be accomplished
by adding its configuration into Estimator object constructor:

.. code:: python

   sagemaker_simple_estimator = MXNet(...,
       # Parameters required to enable spot training.
       train_use_spot_instances=True, #Set it to True to enable spot training.
       train_max_wait = 10000  # This should be equal to or greater than train_max_run in seconds'
       checkpoint_local_path = '/opt/ml/checkpoints/' # This is local path where checkpoints will be stored during training. Default path is /opt/ml/checkpoints'.The training script should generate the checkpoints.
       checkpoint_s3_uri = 's3://bucket/prefix' # Uri to S3 bucket where the checkpoints captured by the model will be stored.
       ## Rule Parameter
       rules = [Rule.sagemaker(rule_configs.vanishing_gradient())]
   )

In this section, we will focus on parameters that are needed to enable
Spot Training.

-  ``train_use_spot_instance`` : This parameter should be set to ‘True’
   to enable the spot training.
-  ``train_max_wait`` : This parameter (in seconds) should be set equal
   to or greater than ‘train_max_run’.
-  ``checkpoint_s3_uri`` : This is URI to S3 bucket where the
   checkpoints will be stored before the spot instance terminated. Once
   the training is resumed, the checkpoints from this S3 bucket will be
   restored to ‘checkpoint_local_path’ in the new instance. Ensure that
   the S3 bucket is created in the same region as that of current
   session.
-  ``checkpoint_local_path``: This is the local path where the model
   will save the checkpoints perodically. The default path is set to
   ‘/opt/ml/checkpoints’. Ensure that the model under training is saving
   the checkpoints in this path. Note that in hyperparameters we are
   setting ‘checkpoint-path’ so that the training script will save the
   checkpoints in that directory.

Rule Parameter
~~~~~~~~~~~~~~

We are going to run the *vanishing_gradient* rule during this training.
By specifying this parameter, we are enabling the Amazon SageMaker
Debugger functionality to collect the *gradients* during this training.
The *gradients* will be collected every 500th step as part of the
default configurations for this Rule.

How Spot Training works with Amazon SageMaker Debugger
------------------------------------------------------

Amazon SageMaker Debugger can be enabled even for training with Spot
Instances. Spot instances can be interrupted, causing jobs to take
longer to start or finish. To leverage the managed spot instance support
that Amazon SageMaker provides, you need to configure your training job
to save checkpoints. Amazon SageMaker copies checkpoint data from a
local path to Amazon S3. When the job is restarted on a different
instance, Amazon SageMaker copies the data from Amazon S3 back into the
local path. The training can then resume from the last checkpoint
instead of restarting.

Amazon SageMaker Debugger relies on the checkpoints mechanism to
continue emitting tensors from the last saved checkpoint. The Amazon
SageMaker Debugger saves the metadata containing last saved state
whenver user creates a checkpoint in *checkpoint_local_path*. Along with
the checkpoints, this metadata also gets saved to Amazon S3 when the
instance is interrupted. Upon restart, along with the checkpoints, this
metadata is also copied back to the instance. The Amazon SageMaker
Debugger reads the last saved state from the metadata and continues to
emit the tensors from that step. This minimizes the emission of
duplicate tensors. Note that currently, the rule job continues to wait
till even if the training job is interrupted.

.. code:: ipython3

    # Make sure to set this to your bucket and location
    # Ensure that the bucket exists in the same region as that of current region.
    BUCKET_NAME = sagemaker_session.default_bucket()
    LOCATION_IN_BUCKET = 'smdebug-checkpoints'
    
    checkpoint_s3_bucket = 's3://{BUCKET_NAME}/{LOCATION_IN_BUCKET}'.format(BUCKET_NAME=BUCKET_NAME, LOCATION_IN_BUCKET=LOCATION_IN_BUCKET)
    
    # Local path where the model will save its checkpoints.
    checkpoint_local_path = '/opt/ml/checkpoints'


.. code:: ipython3

    estimator = MXNet(
        role=sagemaker.get_execution_role(),
        base_job_name='smdebugger-spot-training-demo-mxnet',
        train_instance_count=1,
        train_instance_type='ml.m4.xlarge',
        train_volume_size = 400,
        entry_point=entrypoint_script,
        hyperparameters = hyperparameters,
        framework_version='1.6.0',
        py_version='py3',
        train_max_run=3600,
        sagemaker_session=sagemaker_session,
        
        # Parameters required to enable spot training.
        train_use_spot_instances=True, #Set it to True to enable spot training.
        train_max_wait = 3600, #This should be equal to or greater than train_max_run in seconds
        checkpoint_s3_uri = checkpoint_s3_bucket, #Set the S3 URI to store the checkpoints.
        checkpoint_local_path = checkpoint_local_path, #This is default path where checkpoints will be stored. The training script should generate the checkpoints.
        
        ## Rule parameter
        rules = [Rule.sagemaker(rule_configs.vanishing_gradient())]
    )

.. code:: ipython3

    estimator.fit()
