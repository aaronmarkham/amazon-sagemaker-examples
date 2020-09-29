Managed Spot Training for XGBoost
=================================

This notebook shows usage of SageMaker Managed Spot infrastructure for
XGBoost training. Below we show how Spot instances can be used for the
‘algorithm mode’ and ‘script mode’ training methods with the XGBoost
container.

`Managed Spot
Training <https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html>`__
uses Amazon EC2 Spot instance to run training jobs instead of on-demand
instances. You can specify which training jobs use spot instances and a
stopping condition that specifies how long Amazon SageMaker waits for a
job to run using Amazon EC2 Spot instances.

In this notebook we will perform XGBoost training as described
`here <>`__. See the original notebook for more details on the data.

Prerequisites
-------------

Ensuring the latest sagemaker sdk is installed. For a major version
upgrade, there might be some apis that may get deprecated.

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install -qU awscli boto3 "sagemaker>=1.71.0,<2.0.0"

Setup variables and define functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    
    import io
    import os
    import boto3
    import sagemaker
    import urllib
    
    role = sagemaker.get_execution_role()
    region = boto3.Session().region_name
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-xgboost-spot'
    # customize to your bucket where you have would like to store the data

Fetching the dataset
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    
    # Load the dataset
    FILE_DATA = 'abalone'
    urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone", FILE_DATA)
    sagemaker.Session().upload_data(FILE_DATA, bucket=bucket, key_prefix=prefix+'/train')

Obtaining the latest XGBoost container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We obtain the new container by specifying the framework version
(0.90-1). This version specifies the upstream XGBoost framework version
(0.90) and an additional SageMaker version (1). If you have an existing
XGBoost workflow based on the previous (0.72) container, this would be
the only change necessary to get the same workflow working with the new
container.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(region, 'xgboost', '1.0-1')

Training the XGBoost model
~~~~~~~~~~~~~~~~~~~~~~~~~~

After setting training parameters, we kick off training, and poll for
status until training is completed, which in this example, takes few
minutes.

To run our training script on SageMaker, we construct a
sagemaker.xgboost.estimator.XGBoost estimator, which accepts several
constructor arguments:

-  **entry_point**: The path to the Python script SageMaker runs for
   training and prediction.
-  **role**: Role ARN
-  **hyperparameters**: A dictionary passed to the train function as
   hyperparameters.
-  **train_instance_type** *(optional)*: The type of SageMaker instances
   for training. **Note**: This particular mode does not currently
   support training on GPU instance types.
-  **sagemaker_session** *(optional)*: The session used to train on
   Sagemaker.

.. code:: ipython3

    hyperparameters = {
            "max_depth":"5",
            "eta":"0.2",
            "gamma":"4",
            "min_child_weight":"6",
            "subsample":"0.7",
            "silent":"0",
            "objective":"reg:squarederror",
            "num_round":"50"}
    
    instance_type = 'ml.m5.2xlarge'
    output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-xgb')
    content_type = "libsvm"

If Spot instances are used, the training job can be interrupted, causing
it to take longer to start or finish. If a training job is interrupted,
a checkpointed snapshot can be used to resume from a previously saved
point and can save training time (and cost).

To enable checkpointing for Managed Spot Training using SageMaker
XGBoost we need to configure three things:

1. Enable the ``train_use_spot_instances`` constructor arg - a simple
   self-explanatory boolean.

2. Set the ``train_max_wait constructor`` arg - this is an int arg
   representing the amount of time you are willing to wait for Spot
   infrastructure to become available. Some instance types are harder to
   get at Spot prices and you may have to wait longer. You are not
   charged for time spent waiting for Spot infrastructure to become
   available, you’re only charged for actual compute time spent once
   Spot instances have been successfully procured.

3. Setup a ``checkpoint_s3_uri`` constructor arg - this arg will tell
   SageMaker an S3 location where to save checkpoints. While not
   strictly necessary, checkpointing is highly recommended for Manage
   Spot Training jobs due to the fact that Spot instances can be
   interrupted with short notice and using checkpoints to resume from
   the last interruption ensures you don’t lose any progress made before
   the interruption.

Feel free to toggle the ``train_use_spot_instances`` variable to see the
effect of running the same job using regular (a.k.a. “On Demand”)
infrastructure.

Note that ``train_max_wait`` can be set if and only if
``train_use_spot_instances`` is enabled and must be greater than or
equal to ``train_max_run``.

.. code:: ipython3

    import time
    
    job_name = 'DEMO-xgboost-spot-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print("Training job", job_name)
    
    train_use_spot_instances = True
    train_max_run = 3600
    train_max_wait = 7200 if train_use_spot_instances else None
    checkpoint_s3_uri = ('s3://{}/{}/checkpoints/{}'.format(bucket, prefix, job_name) if train_use_spot_instances 
                          else None)
    print("Checkpoint path:", checkpoint_s3_uri)
    
    estimator = sagemaker.estimator.Estimator(container, 
                                              role, 
                                              hyperparameters=hyperparameters,
                                              train_instance_count=1, 
                                              train_instance_type=instance_type, 
                                              train_volume_size=5,         # 5 GB 
                                              output_path=output_path, 
                                              sagemaker_session=sagemaker.Session(),
                                              train_use_spot_instances=train_use_spot_instances, 
                                              train_max_run=train_max_run, 
                                              train_max_wait=train_max_wait,
                                              checkpoint_s3_uri=checkpoint_s3_uri
                                             );
    train_input = sagemaker.s3_input(s3_data='s3://{}/{}/{}'.format(bucket, prefix, 'train'), content_type='libsvm')
    estimator.fit({'train': train_input}, job_name=job_name)

Savings
~~~~~~~

Towards the end of the job you should see two lines of output printed:

-  ``Training seconds: X`` : This is the actual compute-time your
   training job spent
-  ``Billable seconds: Y`` : This is the time you will be billed for
   after Spot discounting is applied.

If you enabled the ``train_use_spot_instances``, then you should see a
notable difference between ``X`` and ``Y`` signifying the cost savings
you will get for having chosen Managed Spot Training. This should be
reflected in an additional line: -
``Managed Spot Training savings: (1-Y/X)*100 %``

Enabling checkpointing for script mode
--------------------------------------

An additional mode of operation is to run customizable scripts as part
of the training and inference jobs. See `this
notebook <./xgboost_abalone_dist_script_mode.ipynb>`__ for details on
how to setup script mode.

Here we highlight the specific changes that would enable checkpointing
and use Spot instances.

Checkpointing in the framework mode for SageMaker XGBoost can be
performed using two convenient functions:

-  ``save_checkpoint``: this returns a callback function that performs
   checkpointing of the model for each round. This is passed to XGBoost
   as part of the
   ```callbacks`` <https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train>`__
   argument.

-  ``load_checkpoint``: This is used to load existing checkpoints to
   ensure training resumes from where it previously stopped.

Both functions take the checkpoint directory as input, which in the
below example is set to ``/opt/ml/checkpoints``. The primary arguments
that change for the ``xgb.train`` call are

1. ``xgb_model``: This refers to the previous checkpoint (saved from a
   previously run partial job) obtained by ``load_checkpoint``. This
   would be ``None`` if no previous checkpoint is available.
2. ``callbacks``: This contains a function that performs the
   checkpointing

Updated script looks like the following.

--------------

::

   CHECKPOINTS_DIR = '/opt/ml/checkpoints'   # default location for Checkpoints
   callbacks = [save_checkpoint(CHECKPOINTS_DIR)]
   prev_checkpoint, n_iterations_prev_run = load_checkpoint(CHECKPOINTS_DIR)
   bst = xgb.train(
           params=train_hp,
           dtrain=dtrain,
           evals=watchlist,
           num_boost_round=(args.num_round - n_iterations_prev_run),
           xgb_model=prev_checkpoint,
           callbacks=callbacks
       )

Using the SageMaker XGBoost Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The XGBoost estimator class in the SageMaker Python SDK allows us to run
that script as a training job on the Amazon SageMaker managed training
infrastructure. We’ll also pass the estimator our IAM role, the type of
instance we want to use, and a dictionary of the hyperparameters that we
want to pass to our script.

.. code:: ipython3

    from sagemaker.session import s3_input
    from sagemaker.xgboost.estimator import XGBoost
    
    job_name = 'DEMO-xgboost-regression-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print("Training job", job_name)
    checkpoint_s3_uri = ('s3://{}/{}/checkpoints/{}'.format(bucket, prefix, job_name) if train_use_spot_instances 
                          else None)
    print("Checkpoint path:", checkpoint_s3_uri)
    
    xgb_script_mode_estimator = XGBoost(
        entry_point="abalone.py",
        hyperparameters=hyperparameters,
        image_name=container,
        role=role, 
        train_instance_count=1,
        train_instance_type=instance_type,
        framework_version="0.90-1",
        output_path="s3://{}/{}/{}/output".format(bucket, prefix, "xgboost-script-mode"),
        train_use_spot_instances=train_use_spot_instances,
        train_max_run=train_max_run,
        train_max_wait=train_max_wait,
        checkpoint_s3_uri=checkpoint_s3_uri
    )

Training is as simple as calling ``fit`` on the Estimator. This will
start a SageMaker Training job that will download the data, invoke the
entry point code (in the provided script file), and save any model
artifacts that the script creates. In this case, the script requires a
``train`` and a ``validation`` channel. Since we only created a
``train`` channel, we re-use it for validation.

.. code:: ipython3

    xgb_script_mode_estimator.fit({'train': train_input, 'validation': train_input}, job_name=job_name)
