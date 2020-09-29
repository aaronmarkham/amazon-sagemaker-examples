Amazon SageMaker Experiment Trials for Distirbuted Training of Mask-RCNN
========================================================================

This notebook is a step-by-step tutorial on Amazon SageMaker Experiment
Trials for distributed tranining of `Mask
R-CNN <https://arxiv.org/abs/1703.06870>`__ implemented in
`TensorFlow <https://www.tensorflow.org/>`__ framework.

Concretely, we will describe the steps for SagerMaker Experiment Trials
for training `TensorPack
Faster-RCNN/Mask-RCNN <https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN>`__
and `AWS Samples Mask
R-CNN <https://github.com/aws-samples/mask-rcnn-tensorflow>`__ in
`Amazon SageMaker <https://aws.amazon.com/sagemaker/>`__ using `Amazon
S3 <https://aws.amazon.com/s3/>`__ as data source.

The outline of steps is as follows:

1. Stage COCO 2017 dataset in `Amazon S3 <https://aws.amazon.com/s3/>`__
2. Build SageMaker training image and push it to `Amazon
   ECR <https://aws.amazon.com/ecr/>`__
3. Configure data input channels
4. Configure hyper-prarameters
5. Define training metrics
6. Define training job
7. Define SageMaker Experiment Trials to start the training jobs

Before we get started, let us initialize two python variables
``aws_region`` and ``s3_bucket`` that we will use throughout the
notebook:

.. code:: ipython3

    aws_region = # aws-region-code e.g. us-east-1
    s3_bucket  = # your-s3-bucket-name

Stage COCO 2017 dataset in Amazon S3
------------------------------------

We use `COCO 2017 dataset <http://cocodataset.org/#home>`__ for
training. We download COCO 2017 training and validation dataset to this
notebook instance, extract the files from the dataset archives, and
upload the extracted files to your Amazon `S3
bucket <https://docs.aws.amazon.com/en_pv/AmazonS3/latest/gsg/CreatingABucket.html>`__
with the prefix ``mask-rcnn/sagemaker/input/train``. The
``prepare-s3-bucket.sh`` script executes this step.

.. code:: ipython3

    !cat ./prepare-s3-bucket.sh

Using your *Amazon S3 bucket* as argument, run the cell below. If you
have already uploaded COCO 2017 dataset to your Amazon S3 bucket *in
this AWS region*, you may skip this step. The expected time to execute
this step is 20 minutes.

.. code:: ipython3

    %%time
    !./prepare-s3-bucket.sh {s3_bucket}

Build and push SageMaker training images
----------------------------------------

For this step, the `IAM
Role <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html>`__
attached to this notebook instance needs full access to Amazon ECR
service. If you created this notebook instance using the
``./stack-sm.sh`` script in this repository, the IAM Role attached to
this notebook instance is already setup with full access to ECR service.

Below, we have a choice of two different implementations:

1. `TensorPack
   Faster-RCNN/Mask-RCNN <https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN>`__
   implementation supports a maximum per-GPU batch size of 1, and does
   not support mixed precision. It can be used with mainstream
   TensorFlow releases.

2. `AWS Samples Mask
   R-CNN <https://github.com/aws-samples/mask-rcnn-tensorflow>`__ is an
   optimized implementation that supports a maximum batch size of 4 and
   supports mixed precision. This implementation uses custom TensorFlow
   ops. The required custom TensorFlow ops are available in `AWS Deep
   Learning
   Container <https://github.com/aws/deep-learning-containers/blob/master/available_images.md>`__
   images in ``tensorflow-training`` repository with image tag
   ``1.15.2-gpu-py36-cu100-ubuntu18.04``, or later.

It is recommended that you build and push both SageMaker training images
and use either image for training later.

TensorPack Faster-RCNN/Mask-RCNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``./container/build_tools/build_and_push.sh`` script to build and
push the TensorPack Faster-RCNN/Mask-RCNN training image to Amazon ECR.

.. code:: ipython3

    !cat ./container/build_tools/build_and_push.sh

Using your *AWS region* as argument, run the cell below.

.. code:: ipython3

    %%time
    ! ./container/build_tools/build_and_push.sh {aws_region}

Set ``tensorpack_image`` below to Amazon ECR URI of the image you pushed
above.

.. code:: ipython3

    tensorpack_image = # mask-rcnn-tensorpack-sagemaker ECR URI

AWS Samples Mask R-CNN
~~~~~~~~~~~~~~~~~~~~~~

Use ``./container-optimized/build_tools/build_and_push.sh`` script to
build and push the AWS Samples Mask R-CNN training image to Amazon ECR.

.. code:: ipython3

    !cat ./container-optimized/build_tools/build_and_push.sh

Using your *AWS region* as argument, run the cell below.

.. code:: ipython3

    %%time
    ! ./container-optimized/build_tools/build_and_push.sh {aws_region}

Set ``aws_samples_image`` below to Amazon ECR URI of the image you
pushed above.

.. code:: ipython3

    aws_samples_image = # mask-rcnn-tensorflow-sagemaker ECR URI

SageMaker Initialization
------------------------

First we upgrade SageMaker to 2.3.0 API. If your notebook is already
using latest Sagemaker 2.x API, you may skip the next cell.

.. code:: ipython3

    ! pip install --upgrade pip
    ! pip install sagemaker==2.3.0

We have staged the data and we have built and pushed the training docker
image to Amazon ECR. Now we are ready to start using Amazon SageMaker.

.. code:: ipython3

    %%time
    import os
    import time
    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.estimator import Estimator
    
    role = get_execution_role() # provide a pre-existing role ARN as an alternative to creating a new role
    print(f'SageMaker Execution Role:{role}')
    
    client = boto3.client('sts')
    account = client.get_caller_identity()['Account']
    print(f'AWS account:{account}')
    
    session = boto3.session.Session()
    region = session.region_name
    print(f'AWS region:{region}')

Next, we set ``training_image`` to the Amazon ECR image URI you saved in
a previous step.

.. code:: ipython3

    training_image = # set to tensorpack_image or aws_samples_image 
    print(f'Training image: {training_image}')

Define SageMaker Data Channels
------------------------------

Next, we define the *train* data channel using EFS file-system. To do
so, we need to specify the EFS file-system id, which is shown in the
output of the command below.

.. code:: ipython3

    !df -kh | grep 'fs-' | sed 's/\(fs-[0-9a-z]*\).*/\1/'

Set the EFS ``file_system_id`` below to the ouput of the command shown
above. In the cell below, we define the ``train`` data input channel.

.. code:: ipython3

    from sagemaker.inputs import FileSystemInput
    
    # Specify EFS ile system id.
    file_system_id = # 'fs-xxxxxxxx'
    print(f"EFS file-system-id: {file_system_id}")
    
    # Specify directory path for input data on the file system. 
    # You need to provide normalized and absolute path below.
    file_system_directory_path = '/mask-rcnn/sagemaker/input/train'
    print(f'EFS file-system data input path: {file_system_directory_path}')
    
    # Specify the access mode of the mount of the directory associated with the file system. 
    # Directory must be mounted  'ro'(read-only).
    file_system_access_mode = 'ro'
    
    # Specify your file system type
    file_system_type = 'EFS'
    
    train = FileSystemInput(file_system_id=file_system_id,
                                        file_system_type=file_system_type,
                                        directory_path=file_system_directory_path,
                                        file_system_access_mode=file_system_access_mode)

Next, we define the model output location in S3 bucket.

.. code:: ipython3

    prefix = "mask-rcnn/sagemaker" #prefix in your bucket
    s3_output_location = f's3://{s3_bucket}/{prefix}/output'
    print(f'S3 model output location: {s3_output_location}')

Configure Hyper-parameters
--------------------------

Next, we define the hyper-parameters.

Note, some hyper-parameters are different between the two
implementations. The batch size per GPU in TensorPack
Faster-RCNN/Mask-RCNN is fixed at 1, but is configurable in AWS Samples
Mask-RCNN. The learning rate schedule is specified in units of steps in
TensorPack Faster-RCNN/Mask-RCNN, but in epochs in AWS Samples
Mask-RCNN.

The detault learning rate schedule values shown below correspond to
training for a total of 24 epochs, at 120,000 images per epoch.

.. raw:: html

   <table align='left'>

.. raw:: html

   <caption>

TensorPack Faster-RCNN/Mask-RCNN Hyper-parameters

.. raw:: html

   </caption>

.. raw:: html

   <tr>

.. raw:: html

   <th style="text-align:center">

Hyper-parameter

.. raw:: html

   </th>

.. raw:: html

   <th style="text-align:center">

Description

.. raw:: html

   </th>

.. raw:: html

   <th style="text-align:center">

Default

.. raw:: html

   </th>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

mode_fpn

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Flag to indicate use of Feature Pyramid Network (FPN) in the Mask R-CNN
model backbone

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

“True”

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

mode_mask

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

A value of “False” means Faster-RCNN model, “True” means Mask R-CNN
moodel

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

“True”

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

eval_period

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Number of epochs period for evaluation during training

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

1

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

lr_schedule

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Learning rate schedule in training steps

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘[240000, 320000, 360000]’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

batch_norm

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Batch normalization option (‘FreezeBN’, ‘SyncBN’, ‘GN’, ‘None’)

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘FreezeBN’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

images_per_epoch

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Images per epoch

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

120000

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

data_train

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Training data under data directory

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘coco_train2017’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

data_val

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Validation data under data directory

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘coco_val2017’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

resnet_arch

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Must be ‘resnet50’ or ‘resnet101’

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘resnet50’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

backbone_weights

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

ResNet backbone weights

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘ImageNet-R50-AlignPadding.npz’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

load_model

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Pre-trained model to load

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

config:

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Any hyperparamter prefixed with config: is set as a model config
parameter

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

.. raw:: html

   <table align='left'>

.. raw:: html

   <caption>

AWS Samples Mask-RCNN Hyper-parameters

.. raw:: html

   </caption>

.. raw:: html

   <tr>

.. raw:: html

   <th style="text-align:center">

Hyper-parameter

.. raw:: html

   </th>

.. raw:: html

   <th style="text-align:center">

Description

.. raw:: html

   </th>

.. raw:: html

   <th style="text-align:center">

Default

.. raw:: html

   </th>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

mode_fpn

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Flag to indicate use of Feature Pyramid Network (FPN) in the Mask R-CNN
model backbone

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

“True”

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

mode_mask

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

A value of “False” means Faster-RCNN model, “True” means Mask R-CNN
moodel

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

“True”

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

eval_period

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Number of epochs period for evaluation during training

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

1

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

lr_epoch_schedule

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Learning rate schedule in epochs

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘[(16, 0.1), (20, 0.01), (24, None)]’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

batch_size_per_gpu

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Batch size per gpu ( Minimum 1, Maximum 4)

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

4

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

batch_norm

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Batch normalization option (‘FreezeBN’, ‘SyncBN’, ‘GN’, ‘None’)

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘FreezeBN’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

images_per_epoch

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Images per epoch

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

120000

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

data_train

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Training data under data directory

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘train2017’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

data_val

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Validation data under data directory

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘val2017’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

resnet_arch

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Must be ‘resnet50’ or ‘resnet101’

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘resnet50’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

backbone_weights

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

ResNet backbone weights

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

‘ImageNet-R50-AlignPadding.npz’

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

load_model

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Pre-trained model to load

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td style="text-align:center">

config:

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:left">

Any hyperparamter prefixed with config: is set as a model config
parameter

.. raw:: html

   </td>

.. raw:: html

   <td style="text-align:center">

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

.. code:: ipython3

    hyperparameters = {
                        "mode_fpn": "True",
                        "mode_mask": "True",
                        "eval_period": 1,
                        "batch_norm": "FreezeBN"
                      }

Define Training Metrics
-----------------------

Next, we define the regular expressions that SageMaker uses to extract
algorithm metrics from training logs and send them to `AWS CloudWatch
metrics <https://docs.aws.amazon.com/en_pv/AmazonCloudWatch/latest/monitoring/working_with_metrics.html>`__.
These algorithm metrics are visualized in SageMaker console.

.. code:: ipython3

    metric_definitions=[
                 {
                    "Name": "fastrcnn_losses/box_loss",
                    "Regex": ".*fastrcnn_losses/box_loss:\\s*(\\S+).*"
                },
                {
                    "Name": "fastrcnn_losses/label_loss",
                    "Regex": ".*fastrcnn_losses/label_loss:\\s*(\\S+).*"
                },
                {
                    "Name": "fastrcnn_losses/label_metrics/accuracy",
                    "Regex": ".*fastrcnn_losses/label_metrics/accuracy:\\s*(\\S+).*"
                },
                {
                    "Name": "fastrcnn_losses/label_metrics/false_negative",
                    "Regex": ".*fastrcnn_losses/label_metrics/false_negative:\\s*(\\S+).*"
                },
                {
                    "Name": "fastrcnn_losses/label_metrics/fg_accuracy",
                    "Regex": ".*fastrcnn_losses/label_metrics/fg_accuracy:\\s*(\\S+).*"
                },
                {
                    "Name": "fastrcnn_losses/num_fg_label",
                    "Regex": ".*fastrcnn_losses/num_fg_label:\\s*(\\S+).*"
                },
                 {
                    "Name": "maskrcnn_loss/accuracy",
                    "Regex": ".*maskrcnn_loss/accuracy:\\s*(\\S+).*"
                },
                {
                    "Name": "maskrcnn_loss/fg_pixel_ratio",
                    "Regex": ".*maskrcnn_loss/fg_pixel_ratio:\\s*(\\S+).*"
                },
                {
                    "Name": "maskrcnn_loss/maskrcnn_loss",
                    "Regex": ".*maskrcnn_loss/maskrcnn_loss:\\s*(\\S+).*"
                },
                {
                    "Name": "maskrcnn_loss/pos_accuracy",
                    "Regex": ".*maskrcnn_loss/pos_accuracy:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(bbox)/IoU=0.5",
                    "Regex": ".*mAP\\(bbox\\)/IoU=0\\.5:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(bbox)/IoU=0.5:0.95",
                    "Regex": ".*mAP\\(bbox\\)/IoU=0\\.5:0\\.95:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(bbox)/IoU=0.75",
                    "Regex": ".*mAP\\(bbox\\)/IoU=0\\.75:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(bbox)/large",
                    "Regex": ".*mAP\\(bbox\\)/large:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(bbox)/medium",
                    "Regex": ".*mAP\\(bbox\\)/medium:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(bbox)/small",
                    "Regex": ".*mAP\\(bbox\\)/small:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(segm)/IoU=0.5",
                    "Regex": ".*mAP\\(segm\\)/IoU=0\\.5:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(segm)/IoU=0.5:0.95",
                    "Regex": ".*mAP\\(segm\\)/IoU=0\\.5:0\\.95:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(segm)/IoU=0.75",
                    "Regex": ".*mAP\\(segm\\)/IoU=0\\.75:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(segm)/large",
                    "Regex": ".*mAP\\(segm\\)/large:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(segm)/medium",
                    "Regex": ".*mAP\\(segm\\)/medium:\\s*(\\S+).*"
                },
                {
                    "Name": "mAP(segm)/small",
                    "Regex": ".*mAP\\(segm\\)/small:\\s*(\\S+).*"
                }  
                
        ]

Define SageMaker Experiment
---------------------------

To define SageMaker Experiment, we first install
``sagemaker-experiments`` package.

.. code:: ipython3

    ! pip install sagemaker-experiments==0.1.20

Next, we import the SageMaker Experiment modules.

.. code:: ipython3

    from smexperiments.experiment import Experiment
    from smexperiments.trial import Trial
    from smexperiments.trial_component import TrialComponent
    from smexperiments.tracker import Tracker
    import time

Next, we define a ``Tracker`` for tracking input data used in the
SageMaker Trials in this Experiment. Specify the S3 URL of your dataset
in the ``value`` below and change the name of the dataset if you are
using a different dataset.

.. code:: ipython3

    sm = session.client('sagemaker')
    with Tracker.create(display_name="Preprocessing", sagemaker_boto_client=sm) as tracker:
        # we can log the s3 uri to the dataset used for training
        tracker.log_input(name="coco-2017-dataset", 
                          media_type="s3/uri", 
                          value= f's3://{s3_bucket}/{prefix}/input/train' # specify S3 URL to your dataset
                         )

Next, we create a SageMaker Experiment.

.. code:: ipython3

    mrcnn_experiment = Experiment.create(
        experiment_name=f"mask-rcnn-experiment-{int(time.time())}", 
        description="Mask R-CNN experiment", 
        sagemaker_boto_client=sm)
    print(mrcnn_experiment)

We run the training job in your private VPC, so we need to set the
``subnets`` and ``security_group_ids`` prior to running the cell below.
You may specify multiple subnet ids in the ``subnets`` list. The subnets
included in the ``sunbets`` list must be part of the output of
``./stack-sm.sh`` CloudFormation stack script used to create this
notebook instance. Specify only one security group id in
``security_group_ids`` list. The security group id must be part of the
output of ``./stack-sm.sh`` script.

.. code:: ipython3

    security_group_ids = # ['sg-xxxxxxxx']
    subnets =  # ['subnet-xxxxxxx', 'subnet-xxxxxxx', 'subnet-xxxxxxx']
    sagemaker_session = sagemaker.session.Session(boto_session=session)

Next, we use SageMaker
`Estimator <https://sagemaker.readthedocs.io/en/stable/estimators.html>`__
API to define a SageMaker Training Job for each SageMaker Trial we need
to run within the SageMaker Experiment.

We recommned using 8 GPUs, so we set ``train_instance_count=1`` and
``train_instance_type='ml.p3.16xlarge'``, because there are 8 Tesla V100
GPUs per ``ml.p3.16xlarge`` instance. We recommend using 100 GB `Amazon
EBS <https://aws.amazon.com/ebs/>`__ storage volume with each training
instance, so we set ``train_volume_size = 100``. We want to replicate
training data to each training instance, so we set
``input_mode= 'File'``.

Next, we will iterate through the Trial parameters and start two trials,
one for ResNet architecture ``resnet50``, and a second Trial for
``resnet101``.

.. code:: ipython3

    trial_params = [ ('resnet50', 'ImageNet-R50-AlignPadding.npz'), 
                    ('resnet101', 'ImageNet-R101-AlignPadding.npz')]
    
    for resnet_arch, backbone_weights in trial_params:
        
        hyperparameters['resnet_arch'] = resnet_arch
        hyperparameters['backbone_weights'] = backbone_weights
        
        trial_name = f"mask-rcnn-{resnet_arch}-{int(time.time())}"
        mrcnn_trial = Trial.create(
                            trial_name=trial_name, 
                            experiment_name=mrcnn_experiment.experiment_name,
                            sagemaker_boto_client=sm,
        )
        
        # associate the proprocessing trial component with the current trial
        mrcnn_trial.add_trial_component(tracker.trial_component)
        print(mrcnn_trial)
    
        mask_rcnn_estimator = Estimator(image_uri=training_image,
                                             role=role, 
                                             instance_count=4, 
                                             instance_type='ml.p3.16xlarge',
                                             volume_size = 100,
                                             max_run = 400000,
                                             input_mode= 'File',
                                             output_path=s3_output_location,
                                             sagemaker_session=sagemaker_session, 
                                             hyperparameters = hyperparameters,
                                             metric_definitions = metric_definitions,
                                             subnets=subnets,
                                             security_group_ids=security_group_ids)
        
        # Specify directory path for log output on the EFS file system.
        # You need to provide normalized and absolute path below.
        # For example, '/mask-rcnn/sagemaker/output/log'
        # Log output directory must not exist
        file_system_directory_path = f'/mask-rcnn/sagemaker/output/{mrcnn_trial.trial_name}'
        print(f"EFS log directory:{file_system_directory_path}")
    
        # Create the log output directory. 
        # EFS file-system is mounted on '$HOME/efs' mount point for this notebook.
        home_dir=os.environ['HOME']
        local_efs_path = os.path.join(home_dir,'efs', file_system_directory_path[1:])
        print(f"Creating log directory on EFS: {local_efs_path}")
    
        assert not os.path.isdir(local_efs_path)
        ! sudo mkdir -p -m a=rw {local_efs_path}
        assert os.path.isdir(local_efs_path)
    
        # Specify the access mode of the mount of the directory associated with the file system. 
        # Directory must be mounted 'rw'(read-write).
        file_system_access_mode = 'rw'
    
    
        log = FileSystemInput(file_system_id=file_system_id,
                                        file_system_type=file_system_type,
                                        directory_path=file_system_directory_path,
                                        file_system_access_mode=file_system_access_mode)
    
        data_channels = {'train': train, 'log': log}
    
        mask_rcnn_estimator.fit(inputs=data_channels, 
                                job_name=mrcnn_trial.trial_name,
                                logs=True,  
                                experiment_config={"TrialName": mrcnn_trial.trial_name, 
                                               "TrialComponentDisplayName": "Training"},
                                wait=False)
    
        # sleep in between starting two trials
        time.sleep(2)

.. code:: ipython3

    search_expression = {
        "Filters":[
            {
                "Name": "DisplayName",
                "Operator": "Equals",
                "Value": "Training",
            },
            {
                "Name": "metrics.maskrcnn_loss/accuracy.max",
                "Operator": "LessThan",
                "Value": "1",
            }
        ],
    }

.. code:: ipython3

    from sagemaker.analytics import ExperimentAnalytics
    
    trial_component_analytics = ExperimentAnalytics(
        sagemaker_session=sagemaker_session,
        experiment_name=mrcnn_experiment.experiment_name,
        search_expression=search_expression,
        sort_by="metrics.maskrcnn_loss/accuracy.max",
        sort_order="Descending",
        parameter_names=['resnet_arch']
    )

.. code:: ipython3

    analytic_table = trial_component_analytics.dataframe()
    for col in analytic_table.columns: 
        print(col) 


.. code:: ipython3

    bbox_map=analytic_table[['resnet_arch',
                             'mAP(bbox)/small - Max', 
                             'mAP(bbox)/medium - Max', 
                             'mAP(bbox)/large - Max']]
    bbox_map

.. code:: ipython3

    segm_map=analytic_table[['resnet_arch',
                             'mAP(segm)/small - Max', 
                             'mAP(segm)/medium - Max', 
                             'mAP(segm)/large - Max']]
    segm_map


