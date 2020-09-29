Create a 3D Point Cloud Labeling Job with Amazon SageMaker Ground Truth
=======================================================================

This notebook will demonstrate how you can pre-process your 3D point
cloud input data to create an `object tracking labeling
job <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-object-tracking.html>`__
and include sensor and camera data for sensor fusion.

In object tracking, you are tracking the movement of an object (e.g., a
pedestrian on the side walk) while your point of reference (e.g., the
autonomous vehicle) is moving. When performing object tracking, your
data must be in a global reference coordinate system such as `world
coordinate
system <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-sensor-fusion-details.html#sms-point-cloud-world-coordinate-system>`__
because the ego vehicle itself is moving in the world. You can transform
point cloud data in local coordinates to the world coordinate system by
multiplying each of the points in a 3D frame with the extrinsic matrix
for the LiDAR sensor.

In this notebook, you will transform 3D frames from a local coordinate
system to a world coordinate system using extrinsic matrices. You will
use the KITTI dataset\ `1 <#The-Dataset-and-Input-Manifest-Files>`__\ ,
an open source autonomous driving dataset. The KITTI dataset provides an
extrinsic matrix for each 3D point cloud frame. You will use
`pykitti <https://github.com/utiasSTARS/pykitti>`__ and the `numpy
matrix multiplication
function <https://numpy.org/doc/1.18/reference/generated/numpy.matmul.html>`__
to multiple this matrix with each point in the frame to translate that
point to the world coordinate system used by the KITTI dataset.

You include camera image data and provide workers with more visual
information about the scene they are labeling. Through sensor fusion,
workers will be able to adjust labels in the 3D scene and in 2D images,
and label adjustments will be mirrored in the other view.

Ground Truth computes your sensor and camera extrinsic matrices for
sensor fusion using sensor and camera **pose data** - position and
heading. The KITTI raw dataset includes rotation matrix and translations
vectors for extrinsic transformations for each frame. This notebook will
demonstrate how you can extract **position** and **heading** from KITTI
rotation matrices and translations vectors using
`pykitti <https://github.com/utiasSTARS/pykitti>`__.

In summary, you will: \* Convert a dataset to a world coordinate system.
\* Learn how you can extract pose data from your LiDAR and camera
extrinsict matrices for sensor fusion. \* Create a sequence input
manifest file for an object tracking labeling job. \* Create an object
tracking labeling job. \* Preview the worker UI and tools provided by
Ground Truth.

Prerequisites
-------------

To run this notebook, you can simply execute each cell in order. To
understand what’s happening, you’ll need: \* An S3 bucket you can write
to – please provide its name in ``BUCKET``. The bucket must be in the
same region as this SageMaker Notebook instance. You can also change the
``EXP_NAME`` to any valid S3 prefix. All the files related to this
experiment will be stored in that prefix of your bucket. **Important:
you must attach the CORS policy to this bucket. See the next section for
more information**. \* Familiarity with the `Ground Truth 3D Point Cloud
Labeling
Job <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud.html>`__.
\* Familiarity with Python and `numpy <http://www.numpy.org/>`__. \*
Basic familiarity with `AWS
S3 <https://docs.aws.amazon.com/s3/index.html>`__. \* Basic
understanding of `AWS Sagemaker <https://aws.amazon.com/sagemaker/>`__.
\* Basic familiarity with `AWS Command Line Interface
(CLI) <https://aws.amazon.com/cli/>`__ – ideally, you should have it set
up with credentials to access the AWS account you’re running this
notebook from.

This notebook has only been tested on a SageMaker notebook instance. The
runtimes given are approximate. We used an ``ml.t2.medium`` instance in
our tests. However, you can likely run it on a local instance by first
executing the cell below on SageMaker and then copying the ``role``
string to your local copy of the notebook.

IMPORTANT: Attach CORS policy to your bucket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You must attach the following CORS policy to your S3 bucket for the
labeling task to render. To learn how to add a CORS policy to your S3
bucket, follow the instructions in `How do I add cross-domain resource
sharing with
CORS? <https://docs.aws.amazon.com/AmazonS3/latest/user-guide/add-cors-configuration.html>`__.
Paste the following policy in the CORS configuration editor:

::

   <?xml version="1.0" encoding="UTF-8"?>
   <CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
   <CORSRule>
       <AllowedOrigin>*</AllowedOrigin>
       <AllowedMethod>GET</AllowedMethod>
       <AllowedMethod>HEAD</AllowedMethod>
       <AllowedMethod>PUT</AllowedMethod>
       <MaxAgeSeconds>3000</MaxAgeSeconds>
       <ExposeHeader>Access-Control-Allow-Origin</ExposeHeader>
       <AllowedHeader>*</AllowedHeader>
   </CORSRule>
   <CORSRule>
       <AllowedOrigin>*</AllowedOrigin>
       <AllowedMethod>GET</AllowedMethod>
   </CORSRule>
   </CORSConfiguration>

.. code:: ipython3

    !pip install boto3==1.14.8
    !pip install -U botocore

.. code:: ipython3

    import boto3
    import time
    import pprint
    import json
    import sagemaker
    from sagemaker import get_execution_role
    from datetime import datetime, timezone
    
    pp = pprint.PrettyPrinter(indent=4)
    
    sagemaker_client = boto3.client('sagemaker')

.. code:: ipython3

    BUCKET = ''
    EXP_NAME = '' # Any valid S3 prefix.

.. code:: ipython3

    # Make sure the bucket is in the same region as this notebook.
    sess = sagemaker.session.Session()
    role = sagemaker.get_execution_role()
    region = boto3.session.Session().region_name
    s3 = boto3.client('s3')
    bucket_region = s3.head_bucket(Bucket=BUCKET)['ResponseMetadata']['HTTPHeaders']['x-amz-bucket-region']
    assert bucket_region == region, "Your S3 bucket {} and this notebook need to be in the same region.".format(BUCKET)

The Dataset and Input Manifest Files
------------------------------------

The dataset and resources used in this notebook are located in the
following Amazon S3 bucket:
https://aws-ml-blog.s3.amazonaws.com/artifacts/gt-point-cloud-demos/.

This bucket contains a single scene from the `KITTI
datasets <http://www.cvlibs.net/datasets/kitti/raw_data.php>`__. KITTI
created datasets for computer vision and machine learning research,
including for 2D and 3D object detection and object tracking. The
datasets are captured by driving around the mid-size city of Karlsruhe,
in rural areas and on highways.

[1] The KITTI dataset is subject to its own license. Please make sure
that any use of the dataset conforms to the license terms and
conditions.

Download and unzip data
-----------------------

.. code:: ipython3

    rm -rf sample_data*

.. code:: ipython3

    !wget https://aws-ml-blog.s3.amazonaws.com/artifacts/gt-point-cloud-demos/sample_data.zip

.. code:: ipython3

    !unzip -o sample_data

Let’s take a look at the sample_data folder. You’ll see that we have
images which can be used for sensor fusion, and point cloud data in
ASCII format (.txt files). We will use a script to convert this point
cloud data from the LiDAR sensor’s local coordinates to a world
coordinate system.

.. code:: ipython3

    !ls sample_data/2011_09_26/2011_09_26_drive_0005_sync/

.. code:: ipython3

    !ls sample_data/2011_09_26/2011_09_26_drive_0005_sync/oxts/data

Use the Kitti2GT script to convert the raw data to Ground Truth format
----------------------------------------------------------------------

You can use this script to do the following: \* Transform the KITTI
dataset with respect to the LIDAR sensor’s orgin in the first frame as
the world cooridinate system ( global frame of reference ), so that it
can be consumed by SageMaker Ground Truth. \* Extract pose data in world
coordinate system using the camera and LiDAR extrinsic matrices. You
will supply this pose data in your sequence file to enable sensor
fusion.

First, the script uses
`pykitti <https://github.com/utiasSTARS/pykitti>`__ python module to
load the KITTI raw data and calibrations. Let’s look at the two main
data-transformation functions of the script:

Data Transformation to a World Coordinate System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In general, multiplying a point in a LIDAR frame with a LIDAR extrinsic
matrix transforms it into world coordinates.

Using pykitti ``dataset.oxts[i].T_w_imu`` gives the lidar extrinsic
transform for the ``i``\ th frame. This matrix can be multiplied with
the points of the frame to convert it to a world frame using the numpy
matrix multiplication, function,
`matmul <https://numpy.org/doc/1.18/reference/generated/numpy.matmul.html>`__:
``matmul(lidar_transform_matrix, points)``. Let’s look at the function
that performs this transformation:

.. code:: ipython3

    # transform points from lidar to global frame using lidar_extrinsic_matrix
    def generate_transformed_pcd_from_point_cloud(points, lidar_extrinsic_matrix):
        tps = []
        for point in points:
            transformed_points = np.matmul(lidar_extrinsic_matrix, np.array([point[0], point[1], point[2], 1], dtype=np.float32).reshape(4,1)).tolist()
            if len(point) > 3 and point[3] is not None:
                tps.append([transformed_points[0][0], transformed_points[1][0], transformed_points[2][0], point[3]])
           
        return tps

If your point cloud data includes more than four elements for each
point, for example, (x,y,z) and r,g,b, modify the ``if`` statement in
the function above to ensure your r, g, b values are copied.

Extracting Pose Data from LiDAR and Camera Extrinsic for Sensor Fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For sensor fusion, you provide your extrinsic matrix in the form of
sensor-pose in terms of origin position (for translation) and heading in
quaternion (for rotation of the 3 axis). The following is an example of
the pose JSON you use in the sequence file.

::


   {
       "position": {
           "y": -152.77584902657554,
           "x": 311.21505956090624,
           "z": -10.854137529636024
         },
         "heading": {
           "qy": -0.7046155108831117,
           "qx": 0.034278837280808494,
           "qz": 0.7070617895701465,
           "qw": -0.04904659893885366
         }
   }

All of the positional coordinates (x, y, z) are in meters. All the pose
headings (qx, qy, qz, qw) are measured in Spatial Orientation in
Quaternion. Separately for each camera, you provide pose data extracted
from the extrinsic of that camera.

Both LIDAR sensors and and cameras have their own extrinsic matrices,
and they are used by SageMaker Ground Truth to enable the sensor fusion
feature. In order to project a label from 3D point cloud to camera image
plane Ground Truth needs to transform 3D points from LIDAR’s own
coordinate system to the camera’s coordinate system. This is typically
done by first transforming 3D points from LIDAR’s own coordinate to a
world coordinate system using the LIDAR extrinsic matrix. Then we use
the camera inverse extrinsic (world to camera) to transform the 3D
points from the world coordinate system we obtained in previous step
into camera image plane. If your 3D data is already transformed into
world coordinate system then the first transformation doesn’t have any
impact and label translation depends only on the camera extrinsic.

If you have a rotation matrix (made up of the axis rotations) and
translation vector (or origin) in world coordinate system instead of a
single 4x4 rigid transformation matrix, then you can directly use
rotation and translation to compute pose. For example:

.. code:: ipython3

    !python -m pip install --user numpy scipy

.. code:: ipython3

    import numpy as np
    
    rotation = [[ 9.96714314e-01, -8.09890350e-02,  1.16333982e-03],
     [ 8.09967396e-02,  9.96661051e-01, -1.03090934e-02],
     [-3.24531964e-04,  1.03694477e-02,  9.99946183e-01]]
     
    origin= [1.71104606e+00,
              5.80000039e-01,
              9.43144935e-01]
    
             
    from scipy.spatial.transform import Rotation as R
    # position is the origin
    position = origin 
    r = R.from_matrix(np.asarray(rotation))
    # heading in WCS using scipy 
    heading = r.as_quat()
    print(f"position:{position}\nheading: {heading}")

If you indeed have a 4x4 extrinsic transformation matrix then the
transformation matrix is just in the form of ``[R T; 0 0 0 1]`` where R
is the rotation matrix and T is the origin translation vector. That
means you can extract rotation matrix and translation vector from the
transformation matrix as follows

.. code:: ipython3

    import numpy as np
    
    transformation  = [[ 9.96714314e-01, -8.09890350e-02,  1.16333982e-03, 1.71104606e+00],
       [ 8.09967396e-02,  9.96661051e-01, -1.03090934e-02, 5.80000039e-01],
       [-3.24531964e-04,  1.03694477e-02,  9.99946183e-01, 9.43144935e-01],
       [0, 0, 0, 1]]
    
    transformation = np.array(transformation)
    rotation = transformation[0:3, 0:3]
    origin= transformation[0:3, 3]
             
    from scipy.spatial.transform import Rotation as R
    # position is the origin
    position = origin 
    r = R.from_matrix(np.asarray(rotation))
    # heading in WCS using scipy 
    heading = r.as_quat()
    print(f"position:{position}\nheading: {heading}")

For convenience, in this blog you will use
`pykitti <https://github.com/utiasSTARS/pykitti>`__ development kit to
load the raw data and calibrations. With pykitti you will extract sensor
pose in the world coordinate system from KITTI extrinsic which is
provided as a rotation matrix and translation vector in the raw
calibrations data. We will then format this pose data using the JSON
format required for the 3D point cloud sequence input manifest.

With pykitti the ``dataset.oxts[i].T_w_imu`` gives the LiDAR extrinsic
matrix ( lidar_extrinsic_transform ) for the i’th frame. Similarly, with
pykitti the camera extrinsic matrix ( camera_extrinsic_transform ) for
cam0 in i’th frame can be calculated by
``inv(matmul(dataset.calib.T_cam0_velo, inv(dataset.oxts[i].T_w_imu)))``
and this can be converted into heading and position for cam0.

In the script, the following functions are used to extract this pose
data from the LiDAR extrinsict and camera inverse extrinsic matrices.

.. code:: ipython3

    # utility to convert extrinsic matrix to pose heading quaternion and position
    def convert_extrinsic_matrix_to_trans_quaternion_mat(lidar_extrinsic_transform):
        position = lidar_extrinsic_transform[0:3, 3]
        rot = np.linalg.inv(lidar_extrinsic_transform[0:3, 0:3])
        quaternion= R.from_matrix(np.asarray(rot)).as_quat()
        trans_quaternions = {
            "translation": {
                "x": position[0],
                "y": position[1],
                "z": position[2]
            },
            "rotation": {
                "qx": quaternion[0],
                "qy": quaternion[1],
                "qz": quaternion[2],
                "qw": quaternion[3]
                }
        }
        return trans_quaternions
    
    def convert_camera_inv_extrinsic_matrix_to_trans_quaternion_mat(camera_extrinsic_transform):
        position = camera_extrinsic_transform[0:3, 3]
        rot = np.linalg.inv(camera_extrinsic_transform[0:3, 0:3])
        quaternion= R.from_matrix(np.asarray(rot)).as_quat()
        trans_quaternions = {
            "translation": {
                "x": position[0],
                "y": position[1],
                "z": position[2]
            },
            "rotation": {
                "qx": quaternion[0],
                "qy": quaternion[1],
                "qz": quaternion[2],
                "qw": -quaternion[3]
                }
        }
        return trans_quaternions

Generate a Sequence File
~~~~~~~~~~~~~~~~~~~~~~~~

After you’ve converted your data to a world coordinate system and
extracted sensor and camera pose data for sensor fusion, you can create
a sequence file. This is accomplished with the function
``convert_to_gt`` in the python script.

A **sequence** specifies a temporal series of point cloud frames. When a
task is created using a sequence file, all point cloud frames in the
sequence are sent to a worker to label. Your input manifest file will
contain a single sequence per line. To learn more about the sequence
input manifest format, see `Create a Point Cloud Frame Sequence Input
Manifest <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-multi-frame-input-data.html>`__.

If you want to use this script to create a frame input manifest file,
which is required for 3D point cloud object tracking and semantic
segmentation labeling jobs, you can modify the for-loop in the function
``convert_to_gt`` to produce the required content for
``source-ref-metadata``. To learn more about the frame input manifest
format, see `Create a Point Cloud Frame Input Manifest
File <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-single-frame-input-data.html>`__.

Now, let’s download the script and run it on the KITTI dataset to
process the data you’ll use for your labeling job.

.. code:: ipython3

    !wget https://aws-ml-blog.s3.amazonaws.com/artifacts/gt-point-cloud-demos/kitti2gt.py

.. code:: ipython3

    !pygmentize kitti2gt.py

Install pykitti
~~~~~~~~~~~~~~~

.. code:: ipython3

    !pip install pykitti

.. code:: ipython3

    from kitti2gt import *
    
    if(EXP_NAME == ''):
        s3loc = f's3://{BUCKET}/frames/'
    else:
        s3loc = f's3://{BUCKET}/{EXP_NAME}/frames/'
        
    convert_to_gt(basedir='sample_data',
                  date='2011_09_26',
                  drive='0005',
                  output_base='sample_data_out',
                 s3prefix = s3loc)

The following folders that will contain the data you’ll use for the
labeling job.

.. code:: ipython3

    !ls sample_data_out/

.. code:: ipython3

    !ls sample_data_out/frames

Now, you’ll upload the data to your bucket in S3.

.. code:: ipython3

    if(EXP_NAME == ''):
        !aws s3 cp sample_data_out/kitti-gt-seq.json s3://{BUCKET}/
    else:
        !aws s3 cp sample_data_out/kitti-gt-seq.json s3://{BUCKET}/{EXP_NAME}/


.. code:: ipython3

    if(EXP_NAME == ''):
        !aws s3 sync sample_data_out/frames/ s3://{BUCKET}/frames/
    else:
        !aws s3 sync sample_data_out/frames s3://{BUCKET}/{EXP_NAME}/frames/

.. code:: ipython3

    if(EXP_NAME == ''):
        !aws s3 sync sample_data_out/images/ s3://{BUCKET}/frames/images/
    else:
        !aws s3 sync sample_data_out/images s3://{BUCKET}/{EXP_NAME}/frames/images/

Write and Upload Multi-Frame Input Manifest File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, let’s create a **sequence input manifest file**. Each line in the
input manifest (in this demo, there is only one) will point to a
sequence file in your S3 bucket, ``BUCKET/EXP_NAME``.

.. code:: ipython3

    with open('manifest.json','w') as f:
        if(EXP_NAME == ''):
            json.dump({"source-ref": "s3://{}/kitti-gt-seq.json".format(BUCKET)},f)
        else:
            json.dump({"source-ref": "s3://{}/{}/kitti-gt-seq.json".format(BUCKET,EXP_NAME)},f)

Our manifest file is one line long, and identifies a single sequence
file in your S3 bucket.

.. code:: ipython3

    !cat manifest.json

.. code:: ipython3

    if(EXP_NAME == ''):
        !aws s3 cp manifest.json s3://{BUCKET}/
        input_manifest_s3uri = f's3://{BUCKET}/manifest.json'
    else:
        !aws s3 cp manifest.json s3://{BUCKET}/{EXP_NAME}/
        input_manifest_s3uri = f's3://{BUCKET}/{EXP_NAME}/manifest.json'
    input_manifest_s3uri

Create a Labeling Job
---------------------

In the following cell, we specify object tracking as our `3D Point Cloud
Task
Type <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-task-types.html>`__.

.. code:: ipython3

    task_type = "3DPointCloudObjectTracking"

Identify Resources for Labeling Job
-----------------------------------

Specify Human Task UI ARN
~~~~~~~~~~~~~~~~~~~~~~~~~

The following will be used to identify the HumanTaskUiArn. When you
create a 3D point cloud labeling job, Ground Truth provides a worker UI
that is specific to your task type. You can learn more about this UI and
the assistive labeling tools that Ground Truth provides for Object
Tracking on the `Object Tracking task type
page <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-object-tracking.html>`__.

.. code:: ipython3

    ## Set up human_task_ui_arn map, to be used in case you chose UI_CONFIG_USE_TASK_UI_ARN
    ## Supported for GA
    ## Set up human_task_ui_arn map, to be used in case you chose UI_CONFIG_USE_TASK_UI_ARN
    human_task_ui_arn = f'arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectTracking'
    human_task_ui_arn

Label Category Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your label category configuration file is used to specify labels, or
classes, for your labeling job.

When you use the object detection or object tracking task types, you can
also include **label attributes** in your `label category configuration
file <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-label-category-config.html>`__.
Workers can assign one or more attributes you provide to annotations to
give more information about that object. For example, you may want to
use the attribute *occluded* to have workers identify when an object is
partially obstructed.

Let’s look at an example of the label category configuration file for an
object detection or object tracking labeling job.

.. code:: ipython3

    !wget https://aws-ml-blog.s3.amazonaws.com/artifacts/gt-point-cloud-demos/label-category-config/label-category.json

.. code:: ipython3

    with open('label-category.json', 'r') as j:
        json_data = json.load(j)
        print("\nA label category configuration file: \n\n",json.dumps(json_data, indent=4, sort_keys=True))

.. code:: ipython3

    if(EXP_NAME == ''):
        !aws s3 cp label-category.json s3://{BUCKET}/label-category.json
        label_category_config_s3uri = f's3://{BUCKET}/label-category.json'
    else:
        !aws s3 cp label-category.json s3://{BUCKET}/{EXP_NAME}/label-category.json
        label_category_config_s3uri = f's3://{BUCKET}/{EXP_NAME}/label-category.json'
    label_category_config_s3uri

To learn more about the label category configuration file, see `Create a
Label Category Configuration
File <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-label-category-config.html>`__

Run the following cell to identify the labeling category configuration
file.

Set up a private work team
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to preview the worker task UI, create a private work team
and add yourself as a worker.

If you have already created a private workforce, follow the instructions
in `Add or Remove
Workers <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-management-private-console.html#add-remove-workers-sm>`__
to add yourself to the work team you use to create a lableing job.

Create a private workforce and add yourself as a worker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create and manage your private workforce, you can use the **Labeling
workforces** page in the Amazon SageMaker console. When following the
instructions below, you will have the option to create a private
workforce by entering worker emails or importing a pre-existing
workforce from an Amazon Cognito user pool. To import a workforce, see
`Create a Private Workforce (Amazon Cognito
Console) <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-create-private-cognito.html>`__.

To create a private workforce using worker emails:

-  Open the Amazon SageMaker console at
   https://console.aws.amazon.com/sagemaker/.

-  In the navigation pane, choose **Labeling workforces**.

-  Choose Private, then choose **Create private team**.

-  Choose **Invite new workers by email**.

-  Paste or type a list of up to 50 email addresses, separated by
   commas, into the email addresses box.

-  Enter an organization name and contact email.

-  Optionally choose an SNS topic to subscribe the team to so workers
   are notified by email when new Ground Truth labeling jobs become
   available.

-  Click the **Create private team** button.

After you import your private workforce, refresh the page. On the
Private workforce summary page, you’ll see your work team ARN. Enter
this ARN in the following cell.

.. code:: ipython3

    ##Use Beta Private Team till GA
    workteam_arn = ''

Task Time Limits
^^^^^^^^^^^^^^^^

3D point cloud annotation jobs can take workers hours. Workers will be
able to save their work as they go, and complete the task in multiple
sittings. Ground Truth will also automatically save workers’ annotations
periodically as they work.

When you configure your task, you can set the total amount of time that
workers can work on each task when you create a labeling job using
``TaskTimeLimitInSeconds``. The maximum time you can set for workers to
work on tasks is 7 days. The default value is 3 days. It is recommended
that you create labeling tasks that can be completed within 12 hours.

If you set ``TaskTimeLimitInSeconds`` to be greater than 8 hours, you
must set ``MaxSessionDuration`` for your IAM execution to at least 8
hours. To update your execution role’s ``MaxSessionDuration``, use
`UpdateRole <https://docs.aws.amazon.com/IAM/latest/APIReference/API_UpdateRole.html>`__
or use the `IAM
console <https://docs.aws.amazon.com/IAM/latest/UserGuide/roles-managingrole-editing-console.html#roles-modify_max-session-duration>`__.
You an identify the name of your role at the end of your role ARN.

.. code:: ipython3

    #See your execution role ARN. The role name is located at the end of the ARN. 
    role

.. code:: ipython3

    ac_arn_map = {'us-west-2': '081040173940',
                  'us-east-1': '432418664414',
                  'us-east-2': '266458841044',
                  'eu-west-1': '568282634449',
                  'ap-northeast-1': '477331159723'}
    
    prehuman_arn = 'arn:aws:lambda:{}:{}:function:PRE-{}'.format(region, ac_arn_map[region],task_type)
    acs_arn = 'arn:aws:lambda:{}:{}:function:ACS-{}'.format(region, ac_arn_map[region],task_type) 

Set Up HumanTaskConfig
----------------------

``HumanTaskConfig`` is used to specify your work team, and configure
your labeling job tasks. Modify the following cell to identify a
``task_description``, ``task_keywords``, ``task_title``, and
``job_name``.

.. code:: ipython3

    from datetime import datetime
    ## Set up Human Task Config
    
    ## Modify the following
    task_description = 'add a task description here'
    #example keywords
    task_keywords = ['lidar', 'pointcloud']
    #add a task title
    task_title = 'Add a Task Title Here - This is Displayed to Workers'
    #add a job name to identify your labeling job
    job_name = 'add-job-name'
    
    human_task_config = {
          "AnnotationConsolidationConfig": {
            "AnnotationConsolidationLambdaArn": acs_arn,
          },
            "UiConfig": { 
               "HumanTaskUiArn": human_task_ui_arn,
          },
          "WorkteamArn": workteam_arn,
          "PreHumanTaskLambdaArn": prehuman_arn,
          "MaxConcurrentTaskCount": 200, # 200 images will be sent at a time to the workteam.
          "NumberOfHumanWorkersPerDataObject": 1, # One worker will work on each task
          "TaskAvailabilityLifetimeInSeconds": 18000, # Your workteam has 5 hours to complete all pending tasks.
          "TaskDescription": task_description,
          "TaskKeywords": task_keywords,
          "TaskTimeLimitInSeconds": 3600, # Each seq must be labeled within 1 hour.
          "TaskTitle": task_title
        }


.. code:: ipython3

    print(json.dumps(human_task_config, indent=4, sort_keys=True))

Set up Create Labeling Request
------------------------------

The following formats your labeling job request. For Object Tracking
task types, the ``LabelAttributeName`` must end in ``-ref``.

.. code:: ipython3

    if(EXP_NAME == ''):
        s3_output_path = f's3://{BUCKET}'
    else:
        s3_output_path = f's3://{BUCKET}/{EXP_NAME}'
    s3_output_path

.. code:: ipython3

    ## Set up Create Labeling Request
    
    labelAttributeName = job_name + "-ref"
    
    if task_type == "3DPointCloudObjectDetection" or task_type == "Adjustment3DPointCloudObjectDetection":
        labelAttributeName = job_name
    
    
    ground_truth_request = {
            "InputConfig" : {
              "DataSource": {
                "S3DataSource": {
                  "ManifestS3Uri": input_manifest_s3uri,
                }
              },
              "DataAttributes": {
                "ContentClassifiers": [
                  "FreeOfPersonallyIdentifiableInformation",
                  "FreeOfAdultContent"
                ]
              },  
            },
            "OutputConfig" : {
              "S3OutputPath": s3_output_path,
            },
            "HumanTaskConfig" : human_task_config,
            "LabelingJobName": job_name,
            "RoleArn": role, 
            "LabelAttributeName": labelAttributeName,
            "LabelCategoryConfigS3Uri": label_category_config_s3uri
        }
    
    print(json.dumps(ground_truth_request, indent=4, sort_keys=True))

Call CreateLabelingJob
----------------------

.. code:: ipython3

    sagemaker_client.create_labeling_job(**ground_truth_request)
    print(f'Labeling Job Name: {job_name}')

Check Status of Labeling Job
----------------------------

.. code:: ipython3

    ## call describeLabelingJob
    describeLabelingJob = sagemaker_client.describe_labeling_job(
        LabelingJobName=job_name
    )
    print(describeLabelingJob)

Start Working on tasks
----------------------

When you add yourself to a private work team, you recieve an email
invitation to access the worker portal that looks similar to this
`image <https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2020/04/16/a2i-critical-documents-26.gif>`__.
Use this invitation to sign in to the protal and view your 3D point
cloud annotation tasks. Tasks may take up to 10 minutes to show up the
worker portal.

Once you are done working on the tasks, click **Submit**.

View Output Data
~~~~~~~~~~~~~~~~

Once you have completed all of the tasks, you can view your output data
in the S3 location you specified in ``OutputConfig``.

To read more about Ground Truth output data format for your task type,
see `Output
Data <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-data-output.html#sms-output-point-cloud-object-tracking>`__.

Acknowledgments
===============

We would like to thank the KITTI team for letting us use this dataset to
demonstrate how to prepare your 3D point cloud data for use in SageMaker
Ground Truth.

