Create a 3D Point Cloud Labeling Job with Amazon SageMaker Ground Truth
=======================================================================

This sample notebook takes you through an end-to-end workflow to
demonstrate the functionality of SageMaker Ground Truth 3D point cloud
built-in task types.

What is a Point Cloud
~~~~~~~~~~~~~~~~~~~~~

A point cloud frame is defined as a collection of 3D points describing a
3D scene. Each point is described using three coordinates, x, y, and z.
To add color and/or variations in point intensity to the point cloud,
points may have additional attributes, such as i for intensity or values
for the red (r), green (g), and blue (b) color channels (8-bit). All of
the positional coordinates (x, y, z) are in meters. Point clouds are
most commonly created from data that was collected by scanning the real
world through various scanning methods, such as laser scanning and
photogrammetry. Ground Truth currently also supports sensor fusion with
video camera data.

3D Point Cloud Built in Task Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use Ground Truth 3D point cloud labeling built-in task types to
annotate 3D point cloud data. The following list briefly describes each
task type. See `3D Point Cloud Task
types <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-task-types.html>`__
for more information.

-  3D point cloud object detection – Use this task type when you want
   workers to indentify the location of and classify objects in a 3D
   point cloud by drawing 3D cuboids around objects. You can include one
   or more attributes for each class (label) you provide.

-  3D point cloud object tracking – Use this task type when you want
   workers to track the trajectory of an object across a sequence of 3D
   point cloud frames. For example, you can use this task type to ask
   workers to track the movement of vehicles across a sequence of point
   cloud frames.

-  3D point cloud semantic segmentation – Use this task type when you
   want workers to create a point-level semantic segmentation mask by
   painting objects in a 3D point cloud using different colors where
   each color is assigned to one of the classes you specify.

You can use the Adjustment task types to verify and adjust annotations
created for the three task types above.

Sensor Fusion
~~~~~~~~~~~~~

One of the important features of this product is sensor fusion, which
fuses the inputs of multiple sensors to provide labelers with a better
understanding of the 3D scene.

When you create a 3D point cloud labeling job, you can optionally
provide camera data for sensor fusion. Ground Truth uses your camera
data to include images in the worker UI. These images give workers more
visual information about scenes in the 3D point cloud visualization and
can be used to annotate (draw 3D cuboids or paint) objects. When a
worker adjusts annotation around an object in either the 2D image or the
3D point cloud, the adjustments shows up in the other medium. This
tutorial will demonstrate how you can include image data in your input
manifest file for sensor fusion.

This Demo
---------

In this demo, you’ll start by inspecting the input data and manifest
files used to in the demo. Then, you will specifying resources needed to
create a labeling job. In this step, you’ll have the option to make
yourself a worker on a private work team that you send the labeling job
tasks to. This will allow you can preview and interact with the worker
UI. Finally, you’ll configure and send your CreateLabelingJob request.

After your labeling job has been created, you can check your labeling
job status. When the job is completed, you can view the output in Amazon
S3.

Prerequisites
-------------

To run this notebook, you can simply execute each cell in order. To
understand what’s happening, you’ll need:

-  An S3 bucket you can write to – please provide its name in
   ``BUCKET``. The bucket must be in the same region as this SageMaker
   Notebook instance. You can also change the ``EXP_NAME`` to any valid
   S3 prefix. All the files related to this experiment will be stored in
   that prefix of your bucket. **Important: you must attach the CORS
   policy to this bucket. See the next section for more information**.
-  Familiarity with the `Ground Truth 3D Point Cloud Labeling
   Job <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud.html>`__.
-  Familiarity with Python and `numpy <http://www.numpy.org/>`__.
-  Basic familiarity with `AWS
   S3 <https://docs.aws.amazon.com/s3/index.html>`__.
-  Basic understanding of `AWS
   Sagemaker <https://aws.amazon.com/sagemaker/>`__.
-  Basic familiarity with `AWS Command Line Interface
   (CLI) <https://aws.amazon.com/cli/>`__ – ideally, you should have it
   set up with credentials to access the AWS account you’re running this
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
    import botocore
    import time
    import pprint
    import json
    import sagemaker
    from sagemaker import get_execution_role
    from datetime import datetime, timezone
    
    pp = pprint.PrettyPrinter(indent=4)
    
    sess = sagemaker.session.Session()
    role = sagemaker.get_execution_role()
    region = boto3.session.Session().region_name
    
    sagemaker_client = boto3.client('sagemaker')
    s3 = boto3.client('s3')
    iam = boto3.client('iam')

.. code:: ipython3

    ### The following cell will set up your S3 bucket and execution role

.. code:: ipython3

    BUCKET = ''
    EXP_NAME = '' # Any valid S3 prefix.

.. code:: ipython3

    # Make sure the bucket is in the same region as this notebook.
    bucket_region = s3.head_bucket(Bucket=BUCKET)['ResponseMetadata']['HTTPHeaders']['x-amz-bucket-region']
    assert bucket_region == region, "Your S3 bucket {} and this notebook need to be in the same region.".format(BUCKET)

Copy and modify files from the sample bucket
--------------------------------------------

The sample files for this demo are in a public bucket to provide you
with the inputs to try this demo. In order for this demo to work, we
will need to copy these files to the bucket you specified in ``BUCKET``
so that there are in a place where you have read/write access.
Additionally, we will have to update the file paths that refer to our
public demo bucket to the bucket you specified above.

.. code:: bash

    %%bash -s "$BUCKET"
    mkdir -p sample_files
    
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/manifests sample_files/manifests --quiet --recursive
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/label-category-config sample_files/label-category-config --quiet --recursive
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/output sample_files/output --quiet --recursive
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/sequences sample_files/sequences --quiet --recursive
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/0.txt sample_files/frames/0.txt --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_0_camera_0.jpg sample_files/frames/images/frame_0_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_1_camera_0.jpg sample_files/frames/images/frame_1_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_2_camera_0.jpg sample_files/frames/images/frame_2_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_3_camera_0.jpg sample_files/frames/images/frame_3_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_4_camera_0.jpg sample_files/frames/images/frame_4_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_5_camera_0.jpg sample_files/frames/images/frame_5_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_6_camera_0.jpg sample_files/frames/images/frame_6_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_7_camera_0.jpg sample_files/frames/images/frame_7_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_8_camera_0.jpg sample_files/frames/images/frame_8_camera_0.jpg --quiet
    aws s3 cp s3://aws-ml-blog/artifacts/gt-point-cloud-demos/frames/images/frame_9_camera_0.jpg sample_files/frames/images/frame_9_camera_0.jpg --quiet
        
    find ./sample_files/ -type f -name "*.json" -print0 | xargs -0 sed -i -e "s/aws-ml-blog/$1/g"
    
    aws s3 cp ./sample_files/ s3://$1/artifacts/gt-point-cloud-demos/ --quiet --recursive

The Dataset and Input Manifest Files
------------------------------------

The dataset and resources used in this notebook are located in the
following Amazon S3 bucket.
s3://aws-ml-blog/artifacts/gt-point-cloud-demos.

This bucket contains our input manifest files and our input data: 3D
point cloud frame files and images for sensor fusion. First, we’ll
inspect the input data and input manifest files. In the next section,
you will select your labeling job type, and identify resources required
to create a labeling job.

.. code:: ipython3

    !aws s3 ls s3://$BUCKET/artifacts/gt-point-cloud-demos/

Depending on the task type that you choose in the following section, you
will use a **manifest with single-frame per task**, or *frame input
manifest* or **manifest with multi-frame sequence per task**, or a
*sequence input manifest*. To learn more about the types of 3D Point
Cloud input manfiest files, see `3D Point Cloud Input
Data <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-input-data.html>`__.

Input Manifest File With Single Frame Per Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you use a frame input manifest for 3D point cloud object detection
and semantic segmentation task types, each line in the input manifest
will identify the location of a single point cloud file in Amazon S3.
When a task is created, workers will be asked to classify or add a
segmentation mask to objects in that frame (depending on the task type).

Let’s look at the single-frame input manfiest. You’ll see that this
manifest file contains the location of a point cloud file in
``source-ref``, as well as the pose of the vehicle used to collect the
data (ego-vehicle), image pose information and other image data used for
sensor fusion. See `Create a Point Cloud Frame Input Manifest
File <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-single-frame-input-data.html>`__
to learn more about these parameters.

.. code:: ipython3

    !wget https://s3.amazonaws.com/aws-ml-blog/artifacts/gt-point-cloud-demos/manifests/SingleFrame-manifest.json -O SingleFrame-manifest.json

.. code:: ipython3

    print("\nThe single-frame input manifest file:")
    with open('SingleFrame-manifest.json', 'r') as j:
        json_data = json.load(j)
        print("\n",json.dumps(json_data, indent=4, sort_keys=True))

The point cloud data in the file, ``0.txt``, identified in the manfiest
above is in ASCII format. Each line in the point cloud file contains
information about a single point. The first three values are x, y, and z
location coordinates, and the last element is the pixel intensity. To
learn more about this raw data format, see `ASCII
Format <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-raw-data-types.html#sms-point-cloud-raw-data-ascii-format>`__.

.. code:: ipython3

    !wget https://s3.amazonaws.com/aws-ml-blog/artifacts/gt-point-cloud-demos/frames/0.txt -O 0.txt

.. code:: ipython3

    frame = open('0.txt')
    print("\nA single line from the point cloud file with x, y, z and pixel intensity values: \n")
    frame.readline()

Input Manifest File With Multi-Frame Sequence Per Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you chooose a sequence input manifest file, each line in the input
manifest will point to a *sequence file* in Amazon S3. A sequence
specifies a temporal series of point cloud frames. When a task is
created using a sequence file, all point cloud frames in the sequence
are sent to a worker to label. Workers can navigate back and forth
between and annotate (with 3D cuboids) the sequence of frames to track
the trajectory of objects across frames.

Let’s look at the sequence input manifest file. You’ll see that this
input manifest contains the location of a single sequence file.

.. code:: ipython3

    !wget https://s3.amazonaws.com/aws-ml-blog/artifacts/gt-point-cloud-demos/manifests/OT-manifest-10-frame.json -O OT-manifest-10-frame.json

.. code:: ipython3

    print("\nThe multi-frame input manifest file:")
    with open('OT-manifest-10-frame.json', 'r') as j:
        json_data = json.load(j)
        print("\n",json.dumps(json_data, indent=4, sort_keys=True))

Let’s look at the sequence file, seq1.json. You will see that this
single sequence file contains the location of 10 frames, as well as pose
information on the vehicle (ego-vehicle) and camera. See `Create a Point
Cloud Frame Sequence Input
Manifest <http://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-multi-frame-input-data.html>`__
to learn more about these parameters.

.. code:: ipython3

    !wget https://s3.amazonaws.com/aws-ml-blog/artifacts/gt-point-cloud-demos/sequences/seq2.json -O seq1.json

.. code:: ipython3

    with open('seq1.json', 'r') as j:
        json_data = json.load(j)
        print("\nA single sequence file: \n\n",json.dumps(json_data, indent=4, sort_keys=True))

Select a 3D Point Cloud Labeling Job Task Type
----------------------------------------------

In the following cell, select a `3D Point Cloud Task
Type <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-task-types.html>`__
by sepcifying a value for ``task_type``.

.. code:: ipython3

    ## Choose from following:
    ## 3DPointCloudObjectDetection
    ## 3DPointCloudObjectTracking
    ## 3DPointCloudSemanticSegmentation
    ## Adjustment3DPointCloudObjectDetection
    ## Adjustment3DPointCloudObjectTracking
    ## Adjustment3DPointCloudSemanticSegmentation
    
    task_type = "3DPointCloudObjectTracking"

Identify Resources for Labeling Job
-----------------------------------

The following will be used to select the HumanTaskUiArn. When you create
a 3D point cloud labeling job, Ground Truth provides the worker task UI.
The following cell identifies the correct HumanTaskUiArn to use a worker
UI that is specific to your task type. You can see examples of the
worker UIs on the `3D Point Cloud Task
Type <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-task-types.html>`__
pages.

.. code:: ipython3

    ## Set up human_task_ui_arn map
    
    human_task_ui_arn_map = {'3DPointCloudObjectDetection': f'arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectDetection',
                  '3DPointCloudObjectTracking': f'arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectTracking',
                  '3DPointCloudSemanticSegmentation': f'arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudSemanticSegmentation',
                  'Adjustment3DPointCloudObjectDetection': f'arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectDetection',
                  'Adjustment3DPointCloudObjectTracking': f'arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudObjectTracking',
                  'Adjustment3DPointCloudSemanticSegmentation': f'arn:aws:sagemaker:{region}:394669845002:human-task-ui/PointCloudSemanticSegmentation'}

Input Data and Input Manifest File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following task types (and associated adjustment labeling jobs)
require the following types of input manifest files.

-  3D point cloud object detection – frame input manifest
-  3D point cloud semantic segmentation – frame input manifest
-  3D point cloud object tracking – sequence frame input manifest

Run the following to identify an input manifest file for the task type
you selected in the previous section.

.. code:: ipython3

    ## Set up manifest_s3_uri_map, to be used to set up Input ManifestS3Uri
    
    manifest_s3_uri_map = {'3DPointCloudObjectDetection': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/manifests/SingleFrame-manifest.json',
                  '3DPointCloudObjectTracking': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/manifests/OT-manifest-10-frame.json',
                  '3DPointCloudSemanticSegmentation': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/manifests/SS-manifest.json',
                  'Adjustment3DPointCloudObjectDetection': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/manifests/OD-adjustment-manifest.json',
                  'Adjustment3DPointCloudObjectTracking': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/manifests/OT-adjustment-manifest.json',
                  'Adjustment3DPointCloudSemanticSegmentation': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/manifests/SS-audit-manifest-5-17.json'}

Label Category Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your label category configuration file is used to specify labels, or
classes, for your labeling job.

When you use the object detection or object tracking task types, you can
also include `label category
attributes <http://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-general-information.html#sms-point-cloud-worker-task-ui>`__
in your label category configuration file. Workers can assign one or
more attributes you provide to annotations to give more information
about that object. For example, you may want to use the attribute
*occluded* to have workers identify when an object is partially
obstructed.

Let’s look at an example of the label category configuration file for an
object detection or object tracking labeling job.

.. code:: ipython3

    !wget https://s3.amazonaws.com/aws-ml-blog/artifacts/gt-point-cloud-demos/label-category-config/label-category.json -O label-category.json

.. code:: ipython3

    with open('label-category.json', 'r') as j:
        json_data = json.load(j)
        print("\nA label category configuration file: \n\n",json.dumps(json_data, indent=4, sort_keys=True))

To learn more about the label category configuration file, see `Create a
Label Category Configuration
File <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-point-cloud-label-category-config.html>`__.

Run the following cell to identify the labeling category configuration
file.

.. code:: ipython3

    label_category_file_s3_uri_map = {
        '3DPointCloudObjectDetection': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/label-category-config/label-category.json',
          '3DPointCloudObjectTracking': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/label-category-config/label-category.json',
          '3DPointCloudSemanticSegmentation': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/label-category-config/label-category.json',
          'Adjustment3DPointCloudObjectDetection': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/label-category-config/od-adjustment-label-categories-file.json',
          'Adjustment3DPointCloudObjectTracking': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/label-category-config/ot-adjustment-label-categories-file.json',
          'Adjustment3DPointCloudSemanticSegmentation': f's3://{BUCKET}/artifacts/gt-point-cloud-demos/label-category-config/SS-audit-5-17-updated-manually-created-label-categories-file.json'
    }

.. code:: ipython3

    #You can use this to identify your labeling job by appending these abbreviations to your lableing job name. 
    name_abbreviation_map = {
         '3DPointCloudObjectDetection': 'OD',
          '3DPointCloudObjectTracking': 'OT',
          '3DPointCloudSemanticSegmentation': 'SS',
          'Adjustment3DPointCloudObjectDetection': 'OD-ADJ',
          'Adjustment3DPointCloudObjectTracking': 'OT-ADJ',
          'Adjustment3DPointCloudSemanticSegmentation': "SS-ADJ"
    }

Set up Human Task Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HumanTaskConfig`` is used to specify your work team, and configure
your labeling job tasks.

If you want to preview the worker task UI, create a private work team
and add yourself as a worker.

If you have already created a private workforce, follow the instructions
in `Add or Remove
Workers <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-management-private-console.html#add-remove-workers-sm>`__
to add yourself to the work team you use to create a lableing job.

Create a Private Workforce and Add Yourself as a Worker
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

    workteam_arn = ''

Task Time Limits
^^^^^^^^^^^^^^^^

3D point cloud annotation jobs can take workers hours or days to
complete. Workers will be able to start your labeling task, save their
work as they go, and complete the task in multiple sittings. Ground
Truth will also automatically save workers’ annotations periodically as
they work.

When you configure your task, you can set the total amount of time that
workers can work on each task when you create a labeling job using
``TaskTimeLimitInSeconds``. The maximum time you can set for workers to
work on tasks is 7 days. The default value is 3 days.

If you set ``TaskTimeLimitInSeconds`` to be greater than 8 hours, you
must set ``MaxSessionDuration`` for your IAM execution to at least 8
hours. To see how to update this value for your IAM role, see `Modifying
a
Role <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_manage_modify.html>`__
in the IAM User Guide, choose your preferred method to modify the role,
and then follow the steps in `Modifying a Role Maximum Session
Duration <https://docs.aws.amazon.com/IAM/latest/UserGuide/roles-managingrole-editing-console.html#roles-modify_max-session-duration>`__.

.. code:: ipython3

    #See your execution role ARN. The role name is located at the end of the ARN. 
    print(role)

.. code:: ipython3

    ac_arn_map = {'us-west-2': '081040173940',
                  'us-east-1': '432418664414',
                  'us-east-2': '266458841044',
                  'eu-west-1': '568282634449',
                  'ap-northeast-1': '477331159723'}
    
    prehuman_arn = 'arn:aws:lambda:{}:{}:function:PRE-{}'.format(region, ac_arn_map[region],task_type)
    acs_arn = 'arn:aws:lambda:{}:{}:function:ACS-{}'.format(region, ac_arn_map[region],task_type) 

.. code:: ipython3

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
          "WorkteamArn": workteam_arn,
          "PreHumanTaskLambdaArn": prehuman_arn,
          "MaxConcurrentTaskCount": 200, # 200 data objects (frames for OD and SS or sequences for OT) will be sent at a time to the workteam.
          "NumberOfHumanWorkersPerDataObject": 1, # One worker will work on each task
          "TaskAvailabilityLifetimeInSeconds": 18000, # Your workteam has 5 hours to complete all pending tasks.
          "TaskDescription": task_description,
          "TaskKeywords": task_keywords,
          "TaskTimeLimitInSeconds": 3600, # Each seq/frame must be labeled within 1 hour.
          "TaskTitle": task_title
        }
    
    
    human_task_config['UiConfig'] = {
        "HumanTaskUiArn":"{}".format(human_task_ui_arn_map[task_type])
    }

.. code:: ipython3

    print(json.dumps(human_task_config, indent=4, sort_keys=True))

Set up Create Labeling Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following formats your labeling job request. For 3D point cloud
object tracking and semantic segmentation task types, the
``LabelAttributeName`` must end in ``-ref``. For other task types, the
label attribute name may not end in ``-ref``.

.. code:: ipython3

    ## Set up Create Labeling Request
    
    labelAttributeName = job_name + "-ref"
    
    if task_type == "3DPointCloudObjectDetection" or task_type == "Adjustment3DPointCloudObjectDetection":
        labelAttributeName = job_name
    
    
    ground_truth_request = {
            "InputConfig" : {
              "DataSource": {
                "S3DataSource": {
                  "ManifestS3Uri": '{}'.format(manifest_s3_uri_map[task_type]),
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
              "S3OutputPath": f's3://{BUCKET}/{EXP_NAME}/output/',
            },
            "HumanTaskConfig" : human_task_config,
            "LabelingJobName": job_name,
            "RoleArn": role, 
            "LabelAttributeName": labelAttributeName,
            "LabelCategoryConfigS3Uri": label_category_file_s3_uri_map[task_type]
        }
    
    print(json.dumps(ground_truth_request, indent=4, sort_keys=True))

Call CreateLabelingJob
~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

When you add yourself to a private work team, you recieve an email
invitation to access the worker portal. Use this invitation to sign in
to the protal and view your 3D point cloud annotation tasks. Tasks may
take up to 10 minutes to show up the worker portal.

Once you are done working on the tasks, click **Submit**.

View Output Data
~~~~~~~~~~~~~~~~

Once you have completed all of the tasks, you can view your output data
in the S3 location you specified in ``OutputConfig``.

To read more about Ground Truth output data format for your task type,
see `Output
Data <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-data-output.html>`__.
