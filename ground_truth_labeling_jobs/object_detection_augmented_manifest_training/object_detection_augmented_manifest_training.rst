Training Object Detection Models in SageMaker with Augmented Manifests
======================================================================

This notebook demonstrates the use of an “augmented manifest” to train
an object detection machine learning model with AWS SageMaker.

Setup
-----

Here we define S3 file paths for input and output data, the training
image containing the semantic segmentation algorithm, and instantiate a
SageMaker session.

.. code:: ipython3

    import boto3
    import re
    import sagemaker
    from sagemaker import get_execution_role
    import time
    from time import gmtime, strftime
    import json
    
    role = get_execution_role()
    sess = sagemaker.Session()
    s3 = boto3.resource('s3')
    
    training_image = sagemaker.amazon.amazon_estimator.get_image_uri(boto3.Session().region_name, 'object-detection', repo_version='latest')

Required Inputs
~~~~~~~~~~~~~~~

*Be sure to edit the file names and paths below for your own use!*

.. code:: ipython3

    augmented_manifest_filename_train = 'augmented-manifest-train.manifest' # Replace with the filename for your training data.
    augmented_manifest_filename_validation = 'augmented-manifest-validation.manifest' # Replace with the filename for your validation data.
    bucket_name = "ground-truth-augmented-manifest-demo" # Replace with your bucket name.
    s3_prefix = 'object-detection' # Replace with the S3 prefix where your data files reside.
    s3_output_path = 's3://{}/output'.format(bucket_name) # Replace with your desired output directory.

The setup section concludes with a few more definitions and constants.

.. code:: ipython3

    # Defines paths for use in the training job request.
    s3_train_data_path = 's3://{}/{}/{}'.format(bucket_name, s3_prefix, augmented_manifest_filename_train)
    s3_validation_data_path = 's3://{}/{}/{}'.format(bucket_name, s3_prefix, augmented_manifest_filename_validation)
    
    print("Augmented manifest for training data: {}".format(s3_train_data_path))
    print("Augmented manifest for validation data: {}".format(s3_validation_data_path))

Understanding the Augmented Manifest format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Augmented manifests provide two key benefits. First, the format is
consistent with that of a labeling job output manifest. This means that
you can take your output manifests from a Ground Truth labeling job and,
whether the dataset objects were entirely human-labeled, entirely
machine-labeled, or anything in between, and use them as inputs to
SageMaker training jobs - all without any additional translation or
reformatting! Second, the dataset objects and their corresponding ground
truth labels/annotations are captured *inline*. This effectively reduces
the required number of channels by half, since you no longer need one
channel for the dataset objects alone and another for the associated
ground truth labels/annotations.

The augmented manifest format is essentially the `json-lines
format <http://jsonlines.org/>`__, also called the new-line delimited
JSON format. This format consists of an arbitrary number of well-formed,
fully-defined JSON objects, each on a separate line. Augmented manifests
must contain a field that defines a dataset object, and a field that
defines the corresponding annotation. Let’s look at an example for an
object detection problem.

The Ground Truth output format is discussed more fully for various types
of labeling jobs in the `official
documenation <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-data-output.html>`__.

{“source-ref”: “s3://bucket_name/path_to_a_dataset_object.jpeg”,
“labeling-job-name”:
{“annotations”:[{“class_id”:“0”,\ ``<bounding box dimensions>``}],“image_size”:[{``<image size simensions>``}]}

The first field will always be either ``source`` our ``source-ref``.
This defines an individual dataset object. The name of the second field
depends on whether the labeling job was created from the SageMaker
console or through the Ground Truth API. If the job was created through
the console, then the name of the field will be the labeling job name.
Alternatively, if the job was created through the API, then this field
maps to the ``LabelAttributeName`` parameter in the API.

The training job request requires a parameter called ``AttributeNames``.
This should be a two-element list of strings, where the first string is
“source-ref”, and the second string is the label attribute name from the
augmented manifest. This corresponds to the blue text in the example
above. In this case, we would define
``attribute_names = ["source-ref", "labeling-job-name"]``.

*Be sure to carefully inspect your augmented manifest so that you can
define the ``attribute_names`` variable below.*

Preview Input Data
~~~~~~~~~~~~~~~~~~

Let’s read the augmented manifest so we can inspect its contents to
better understand the format.

.. code:: ipython3

    augmented_manifest_s3_key = s3_train_data_path.split(bucket_name)[1][1:]
    s3_obj = s3.Object(bucket_name, augmented_manifest_s3_key)
    augmented_manifest = s3_obj.get()['Body'].read().decode('utf-8')
    augmented_manifest_lines = augmented_manifest.split('\n')
    
    num_training_samples = len(augmented_manifest_lines) # Compute number of training samples for use in training job request.
    
    
    print('Preview of Augmented Manifest File Contents')
    print('-------------------------------------------')
    print('\n')
    
    for i in range(2):
        print('Line {}'.format(i+1))
        print(augmented_manifest_lines[i])
        print('\n')

The key feature of the augmented manifest is that it has both the data
object itself (i.e., the image), and the annotation in-line in a single
JSON object. Note that the ``annotations`` keyword contains dimensions
and coordinates (e.g., width, top, height, left) for bounding boxes! The
augmented manifest can contain an arbitrary number of lines, as long as
each line adheres to this format.

Let’s discuss this format in more detail by descibing each parameter of
this JSON object format.

-  The ``source-ref`` field defines a single dataset object, which in
   this case is an image over which bounding boxes should be drawn. Note
   that the name of this field is arbitrary.
-  The ``object-detection-job-name`` field defines the ground truth
   bounding box annotations that pertain to the image identified in the
   ``source-ref`` field. As mentioned above, note that the name of this
   field is arbitrary. You must take care to define this field in the
   ``AttributeNames`` parameter of the training job request, as shown
   later on in this notebook.
-  Because this example augmented manifest was generated through a
   Ground Truth labeling job, this example also shows an additional
   field called ``object-detection-job-name-metadata``. This field
   contains various pieces of metadata from the labeling job that
   produced the bounding box annotation(s) for the associated image,
   e.g., the creation date, confidence scores for the annotations, etc.
   This field is ignored during the training job. However, to make it as
   easy as possible to translate Ground Truth labeling jobs into trained
   SageMaker models, it is safe to include this field in the augmented
   manifest you supply to the training job.

.. code:: ipython3

    attribute_names = ["source-ref","XXXX"] # Replace as appropriate for your augmented manifest.

Create Training Job
===================

First, we’ll construct the request for the training job.

.. code:: ipython3

    try:
        if attribute_names == ["source-ref","XXXX"]:
            raise Exception("The 'attribute_names' variable is set to default values. Please check your augmented manifest file for the label attribute name and set the 'attribute_names' variable accordingly.")
    except NameError:
        raise Exception("The attribute_names variable is not defined. Please check your augmented manifest file for the label attribute name and set the 'attribute_names' variable accordingly.")
    
    # Create unique job name 
    job_name_prefix = 'groundtruth-augmented-manifest-demo'
    timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
    job_name = job_name_prefix + timestamp
    
    training_params = \
    {
        "AlgorithmSpecification": {
            "TrainingImage": training_image, # NB. This is one of the named constants defined in the first cell.
            "TrainingInputMode": "Pipe"
        },
        "RoleArn": role,
        "OutputDataConfig": {
            "S3OutputPath": s3_output_path
        },
        "ResourceConfig": {
            "InstanceCount": 1,   
            "InstanceType": "ml.p3.2xlarge",
            "VolumeSizeInGB": 50
        },
        "TrainingJobName": job_name,
        "HyperParameters": { # NB. These hyperparameters are at the user's discretion and are beyond the scope of this demo.
             "base_network": "resnet-50",
             "use_pretrained_model": "1",
             "num_classes": "1",
             "mini_batch_size": "1",
             "epochs": "5",
             "learning_rate": "0.001",
             "lr_scheduler_step": "3,6",
             "lr_scheduler_factor": "0.1",
             "optimizer": "rmsprop",
             "momentum": "0.9",
             "weight_decay": "0.0005",
             "overlap_threshold": "0.5",
             "nms_threshold": "0.45",
             "image_shape": "300",
             "label_width": "350",
             "num_training_samples": str(num_training_samples)
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 86400
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "AugmentedManifestFile", # NB. Augmented Manifest
                        "S3Uri": s3_train_data_path,
                        "S3DataDistributionType": "FullyReplicated",
                        "AttributeNames": attribute_names # NB. This must correspond to the JSON field names in your augmented manifest.
                    }
                },
                "ContentType": "application/x-recordio",
                "RecordWrapperType": "RecordIO",
                "CompressionType": "None"
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "AugmentedManifestFile", # NB. Augmented Manifest
                        "S3Uri": s3_validation_data_path,
                        "S3DataDistributionType": "FullyReplicated",
                        "AttributeNames": attribute_names # NB. This must correspond to the JSON field names in your augmented manifest.
                    }
                },
                "ContentType": "application/x-recordio",
                "RecordWrapperType": "RecordIO",
                "CompressionType": "None"
            }
        ]
    }
     
    print('Training job name: {}'.format(job_name))
    print('\nInput Data Location: {}'.format(training_params['InputDataConfig'][0]['DataSource']['S3DataSource']))

Now we create the Amazon SageMaker training job.

.. code:: ipython3

    client = boto3.client(service_name='sagemaker')
    client.create_training_job(**training_params)
    
    # Confirm that the training job has started
    status = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print('Training job current status: {}'.format(status))


.. code:: ipython3

    TrainingJobStatus = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    SecondaryStatus = client.describe_training_job(TrainingJobName=job_name)['SecondaryStatus']
    print(TrainingJobStatus, SecondaryStatus)
    while TrainingJobStatus !='Completed' and TrainingJobStatus!='Failed':
        time.sleep(60)
        TrainingJobStatus = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
        SecondaryStatus = client.describe_training_job(TrainingJobName=job_name)['SecondaryStatus']
        print(TrainingJobStatus, SecondaryStatus)

.. code:: ipython3

    training_info = client.describe_training_job(TrainingJobName=job_name)
    print(training_info)

Conclusion
==========

That’s it! Let’s review what we’ve learned. \* Augmented manifests are a
new format that provide a seamless interface between Ground Truth
labeling jobs and SageMaker training jobs. \* In augmented manifests,
you specify the dataset objects and the associated annotations in-line.
\* Be sure to pay close attention to the ``AttributeNames`` parameter in
the training job request. The strings you specifuy in this field must
correspond to those that are present in your augmented manifest.
