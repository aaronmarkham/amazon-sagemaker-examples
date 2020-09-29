Object Detection using Managed Spot Training
============================================

The example here is almost the same as `Amazon SageMaker Object
Detection using the RecordIO
format <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_recordio_format.ipynb>`__.

This notebook tackles the exact same problem with the same solution, but
it has been modified to be able to run using SageMaker Managed Spot
infrastructure. SageMaker Managed Spot uses `EC2 Spot
Instances <https://aws.amazon.com/ec2/spot/>`__ to run Training at a
lower cost.

Please read the original notebook and try it out to gain an
understanding of the ML use-case and how it is being solved. We will not
delve into that here in this notebook.

Setup
-----

Again, we won’t go into detail explaining the code below, it has been
lifted verbatim from `Amazon SageMaker Object Detection using the
RecordIO
format <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_recordio_format.ipynb>`__.

.. code:: ipython3

    !pip install -qU awscli boto3 sagemaker

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    role = get_execution_role()
    sess = sagemaker.Session()
    bucket = sess.default_bucket() 
    prefix = 'DEMO-ObjectDetection'
    training_image = get_image_uri(sess.boto_region_name, 'object-detection', repo_version="latest")


Download And Prepare Data
~~~~~~~~~~~~~~~~~~~~~~~~~

Note: this notebook downloads and uses the Pascal VOC dateset, please be
aware of the database usage rights: “The VOC data includes images
obtained from the”flickr" website. Use of these images must respect the
corresponding terms of use: \* “flickr” terms of use
(https://www.flickr.com/help/terms)"

.. code:: ipython3

    # Download the dataset
    !wget -P /tmp http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    !wget -P /tmp http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    !wget -P /tmp http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    
    # Extract the data.
    !tar -xf /tmp/VOCtrainval_11-May-2012.tar && rm /tmp/VOCtrainval_11-May-2012.tar
    !tar -xf /tmp/VOCtrainval_06-Nov-2007.tar && rm /tmp/VOCtrainval_06-Nov-2007.tar
    !tar -xf /tmp/VOCtest_06-Nov-2007.tar && rm /tmp/VOCtest_06-Nov-2007.tar
    
    # Convert data into RecordIO
    !python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target VOCdevkit/train.lst
    !rm -rf VOCdevkit/VOC2012
    !python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target VOCdevkit/val.lst --no-shuffle
    !rm -rf VOCdevkit/VOC2007

Upload data to S3
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Upload the RecordIO files to train and validation channels
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    
    sess.upload_data(path='VOCdevkit/train.rec', bucket=bucket, key_prefix=train_channel)
    sess.upload_data(path='VOCdevkit/val.rec', bucket=bucket, key_prefix=validation_channel)
    
    s3_train_data = 's3://{}/{}'.format(bucket, train_channel)
    s3_validation_data = 's3://{}/{}'.format(bucket, validation_channel)

Object Detection using Managed Spot Training
============================================

For Managed Spot Training using Object Detection we need to configure
two things: 1. Enable the ``train_use_spot_instances`` constructor arg -
a simple self-explanatory boolean. 2. Set the ``train_max_wait``
constructor arg - this is an int arg representing the amount of time you
are willing to wait for Spot infrastructure to become available. Some
instance types are harder to get at Spot prices and you may have to wait
longer. You are not charged for time spent waiting for Spot
infrastructure to become available, you’re only charged for actual
compute time spent once Spot instances have been successfully procured.

Feel free to toggle the ``train_use_spot_instances`` variable to see the
effect of running the same job using regular (a.k.a. “On Demand”)
infrastructure.

Note that ``train_max_wait`` can be set if and only if
``train_use_spot_instances`` is enabled and **must** be greater than or
equal to ``train_max_run``.

.. code:: ipython3

    train_use_spot_instances = True
    train_max_run=3600
    train_max_wait = 3600 if train_use_spot_instances else None

Training
--------

Now that we are done with all the setup that is needed, we are ready to
train our object detector. To begin, let us create a
``sageMaker.estimator.Estimator`` object. This estimator will launch the
training job.

.. code:: ipython3

    s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)
    od_model = sagemaker.estimator.Estimator(training_image,
                                             role, 
                                             train_instance_count=1, 
                                             train_instance_type='ml.p3.2xlarge',
                                             train_volume_size = 50,
                                             input_mode= 'File',
                                             output_path=s3_output_location,
                                             sagemaker_session=sess,
                                             train_use_spot_instances=train_use_spot_instances,
                                             train_max_run=train_max_run,
                                             train_max_wait=train_max_wait)
    
    od_model.set_hyperparameters(base_network='resnet-50',
                                 use_pretrained_model=1,
                                 num_classes=20,
                                 mini_batch_size=32,
                                 epochs=1,
                                 learning_rate=0.001,
                                 lr_scheduler_step='3,6',
                                 lr_scheduler_factor=0.1,
                                 optimizer='sgd',
                                 momentum=0.9,
                                 weight_decay=0.0005,
                                 overlap_threshold=0.5,
                                 nms_threshold=0.45,
                                 image_shape=300,
                                 label_width=350,
                                 num_training_samples=16551)
    
    train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                            content_type='application/x-recordio', s3_data_type='S3Prefix')
    validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                                 content_type='application/x-recordio', s3_data_type='S3Prefix')
    data_channels = {'train': train_data, 'validation': validation_data}
    
    od_model.fit(inputs=data_channels, logs=True)

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
