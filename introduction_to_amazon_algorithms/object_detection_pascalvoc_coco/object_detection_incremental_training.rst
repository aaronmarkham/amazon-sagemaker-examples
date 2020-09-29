Amazon SageMaker Object Detection Incremental Training
======================================================

1.  `Introduction <#Introduction>`__
2.  `Setup <#Setup>`__
3.  `Data Preparation <#Data-Preparation>`__
4.  `Download data <#Download-data>`__
5.  `Convert data into RecordIO <#Convert-data-into-RecordIO>`__
6.  `Upload data to S3 <#Upload-data-to-S3>`__
7.  `Intial Training <#Initial-Training>`__
8.  `Incremental Training <#Incremental-Training>`__
9.  `Hosting <#Hosting>`__
10. `Inference <#Inference>`__

Introduction
------------

In this example, we will show you how to train an object detector by
re-using a model you previously trained in the SageMaker. With this
model re-using ability, you can save the training time when you update
the model with new data or improving the model quality with the same
data. In the first half of this notebook (`Intial
Training <#Initial-Training>`__), we will follow the `training with
RecordIO format
example <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_recordio_format.ipynb>`__
to train a object detection model on the `Pascal VOC
dataset <http://host.robots.ox.ac.uk/pascal/VOC/>`__. In the second
half, we will show you how you can re-use the trained model and improve
its quality without repeating the entire training process.

Setup
-----

To train the Object Detection algorithm on Amazon SageMaker, we need to
setup and authenticate the use of AWS services. To begin with we need an
AWS account role with SageMaker access. This role is used to give
SageMaker access to your data in S3 will automatically be obtained from
the role used to start the notebook.

.. code:: ipython3

    %%time
    import sagemaker
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    print(role)
    sess = sagemaker.Session()

We also need the S3 bucket that you want to use for training and to
store the tranied model artifacts. In this notebook, we require a custom
bucket that exists so as to keep the naming clean. You can end up using
a default bucket that SageMaker comes with as well.

.. code:: ipython3

    bucket = sess.default_bucket() # Use the default bucket. You can also customize bucket name.
    prefix = 'DEMO-ObjectDetection'

Lastly, we need the Amazon SageMaker Object Detection docker image,
which is static and need not be changed.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    training_image = get_image_uri(sess.boto_region_name, 'object-detection', repo_version="latest")
    print (training_image)

Data Preparation
----------------

`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__ was a popular
computer vision challenge and they released annual challenge datasets
for object detection from 2005 to 2012. In this notebook, we will use
the data sets from 2007 and 2012, named as VOC07 and VOC12 respectively.
Cumulatively, we have more than 20,000 images containing about 50,000
annotated objects. These annotated objects are grouped into 20
categories.

While using the Pascal VOC dateset, please be aware of the database
usage rights: “The VOC data includes images obtained from the”flickr"
website. Use of these images must respect the corresponding terms of
use: \* “flickr” terms of use (https://www.flickr.com/help/terms)"

Download data
~~~~~~~~~~~~~

Let us download the Pascal VOC datasets from 2007 and 2012.

.. code:: ipython3

    %%time
    
    # Download the dataset
    !wget -P /tmp http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    !wget -P /tmp http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    !wget -P /tmp http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    # # Extract the data.
    !tar -xf /tmp/VOCtrainval_11-May-2012.tar && rm /tmp/VOCtrainval_11-May-2012.tar
    !tar -xf /tmp/VOCtrainval_06-Nov-2007.tar && rm /tmp/VOCtrainval_06-Nov-2007.tar
    !tar -xf /tmp/VOCtest_06-Nov-2007.tar && rm /tmp/VOCtest_06-Nov-2007.tar

Convert data into RecordIO
~~~~~~~~~~~~~~~~~~~~~~~~~~

`RecordIO <https://mxnet.incubator.apache.org/architecture/note_data_loading.html>`__
is a highly efficient binary data format from
`MXNet <https://mxnet.incubator.apache.org/>`__ that makes it easy and
simple to prepare the dataset and transfer to the instance that will run
the training job. To generate a RecordIO file, we will use the tools
from MXNet. The provided tools will first generate a list file and then
use the `im2rec
tool <https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py>`__
to create the
`RecordIO <https://mxnet.incubator.apache.org/architecture/note_data_loading.html>`__
file. More details on how to generate RecordIO file for object detection
task, see the `MXNet
example <https://github.com/apache/incubator-mxnet/tree/master/example/ssd>`__.

We will combine the training and validation sets from both 2007 and 2012
as the training data set, and use the test set from 2007 as our
validation set.

.. code:: ipython3

    !python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target VOCdevkit/train.lst
    !rm -rf VOCdevkit/VOC2012
    !python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target VOCdevkit/val.lst --no-shuffle
    !rm -rf VOCdevkit/VOC2007

Along with this notebook, we have provided tools that can directly
generated the RecordIO files so that you do not need to do addtional
work. These tools work with the Pascal datasets lst format, which is
also quite the common among most datasets. If your data are stored in a
different format or the annotation of your data is in a different format
than the Pascal VOC dataset, you can also create the RecordIO by first
generating the .lst file and then using the im2rec tool provided by
MXNet. To make things clear, we will explain the definition of a .lst
file so that you can prepare it in your own way. The following example
is the first three lines of the .lst file we just generated for the
Pascal VOC dataset.

.. code:: ipython3

    !head -n 3 VOCdevkit/train.lst > example.lst
    f = open('example.lst','r')
    lst_content = f.read()
    print(lst_content)

As can be seen that each line in the .lst file represents the
annotations for a image. A .lst file is a **tab**-delimited file with
multiple columns. The rows of the file are annotations of the of image
files. The first column specifies a unique image index. The second
column specifies the header size of the current row. In the above
example .lst file, 2 from the second column means the second and third
columns are header information, which will not be considered as label
and bounding box information of the image specified by the current row.

The third column specifies the label width of a single object. In the
first row of above sample .lst file, 5 from the third row means each
object within an image will have 5 numbers to describe its label
information, including class index, and bounding box coordinates. If
there are multiple objects within one image, all the label information
should be listed in one line. The annotation information for each object
is represented as ``[class_index, xmin, ymin, xmax, ymax]``.

The classes should be labeled with successive numbers and start with 0.
The bounding box coordinates are ratios of its top-left (xmin, ymin) and
bottom-right (xmax, ymax) corner indices to the overall image size. Note
that the top-left corner of the entire image is the origin (0, 0). The
last column specifies the relative path of the image file.

After generating the .lst file, the RecordIO can be created by running
the following command:

.. code:: ipython3

    #python /tools/im2rec.py --pack-label --num-thread 4 your_lst_file_name /your_image_folder

Upload data to S3
~~~~~~~~~~~~~~~~~

Upload the data to the S3 bucket. We do this in multiple channels.
Channels are simply directories in the bucket that differentiate between
training and validation data. Let us simply call these directories
``train`` and ``validation``.

.. code:: ipython3

    %%time
    
    # Upload the RecordIO files to train and validation channels
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    
    s3_train_data = 's3://{}/{}'.format(bucket, train_channel)
    s3_validation_data = 's3://{}/{}'.format(bucket, validation_channel)
    
    sess.upload_data(path='VOCdevkit/train.rec', bucket=bucket, key_prefix=train_channel)
    sess.upload_data(path='VOCdevkit/val.rec', bucket=bucket, key_prefix=validation_channel)

Next we need to setup an output location at S3, where the model artifact
will be dumped. These artifacts are also the output of the algorithm’s
traning job.

.. code:: ipython3

    s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)

Initial Training
----------------

Now that we are done with all the setup that is needed, we are ready to
train our object detector. To begin, let us create a
``sageMaker.estimator.Estimator`` object. This estimator will launch the
training job.

.. code:: ipython3

    od_model = sagemaker.estimator.Estimator(training_image,
                                             role, 
                                             train_instance_count=1, 
                                             train_instance_type='ml.p3.2xlarge',
                                             train_volume_size = 50,
                                             train_max_run = 360000,
                                             input_mode= 'File',
                                             output_path=s3_output_location,
                                             sagemaker_session=sess)

The object detection algorithm at its core is the `Single-Shot Multi-Box
detection algorithm (SSD) <https://arxiv.org/abs/1512.02325>`__. This
algorithm uses a ``base_network``, which is typically a
`VGG <https://arxiv.org/abs/1409.1556>`__ or a
`ResNet <https://arxiv.org/abs/1512.03385>`__. The Amazon SageMaker
object detection algorithm supports VGG-16 and ResNet-50 now. It also
has a lot of options for hyperparameters that help configure the
training job. The next step in our training, is to setup these
hyperparameters and data channels for training the model. Consider the
following example definition of hyperparameters. See the SageMaker
Object Detection
`documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html>`__
for more details on the hyperparameters.

One of the hyperparameters here for instance is the ``epochs``. This
defines how many passes of the dataset we iterate over and determines
that training time of the algorithm. In this example, we train the model
for ``5`` epochs to generate a basic model for the PASCAL VOC dataset.

.. code:: ipython3

    od_model.set_hyperparameters(base_network='resnet-50',
                                 use_pretrained_model=1,
                                 num_classes=20,
                                 mini_batch_size=16,
                                 epochs=5,
                                 learning_rate=0.001,
                                 lr_scheduler_step='3,6',
                                 lr_scheduler_factor=0.1,
                                 optimizer='rmsprop',
                                 momentum=0.9,
                                 weight_decay=0.0005,
                                 overlap_threshold=0.5,
                                 nms_threshold=0.45,
                                 image_shape=300,
                                 label_width=350,
                                 num_training_samples=16551)

Now that the hyperparameters are setup, let us prepare the handshake
between our data channels and the algorithm. To do this, we need to
create the ``sagemaker.session.s3_input`` objects from our data
channels. These objects are then put in a simple dictionary, which the
algorithm consumes.

.. code:: ipython3

    train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                            content_type='application/x-recordio', s3_data_type='S3Prefix')
    validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                                 content_type='application/x-recordio', s3_data_type='S3Prefix')
    data_channels = {'train': train_data, 'validation': validation_data}

We have our ``Estimator`` object, we have set the hyperparameters for
this object and we have our data channels linked with the algorithm. The
only remaining thing to do is to train the algorithm. The following
command will train the algorithm. Training the algorithm involves a few
steps. Firstly, the instances that we requested while creating the
``Estimator`` classes are provisioned and are setup with the appropriate
libraries. Then, the data from our channels are downloaded into the
instance. Once this is done, the training job begins. The provisioning
and data downloading will take time, depending on the size of the data.
Therefore it might be a few minutes before we start getting data logs
for our training jobs. The data logs will also print out Mean Average
Precision (mAP) on the validation data, among other losses, for every
run of the dataset once or one epoch. This metric is a proxy for the
quality of the algorithm.

Once the job has finished a “Training job completed” message will be
printed. The trained model can be found in the S3 bucket that was setup
as ``output_path`` in the estimator.

.. code:: ipython3

    od_model.fit(inputs=data_channels, logs=True)

As you can see that it took about ``18`` minutes to reach a mAP around
``0.4``. To improve the detection quality, you can start a new training
job with an increased ``epochs`` to let the algorithm training for more
iterations. However, the new training job will re-learn everything you
have learned with the previous training job in the first ``5`` epochs.
To avoid wasting the training resources and time, we can start the new
training with a model that was generated in the previous SageMaker
training jobs.

Incremental Training
--------------------

In this section, we start a new training job from the model obtained in
previous section. We setup the estimator and hyperparameters similar to
the previous training job. Note that SageMaker object detection
algorithm currently only support the re-training feature with the same
network, which means the new training job must have the same
``base_network`` and ``num_classes`` as the previous training job.

.. code:: ipython3

    new_od_model = sagemaker.estimator.Estimator(training_image,
                                                 role, 
                                                 train_instance_count=1, 
                                                 train_instance_type='ml.p3.2xlarge',
                                                 train_volume_size = 50,
                                                 train_max_run = 360000,
                                                 input_mode= 'File',
                                                 output_path=s3_output_location,
                                                 sagemaker_session=sess)

.. code:: ipython3

    new_od_model.set_hyperparameters(base_network='resnet-50',
                                     num_classes=20,
                                     mini_batch_size=16,
                                     epochs=1,
                                     learning_rate=0.001,
                                     optimizer='rmsprop',
                                     momentum=0.9,
                                     image_shape=300,
                                     label_width=350,
                                     num_training_samples=16551)

We use the same training data from previous job. To use the pre-trained
model, we just need to add a ``model`` channel to the ``inputs`` and set
its content type to ``application/x-sagemaker-model``.

.. code:: ipython3

    # Use the same data for training and validation as the previous job.
    train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                            content_type='application/x-recordio', s3_data_type='S3Prefix')
    validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                                 content_type='application/x-recordio', s3_data_type='S3Prefix')
    
    # Use the output model from the previous job.  
    s3_model_data = od_model.model_data
    
    model_data = sagemaker.session.s3_input(s3_model_data, distribution='FullyReplicated', 
                                 content_type='application/x-sagemaker-model', s3_data_type='S3Prefix')
    
    # In addition to two data channels, add a 'model' channel for the training.
    new_data_channels = {'train': train_data, 'validation': validation_data, 'model': model_data}

Fit the new model with all three input channels.

.. code:: ipython3

    new_od_model.fit(inputs=new_data_channels, logs=True)

Instead of repeating the first ``5`` epochs from the previous job, we
started the training with the trained model and improved the results
with only one epoch. In this way, models pre-trained in SageMaker can
now be re-used to improve the training efficiency.

Hosting
-------

Once the training is done, we can deploy the trained model as an Amazon
SageMaker real-time hosted endpoint. This will allow us to make
predictions (or inference) from the model. Note that we don’t have to
host on the same instance (or type of instance) that we used to train.
Training is a prolonged and compute heavy job that require a different
of compute and memory requirements that hosting typically do not. We can
choose any type of instance we want to host the model. In our case we
chose the ``ml.p3.2xlarge`` instance to train, but we choose to host the
model on the less expensive cpu instance, ``ml.m4.xlarge``. The endpoint
deployment can be accomplished as follows:

.. code:: ipython3

    object_detector = new_od_model.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

Inference
---------

Now that the trained model is deployed at an endpoint that is
up-and-running, we can use this endpoint for inference. To do this, let
us download an image from `PEXELS <https://www.pexels.com/>`__ which the
algorithm has so-far not seen.

.. code:: ipython3

    !wget -O test.jpg https://images.pexels.com/photos/980382/pexels-photo-980382.jpeg
    file_name = 'test.jpg'
    
    with open(file_name, 'rb') as image:
        f = image.read()
        b = bytearray(f)
        ne = open('n.txt','wb')
        ne.write(b)

Let us use our endpoint to try to detect objects within this image.
Since the image is ``jpeg``, we use the appropriate ``content_type`` to
run the prediction job. The endpoint returns a JSON file that we can
simply load and peek into.

.. code:: ipython3

    import json
    
    object_detector.content_type = 'image/jpeg'
    results = object_detector.predict(b)
    detections = json.loads(results)
    print (detections)

The results are in a format that is similar to the .lst format with an
addition of a confidence score for each detected object. The format of
the output can be represented as
``[class_index, confidence_score, xmin, ymin, xmax, ymax]``. Typically,
we don’t consider low-confidence predictions.

We have provided additional script to easily visualize the detection
outputs. You can visualize the high-confidence predictions with bounding
box by filtering out low-confidence detections using the script below:

.. code:: ipython3

    def visualize_detection(img_file, dets, classes=[], thresh=0.6):
            """
            visualize detections in one image
            Parameters:
            ----------
            img : numpy.array
                image, in bgr format
            dets : numpy.array
                ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
                each row is one object
            classes : tuple or list of str
                class names
            thresh : float
                score threshold
            """
            import random
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
    
            img=mpimg.imread(img_file)
            plt.imshow(img)
            height = img.shape[0]
            width = img.shape[1]
            colors = dict()
            for det in dets:
                (klass, score, x0, y0, x1, y1) = det
                if score < thresh:
                    continue
                cls_id = int(klass)
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(), random.random())
                xmin = int(x0 * width)
                ymin = int(y0 * height)
                xmax = int(x1 * width)
                ymax = int(y1 * height)
                rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=3.5)
                plt.gca().add_patch(rect)
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                plt.gca().text(xmin, ymin - 2,
                                '{:s} {:.3f}'.format(class_name, score),
                                bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                        fontsize=12, color='white')
            plt.show()

For the sake of this notebook, we trained the model with only one epoch.
This implies that the results might not be optimal. To achieve better
detection results, you can try to tune the hyperparameters and train the
model for more epochs.

.. code:: ipython3

    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    # Setting a threshold 0.20 will only plot detection results that have a confidence score greater than 0.20.
    threshold = 0.20
    
    # Visualize the detections.
    visualize_detection(file_name, detections['prediction'], object_categories, threshold)

Delete the Endpoint
-------------------

Having an endpoint running will incur some costs. Therefore as a
clean-up job, we should delete the endpoint.

.. code:: ipython3

    sagemaker.Session().delete_endpoint(object_detector.endpoint)
