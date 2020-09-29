GluonCV SSD Mobilenet training and optimizing using Neo
=======================================================

1.  `Introduction <#Introduction>`__
2.  `Setup <#Setup>`__
3.  `Data Preparation <#Data-Preparation>`__
4.  `Download data <#Download-Data>`__
5.  `Convert data into RecordIO <#Convert-data-into-RecordIO>`__
6.  `Upload to S3 <#Upload-to-S3>`__
7.  `Training <#Training>`__
8.  `Hosting <#Hosting>`__
9.  `Deploy the trained model using
    Neo <#Deploy-the-trained-model-using-Neo>`__
10. `Inference <#Inference>`__

Introduction
------------

This is an end-to-end example of GluonCV SSD model training inside
sagemaker notebook and then compile the trained model using Neo runtime.
In this demo, we will demonstrate how to train and to host a mobilenet
model on the `Pascal VOC
dataset <http://host.robots.ox.ac.uk/pascal/VOC/>`__ using the Single
Shot multibox Detector (`SSD <https://arxiv.org/abs/1512.02325>`__)
algorithm. We will also demonstrate how to optimize this trained model
using Neo.

Make sure you selected ``Python 3 (Data Science)`` kernel.

**This is notebook is for demostration purpose only. Please fine tuning
the training parameters based on your own dataset.**

.. code:: ipython3

    %cd /root/amazon-sagemaker-examples/aws_sagemaker_studio/sagemaker_neo_compilation_jobs/gluoncv_ssd_mobilenet

.. code:: ipython3

    import sys

.. code:: ipython3

    !{sys.executable} -m pip install opencv-python
    !{sys.executable} -m pip install mxnet

Setup
-----

To train the ssd mobilenet model on Amazon SageMaker, we need to setup
and authenticate the use of AWS services. To start, we need an AWS
account role with SageMaker access. This role is used to give SageMaker
access to your data in S3.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    sess = sagemaker.Session()

We also need the S3 bucket that is used for training, and storing the
tranied model artifacts.

.. code:: ipython3

    #bucket = '<your_s3_bucket_name_here>' # custom bucket name.
    bucket = sess.default_bucket() 
    prefix = 'DEMO-ObjectDetection'

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

Download the Pascal VOC datasets from 2007 and 2012.

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
`MXNet <https://mxnet.incubator.apache.org/>`__. Using this format,
dataset is simple to prepare and transfer to the instance that will run
the training job. Please refer to
`object_detection_recordio_format <https://github.com/awslabs/amazon-sagemaker-examples/blob/80333fd4632cf6d924d0b91c33bf80da3bdcf926/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_recordio_format.ipynb>`__
for more information about how to prepare RecordIO dataset

.. code:: ipython3

    !python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target VOCdevkit/train.lst
    !rm -rf VOCdevkit/VOC2012
    !python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target VOCdevkit/val.lst --no-shuffle
    !rm -rf VOCdevkit/VOC2007

Upload data to S3
~~~~~~~~~~~~~~~~~

Upload the data to the S3 bucket.

.. code:: ipython3

    # Upload the RecordIO files to train and validation channels
    train_channel = prefix + '/train'
    
    sess.upload_data(path='VOCdevkit/train.rec', bucket=bucket, key_prefix=train_channel)
    sess.upload_data(path='VOCdevkit/train.idx', bucket=bucket, key_prefix=train_channel)
    
    s3_train_data = 's3://{}/{}'.format(bucket, train_channel)

Next we need to setup an output location at S3, where the model artifact
will be dumped. These artifacts are also the output of the algorithm’s
traning job.

.. code:: ipython3

    s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)
    
    # Location to save your custom code in tar.gz format.
    custom_code_upload_location = 's3://{}/{}/customcode/mxnet'.format(bucket, prefix)

Training
--------

Now that we are done with all the setup that is needed, we are ready to
train our object detector. To begin, let us create a ``sagemaker.MXNet``
object. This estimator will launch the training job.

.. code:: ipython3

    from sagemaker.mxnet import MXNet
    
    ssd_estimator = MXNet(entry_point='ssd_entry_point.py',
                          role=role,
                          output_path=s3_output_location,
                          code_location=custom_code_upload_location,
                          train_instance_count=1,
                          train_instance_type='ml.p3.8xlarge',
                          framework_version='1.4.1',
                          py_version='py3',
                          distributions={'parameter_server': {'enabled': True}},
                          hyperparameters={'epochs': 1,
                                           'data-shape': 512,
                                          }
                         )

.. code:: ipython3

    ssd_estimator.fit({'train': s3_train_data})

Hosting
-------

Once the training is done, we can deploy the trained model as an Amazon
SageMaker real-time hosted endpoint. This will allow us to make
predictions (or inference) from the model. Note that we don’t have to
host on the same insantance (or type of instance) that we used to train.

.. code:: ipython3

    obj_detector = ssd_estimator.deploy(initial_instance_count = 1,
                                        instance_type = 'ml.p3.2xlarge')

.. code:: ipython3

    from sagemaker.predictor import json_serializer, json_deserializer
    
    obj_detector.accept = 'application/json'
    obj_detector.content_type = 'application/json'
    
    obj_detector.serializer = json_serializer
    obj_detector.deserializer = json_deserializer

.. code:: ipython3

    file_name = 'test.jpg'
    
    import PIL.Image
    import numpy as np
    
    image = PIL.Image.open(file_name)
    image = np.asarray(image.resize((512, 512)))
    
    print(image.shape)

.. code:: ipython3

    %%time
    res = obj_detector.predict(image)

We have provided additional script to easily visualize the detection
outputs. You can visualize the high-confidence predictions with bounding
box by filtering out low-confidence detections using the script below:

.. code:: ipython3

    %matplotlib inline
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
            from matplotlib.patches import Rectangle
    
            img=mpimg.imread(img_file)
            plt.imshow(img)
            height = img.shape[0]
            width = img.shape[1]
            colors = dict()
            klasses = dets[0][0]
            scores = dets[1][0]
            bbox = dets[2][0]
            for i in range(len(classes)):
                klass = klasses[i][0]
                score = scores[i][0]
                x0, y0, x1, y1 = bbox[i]
                if score < thresh:
                    continue
                cls_id = int(klass)
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(), random.random())
                xmin = int(x0 * width / 512)
                ymin = int(y0 * height / 512)
                xmax = int(x1 * width / 512)
                ymax = int(y1 * height / 512)
                rect = Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=3.5)
                plt.gca().add_patch(rect)
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                plt.gca().text(xmin, ymin-2,
                                '{:s} {:.3f}'.format(class_name, score),
                                bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                        fontsize=12, color='white')
            plt.show()

.. code:: ipython3

    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                         'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

.. code:: ipython3

    # Setting a threshold 0.20 will only plot detection results that have a confidence score greater than 0.20.
    threshold = 0.20
    
    # Visualize the detections.
    visualize_detection(file_name, res, object_categories, threshold)

.. code:: ipython3

    sess.delete_endpoint(obj_detector.endpoint)

Deploy the trained model using Neo
----------------------------------

Compile trained model for ``ml_p3`` target using Neo. After that, we
will deploy Neo optimized model to the same target to do inderence.

.. code:: ipython3

    compiled_model = ssd_estimator.compile_model(target_instance_family='ml_p3', 
                                                 input_shape={'data':[1, 3, 512, 512]},
                                                 output_path=s3_output_location,
                                                 framework='mxnet', 
                                                 framework_version='1.4.1'
                                                )

.. code:: ipython3

    from sagemaker.predictor import RealTimePredictor
    compiled_model.predictor_cls = RealTimePredictor

.. code:: ipython3

    object_detector = compiled_model.deploy(initial_instance_count = 1,
                                            instance_type = 'ml.p3.2xlarge'
                                           )

Inference
---------

Now that the trained model is deployed at an endpoint that is
up-and-running, we can use this endpoint for inference. To do this, we
use an image from `PEXELS <https://www.pexels.com/>`__ which the
algorithm has so-far not seen.

.. code:: ipython3

    file_name = 'test.jpg'
    
    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload) 

Let us use our endpoint to try to detect objects within this image.
Since the image is ``jpeg``, we use the appropriate ``content_type`` to
run the prediction job. The endpoint returns a JSON file that we can
simply load and peek into.

.. code:: ipython3

    %%time
    object_detector.content_type = 'image/jpeg'
    response = object_detector.predict(payload)

.. code:: ipython3

    import json
    detections = json.loads(response)

The format of the output can be represented as
``[class_index, confidence_score, xmin, ymin, xmax, ymax]``. Typically,
we don’t consider low-confidence predictions.

.. code:: ipython3

    %matplotlib inline
    def neo_visualize_detection(img_file, dets, classes=[], thresh=0.6):
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
            from matplotlib.patches import Rectangle
    
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
                xmin = int(x0 * width / 512)
                ymin = int(y0 * height / 512)
                xmax = int(x1 * width / 512)
                ymax = int(y1 * height / 512)
                rect = Rectangle((xmin, ymin), xmax - xmin,
                                     ymax - ymin, fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=3.5)
                plt.gca().add_patch(rect)
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                plt.gca().text(xmin, ymin-2,
                                '{:s} {:.3f}'.format(class_name, score),
                                bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                        fontsize=12, color='white')
            plt.show()

.. code:: ipython3

    # Setting a threshold 0.20 will only plot detection results that have a confidence score greater than 0.20.
    threshold = 0.20
    
    # Visualize the detections.
    neo_visualize_detection(file_name, detections['prediction'], object_categories, threshold)

Delete the Endpoint
-------------------

Having an endpoint running will incur some costs. Therefore as a
clean-up job, we should delete the endpoint.

.. code:: ipython3

    sess.delete_endpoint(object_detector.endpoint)
