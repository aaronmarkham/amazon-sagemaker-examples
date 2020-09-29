Image classification transfer learning demo
===========================================

1. `Introduction <#Introduction>`__
2. `Prerequisites and Preprocessing <#Prequisites-and-Preprocessing>`__
3. `Fine-tuning the Image classification
   model <#Fine-tuning-the-Image-classification-model>`__
4. `Deploy The Model <#Deploy-the-model>`__
5. `Create model <#Create-model>`__
6. `Batch transform <#Batch-transform>`__
7. `Realtime inference <#Realtime-inference>`__ 1. `Create endpoint
   configuration <#Create-endpoint-configuration>`__ 2. `Create
   endpoint <#Create-endpoint>`__ 3. `Perform
   inference <#Perform-inference>`__ 4. `Clean up <#Clean-up>`__

Introduction
------------

Welcome to our end-to-end example of distributed image classification
algorithm in transfer learning mode. In this demo, we will use the
Amazon sagemaker image classification algorithm in transfer learning
mode to fine-tune a pre-trained model (trained on imagenet data) to
learn to classify a new dataset. In particular, the pre-trained model
will be fine-tuned using `caltech-256
dataset <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`__.

To get started, we need to set up the environment with a few
prerequisite steps, for permissions, configurations, and so on.

Prequisites and Preprocessing
-----------------------------

This notebook uses *MXNet GPU Optimized* kernel in SageMaker Studio.

Permissions and environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we set up the linkage and authentication to AWS services. There are
three parts to this:

-  The roles used to give learning and hosting access to your data. This
   will automatically be obtained from the role used to start the
   notebook
-  The S3 bucket that you want to use for training and model data
-  The Amazon sagemaker image classification docker image which need not
   be changed

.. code:: ipython3

    %%time
    import boto3
    import re
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    role = get_execution_role()
    
    bucket=sagemaker.Session().default_bucket() # customize to your bucket
    
    training_image = get_image_uri(boto3.Session().region_name, 'image-classification')
    
    print(training_image)

Fine-tuning the Image classification model
------------------------------------------

The caltech 256 dataset consist of images from 257 categories (the last
one being a clutter category) and has 30k images with a minimum of 80
images and a maximum of about 800 images per category.

The image classification algorithm can take two types of input formats.
The first is a `recordio
format <https://mxnet.incubator.apache.org/tutorials/basic/record_io.html>`__
and the other is a `lst
format <https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=im2rec>`__.
Files for both these formats are available at
http://data.dmlc.ml/mxnet/data/caltech-256/. In this example, we will
use the recordio format for training and use the training/validation
split `specified here <http://data.dmlc.ml/mxnet/data/caltech-256/>`__.

.. code:: ipython3

    import os 
    import urllib.request
    import boto3
    
    def download(url):
        filename = url.split("/")[-1]
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
    
            
    def upload_to_s3(channel, file):
        s3 = boto3.resource('s3')
        data = open(file, "rb")
        key = channel + '/' + file
        s3.Bucket(bucket).put_object(Key=key, Body=data)
    
    
    # caltech-256
    s3_train_key = "image-classification-transfer-learning/train"
    s3_validation_key = "image-classification-transfer-learning/validation"
    s3_train = 's3://{}/{}/'.format(bucket, s3_train_key)
    s3_validation = 's3://{}/{}/'.format(bucket, s3_validation_key)
    
    download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
    upload_to_s3(s3_train_key, 'caltech-256-60-train.rec')
    download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
    upload_to_s3(s3_validation_key, 'caltech-256-60-val.rec')

Once we have the data available in the correct format for training, the
next step is to actually train the model using the data. Before training
the model, we need to setup the training parameters. The next section
will explain the parameters in detail.

Training parameters
-------------------

There are two kinds of parameters that need to be set for training. The
first one are the parameters for the training job. These include:

-  **Input specification**: These are the training and validation
   channels that specify the path where training data is present. These
   are specified in the “InputDataConfig” section. The main parameters
   that need to be set is the “ContentType” which can be set to
   “application/x-recordio” or “application/x-image” based on the input
   data format and the S3Uri which specifies the bucket and the folder
   where the data is present.
-  **Output specification**: This is specified in the “OutputDataConfig”
   section. We just need to specify the path where the output can be
   stored after training
-  **Resource config**: This section specifies the type of instance on
   which to run the training and the number of hosts used for training.
   If “InstanceCount” is more than 1, then training can be run in a
   distributed manner.

Apart from the above set of parameters, there are hyperparameters that
are specific to the algorithm. These are:

-  **num_layers**: The number of layers (depth) for the network. We use
   18 in this samples but other values such as 50, 152 can be used.
-  **num_training_samples**: This is the total number of training
   samples. It is set to 15420 for caltech dataset with the current
   split
-  **num_classes**: This is the number of output classes for the new
   dataset. Imagenet was trained with 1000 output classes but the number
   of output classes can be changed for fine-tuning. For caltech, we use
   257 because it has 256 object categories + 1 clutter class
-  **epochs**: Number of training epochs
-  **learning_rate**: Learning rate for training
-  **mini_batch_size**: The number of training samples used for each
   mini batch. In distributed training, the number of training samples
   used per batch will be N \* mini_batch_size where N is the number of
   hosts on which training is run

After setting training parameters, we kick off training, and poll for
status until training is completed, which in this example, takes between
10 to 12 minutes per epoch on a p2.xlarge machine. The network typically
converges after 10 epochs. However, to save the training time, we set
the epochs to 2 but please keep in mind that it may not be sufficient to
generate a good model.

.. code:: ipython3

    # The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
    # For this training, we will use 18 layers
    num_layers = 18
    # we need to specify the input image shape for the training data
    image_shape = "3,224,224"
    # we also need to specify the number of training samples in the training set
    # for caltech it is 15420
    num_training_samples = 15420
    # specify the number of output classes
    num_classes = 257
    # batch size for training
    mini_batch_size =  128
    # number of epochs
    epochs = 2
    # learning rate
    learning_rate = 0.01
    top_k=2
    # Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be 
    # initialized with pre-trained weights
    use_pretrained_model = 1

Training
========

Run the training using Amazon sagemaker CreateTrainingJob API

.. code:: ipython3

    %%time
    import time
    import boto3
    from time import gmtime, strftime
    
    
    s3 = boto3.client('s3')
    # create unique job name 
    job_name_prefix = 'DEMO-imageclassification'
    timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
    job_name = job_name_prefix + timestamp
    training_params = \
    {
        # specify the training docker image
        "AlgorithmSpecification": {
            "TrainingImage": training_image,
            "TrainingInputMode": "File"
        },
        "RoleArn": role,
        "OutputDataConfig": {
            "S3OutputPath": 's3://{}/{}/output'.format(bucket, job_name_prefix)
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.p2.xlarge",
            "VolumeSizeInGB": 50
        },
        "TrainingJobName": job_name,
        "HyperParameters": {
            "image_shape": image_shape,
            "num_layers": str(num_layers),
            "num_training_samples": str(num_training_samples),
            "num_classes": str(num_classes),
            "mini_batch_size": str(mini_batch_size),
            "epochs": str(epochs),
            "learning_rate": str(learning_rate),
            "use_pretrained_model": str(use_pretrained_model)
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 360000
        },
    #Training data should be inside a subdirectory called "train"
    #Validation data should be inside a subdirectory called "validation"
    #The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": s3_train,
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "application/x-recordio",
                "CompressionType": "None"
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": s3_validation,
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "application/x-recordio",
                "CompressionType": "None"
            }
        ]
    }
    print('Training job name: {}'.format(job_name))
    print('\nInput Data Location: {}'.format(training_params['InputDataConfig'][0]['DataSource']['S3DataSource']))

.. code:: ipython3

    # create the Amazon SageMaker training job
    sagemaker = boto3.client(service_name='sagemaker')
    sagemaker.create_training_job(**training_params)
    
    # confirm that the training job has started
    status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print('Training job current status: {}'.format(status))
    
    try:
        # wait for the job to finish and report the ending status
        sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
        training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = training_info['TrainingJobStatus']
        print("Training job ended with status: " + status)
    except:
        print('Training failed to start')
         # if exception is raised, that means it has failed
        message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
        print('Training failed with the following error: {}'.format(message))

.. code:: ipython3

    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)

If you see the message,

   ``Training job ended with status: Completed``

then that means training sucessfully completed and the output model was
stored in the output path specified by
``training_params['OutputDataConfig']``.

You can also view information about and the status of a training job
using the AWS SageMaker console. Just click on the “Jobs” tab.

Deploy The Model
================

--------------

A trained model does nothing on its own. We now want to use the model to
perform inference. For this example, that means predicting the topic
mixture representing a given document.

Image-classification only supports encoded .jpg and .png image formats
as inference input for now. The output is the probability values for all
classes encoded in JSON format, or in JSON Lines format for batch
transform.

This section involves several steps,

1. `Create Model <#CreateModel>`__ - Create model for the training
   output
2. `Batch Transform <#BatchTransform>`__ - Create a transform job to
   perform batch inference.
3. `Host the model for realtime inference <#HostTheModel>`__ - Create an
   inference endpoint and perform realtime inference.

Create Model
------------

We now create a SageMaker Model from the training output. Using the
model we can create an Endpoint Configuration.

.. code:: ipython3

    %%time
    import boto3
    from time import gmtime, strftime
    
    sage = boto3.Session().client(service_name='sagemaker') 
    
    model_name="DEMO-image-classification-model-" + time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
    print(model_name)
    info = sage.describe_training_job(TrainingJobName=job_name)
    model_data = info['ModelArtifacts']['S3ModelArtifacts']
    print(model_data)
    
    hosting_image = get_image_uri(boto3.Session().region_name, 'image-classification')
    
    primary_container = {
        'Image': hosting_image,
        'ModelDataUrl': model_data,
    }
    
    create_model_response = sage.create_model(
        ModelName = model_name,
        ExecutionRoleArn = role,
        PrimaryContainer = primary_container)
    
    print(create_model_response['ModelArn'])

Batch transform
~~~~~~~~~~~~~~~

We now create a SageMaker Batch Transform job using the model created
above to perform batch prediction.

Download test data
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Download images under /008.bathtub
    !wget -r -np -nH --cut-dirs=2 -P /tmp/ -R "index.html*" http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/


.. code:: ipython3

    batch_input = 's3://{}/image-classification-transfer-learning/test/'.format(bucket)
    test_images = '/tmp/images/008.bathtub'
    
    !aws s3 cp $test_images $batch_input --recursive --quiet 

Create batch transform job
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
    batch_job_name = "image-classification-model" + timestamp
    request = \
    {
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "MaxConcurrentTransforms": 16,
        "MaxPayloadInMB": 6,
        "BatchStrategy": "SingleRecord",
        "TransformOutput": {
            "S3OutputPath": 's3://{}/{}/output'.format(bucket, batch_job_name)
        },
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": batch_input
                }
            },
            "ContentType": "application/x-image",
            "SplitType": "None",
            "CompressionType": "None"
        },
        "TransformResources": {
                "InstanceType": "ml.p2.xlarge",
                "InstanceCount": 1
        }
    }
    
    print('Transform job name: {}'.format(batch_job_name))
    print('\nInput Data Location: {}'.format(s3_validation))

.. code:: ipython3

    sagemaker = boto3.client('sagemaker')
    sagemaker.create_transform_job(**request)
    
    print("Created Transform job with name: ", batch_job_name)
    
    while(True):
        response = sagemaker.describe_transform_job(TransformJobName=batch_job_name)
        status = response['TransformJobStatus']
        if status == 'Completed':
            print("Transform job ended with status: " + status)
            break
        if status == 'Failed':
            message = response['FailureReason']
            print('Transform failed with the following error: {}'.format(message))
            raise Exception('Transform job failed') 
        time.sleep(30)  

After the job completes, let’s inspect the prediction results. The
accuracy may not be quite good because we set the epochs to 2 during
training which may not be sufficient to train a good model.

.. code:: ipython3

    from urllib.parse import urlparse
    import json
    import numpy as np
    
    s3_client = boto3.client('s3')
    object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
    
    def list_objects(s3_client, bucket, prefix):
        response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
        objects = [content['Key'] for content in response['Contents']]
        return objects
    
    def get_label(s3_client, bucket, prefix):
        filename = prefix.split('/')[-1]
        s3_client.download_file(bucket, prefix, filename)
        with open(filename) as f:
            data = json.load(f)
            index = np.argmax(data['prediction'])
            probability = data['prediction'][index]
        print("Result: label - " + object_categories[index] + ", probability - " + str(probability))
        return object_categories[index], probability
    
    inputs = list_objects(s3_client, bucket, urlparse(batch_input).path.lstrip('/'))
    print("Sample inputs: " + str(inputs[:2]))
    
    outputs = list_objects(s3_client, bucket, batch_job_name + "/output")
    print("Sample output: " + str(outputs[:2]))
    
    # Check prediction result of the first 2 images
    [get_label(s3_client, bucket, prefix) for prefix in outputs[0:10]]

Realtime inference
~~~~~~~~~~~~~~~~~~

We now host the model with an endpoint and perform realtime inference.

This section involves several steps, 1. `Create endpoint
configuration <#CreateEndpointConfiguration>`__ - Create a configuration
defining an endpoint. 1. `Create endpoint <#CreateEndpoint>`__ - Use the
configuration to create an inference endpoint. 1. `Perform
inference <#PerformInference>`__ - Perform inference on some input data
using the endpoint. 1. `Clean up <#CleanUp>`__ - Delete the endpoint and
model

Create Endpoint Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At launch, we will support configuring REST endpoints in hosting with
multiple models, e.g. for A/B testing purposes. In order to support
this, customers create an endpoint configuration, that describes the
distribution of traffic across the models, whether split, shadowed, or
sampled in some way.

In addition, the endpoint configuration describes the instance type
required for model deployment, and at launch will describe the
autoscaling configuration.

.. code:: ipython3

    from time import gmtime, strftime
    
    timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
    endpoint_config_name = job_name_prefix + '-epc-' + timestamp
    endpoint_config_response = sage.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[{
            'InstanceType':'ml.m4.xlarge',
            'InitialInstanceCount':1,
            'ModelName':model_name,
            'VariantName':'AllTraffic'}])
    
    print('Endpoint configuration name: {}'.format(endpoint_config_name))
    print('Endpoint configuration arn:  {}'.format(endpoint_config_response['EndpointConfigArn']))

Create Endpoint
^^^^^^^^^^^^^^^

Lastly, the customer creates the endpoint that serves up the model,
through specifying the name and configuration defined above. The end
result is an endpoint that can be validated and incorporated into
production applications. This takes 9-11 minutes to complete.

.. code:: ipython3

    %%time
    import time
    
    timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
    endpoint_name = job_name_prefix + '-ep-' + timestamp
    print('Endpoint name: {}'.format(endpoint_name))
    
    endpoint_params = {
        'EndpointName': endpoint_name,
        'EndpointConfigName': endpoint_config_name,
    }
    endpoint_response = sagemaker.create_endpoint(**endpoint_params)
    print('EndpointArn = {}'.format(endpoint_response['EndpointArn']))

Finally, now the endpoint can be created. It may take sometime to create
the endpoint…

.. code:: ipython3

    # get the status of the endpoint
    response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    print('EndpointStatus = {}'.format(status))
    
    
    # wait until the status has changed
    sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
    
    
    # print the status of the endpoint
    endpoint_response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    status = endpoint_response['EndpointStatus']
    print('Endpoint creation ended with EndpointStatus = {}'.format(status))
    
    if status != 'InService':
        raise Exception('Endpoint creation failed.')

If you see the message,

   ``Endpoint creation ended with EndpointStatus = InService``

then congratulations! You now have a functioning inference endpoint. You
can confirm the endpoint configuration and status by navigating to the
“Endpoints” tab in the AWS SageMaker console.

We will finally create a runtime object from which we can invoke the
endpoint.

Perform Inference
^^^^^^^^^^^^^^^^^

Finally, the customer can now validate the model for use. They can
obtain the endpoint from the client library using the result from
previous operations, and generate classifications from the trained model
using that endpoint.

.. code:: ipython3

    import boto3
    runtime = boto3.Session().client(service_name='runtime.sagemaker') 

Download test image
'''''''''''''''''''

.. code:: ipython3

    !wget -O /tmp/test.jpg http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/008_0007.jpg
    file_name = '/tmp/test.jpg'
    # test image
    from IPython.display import Image
    Image(file_name)  

.. code:: ipython3

    import json
    import numpy as np
    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType='application/x-image', 
                                       Body=payload)
    result = response['Body'].read()
    # result will be in json format and convert it to ndarray
    result = json.loads(result)
    # the result will output the probabilities for all classes
    # find the class with maximum probability and print the class index
    index = np.argmax(result)
    object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
    print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))

Clean up
^^^^^^^^

When we’re done with the endpoint, we can just delete it and the backing
instances will be released. Run the following cell to delete the
endpoint.

.. code:: ipython3

    sage.delete_endpoint(EndpointName=endpoint_name)

