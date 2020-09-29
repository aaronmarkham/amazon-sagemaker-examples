Using SageMaker Image Classification with Amazon Elastic Inference
==================================================================

1. `Introduction <#Introduction>`__
2. `Prerequisites and Preprocessing <#Prequisites-and-Preprocessing>`__
3. `Permissions and environment
   variables <#Permissions-and-environment-variables>`__
4. `Training the ResNet model <#Training-the-ResNet-model>`__
5. `Deploy The Model <#Deploy-the-model>`__
6. `Create model <#Create-model>`__
7. `Real-time inference <#Real-time-inference>`__ 1. `Create endpoint
   configuration <#Create-endpoint-configuration>`__ 2. `Create
   endpoint <#Create-endpoint>`__ 3. `Perform
   inference <#Perform-inference>`__ 4. `Clean up <#Clean-up>`__

Introduction
------------

This notebook demonstrates how to enable and use Amazon Elastic
Inference (EI) for real-time inference with SageMaker Image
Classification algorithm.

Amazon Elastic Inference (EI) is a service that provides cost-efficient
hardware acceleration meant for inferences in AWS. For more information
please visit: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

This notebook is an adaption of the SageMaker Image Classification’s
`end-to-end
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining-highlevel.ipynb>`__,
with modifications showing the changes needed to use EI for real-time
inference with SageMaker Image Classification algorithm.

In this demo, we will use the Amazon SageMaker image classification
algorithm to train on the `caltech-256
dataset <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`__.

To get started, we need to set up the environment with a few
prerequisite steps, for permissions, configurations, and so on.

Prequisites and Preprocessing
-----------------------------

Permissions and environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we set up the linkage and authentication to AWS services. There are
three parts to this:

-  The roles used to give learning and hosting access to your data. This
   will automatically be obtained from the role used to start the
   notebook
-  The S3 bucket that you want to use for training and model data
-  The Amazon SageMaker Image Classification docker image which need not
   be changed

.. code:: ipython3

    %%time
    import boto3
    import re
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    role = get_execution_role()
    
    bucket = '<<bucket-name>>' # customize to your bucket
    
    training_image = get_image_uri(boto3.Session().region_name, 'image-classification')

Data preparation
~~~~~~~~~~~~~~~~

Download the data and transfer to S3 for use in training.

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
    s3_train_key = "image-classification-full-training/train"
    s3_validation_key = "image-classification-full-training/validation"
    s3_train = 's3://{}/{}/'.format(bucket, s3_train_key)
    s3_validation = 's3://{}/{}/'.format(bucket, s3_validation_key)
    
    download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
    upload_to_s3(s3_train_key, 'caltech-256-60-train.rec')
    download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
    upload_to_s3(s3_validation_key, 'caltech-256-60-val.rec')

Training the ResNet model
-------------------------

In this demo, we are using
`Caltech-256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`__
dataset, which contains 30608 images of 256 objects. For the training
and validation data, we follow the splitting scheme in this MXNet
`example <https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/data/caltech256.sh>`__.
In particular, it randomly selects 60 images per class for training, and
uses the remaining data for validation. The algorithm takes ``RecordIO``
file as input. The user can also provide the image files as input, which
will be converted into ``RecordIO`` format using MXNet’s
`im2rec <https://mxnet.incubator.apache.org/how_to/recordio.html?highlight=im2rec>`__
tool. It takes around 50 seconds to converted the entire Caltech-256
dataset (~1.2GB) on a p2.xlarge instance. However, for this demo, we
will use record io format.

Once we have the data available in the correct format for training, the
next step is to actually train the model using the data. After setting
training parameters, we kick off training, and poll for status until
training is completed.

Training parameters
-------------------

There are two kinds of parameters that need to be set for training. The
first one are the parameters for the training job. These include:

-  **Input specification**: These are the training and validation
   channels that specify the path where training data is present. These
   are specified in the “InputDataConfig” section. The main parameters
   that need to be set is the “ContentType” which can be set to “rec” or
   “lst” based on the input data format and the S3Uri which specifies
   the bucket and the folder where the data is present.
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
   101 in this samples but other values such as 50, 152 can be used.
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
    num_layers = "18" 
    # we need to specify the input image shape for the training data
    image_shape = "3,224,224"
    # we also need to specify the number of training samples in the training set
    # for caltech it is 15420
    num_training_samples = "15420"
    # specify the number of output classes
    num_classes = "257"
    # batch size for training
    mini_batch_size =  "64"
    # number of epochs
    epochs = "2"
    # learning rate
    learning_rate = "0.01"

Training
========

Run the training using Amazon SageMaker CreateTrainingJob API

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
            "learning_rate": str(learning_rate)
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

then that means training successfully completed and the output model was
stored in the output path specified by
``training_params['OutputDataConfig']``.

You can also view information about and the status of a training job
using the Amazon SageMaker console. Just click on the “Jobs” tab.

Deploy The Model
================

--------------

A trained model does nothing on its own. We now want to use the model to
perform inference. For this example, that means predicting the topic
mixture representing a given document.

This section involves several steps,

1. `Create Model <#CreateModel>`__ - Create model for the training
   output
2. `Host the model for real-time inference with EI <#HostTheModel>`__ -
   Create an inference with EI and perform real-time inference using EI.

Create Model
------------

We now create a SageMaker Model from the training output. Using the
model we will create an Endpoint Configuration to start an endpoint for
real-time inference.

.. code:: ipython3

    %%time
    import boto3
    from time import gmtime, strftime
    
    sage = boto3.Session().client(service_name='sagemaker') 
    
    model_name="DEMO-full-image-classification-model-" + time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime()) 
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

Real-time inference
~~~~~~~~~~~~~~~~~~~

We now host the model with an endpoint and perform real-time inference.

This section involves several steps, 1. `Create endpoint
configuration <#CreateEndpointConfiguration>`__ - Create a configuration
defining an endpoint. 1. `Create endpoint <#CreateEndpoint>`__ - Use the
configuration to create an inference endpoint. 1. `Perform
inference <#PerformInference>`__ - Perform inference on some input data
using the endpoint. 1. `Clean up <#CleanUp>`__ - Delete the endpoint and
model

Create Endpoint Configuration with Amazon Elastic Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At launch, we will support configuring REST endpoints in hosting with
multiple models, e.g. for A/B testing purposes. In order to support
this, customers create an endpoint configuration, that describes the
distribution of traffic across the models, whether split, shadowed, or
sampled in some way.

SageMaker Image Classification algorithm also supports running real-time
inference with Amazon Elastic Inference (EI), a resource you can attach
to your Amazon EC2 instances to accelerate your deep learning (DL)
inference workloads. EI allows you to add inference acceleration to a
hosted endpoint for a fraction of the cost of using a full GPU instance.
Add an appropriate EI or accelerator type in addition to a CPU instance
type and the model to the production variant when creating the endpoint
configuration that you use to deploy a hosted endpoint.

In this example, an ``ml.eia1.large`` EI is attached along with
``ml.m4.xlarge`` instance type to the production variant while creating
the endpoint configuration.

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
            'AcceleratorType': 'ml.eia1.large',
            'VariantName':'AllTraffic'}])
    
    print('Endpoint configuration name: {}'.format(endpoint_config_name))
    print('Endpoint configuration arn:  {}'.format(endpoint_config_response['EndpointConfigArn']))

Create Endpoint
^^^^^^^^^^^^^^^

Next, the customer creates the endpoint that serves up the model,
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

Now the endpoint can be created. It may take sometime to create the
endpoint…

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
“Endpoints” tab in the Amazon SageMaker console.

We will finally create a runtime object from which we can invoke the
endpoint.

Perform Inference
^^^^^^^^^^^^^^^^^

Finally, the customer can now validate the model for use. They can
obtain the endpoint from the client library using the result from
previous operations and generate classifications from the trained model
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

Evaluation
''''''''''

Evaluate the image through the network for inteference. The network
outputs class probabilities and typically, one selects the class with
the maximum probability as the final class output.

**Note:** The output class detected by the network may not be accurate
in this example. To limit the time taken and cost of training, we have
trained the model only for a couple of epochs. If the network is trained
for more epochs (say 20), then the output class will be more accurate.

**Note:** The latency for the first inference invocation for endpoint
with EI is higher than the consequent ones. Please run the cell below
more than once for the first time invoking the inference for the
endpoint.

.. code:: ipython3

    %%time
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

