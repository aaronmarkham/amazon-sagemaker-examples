SageMaker PySpark K-Means Clustering MNIST Example
==================================================

1. `Introduction <#Introduction>`__
2. `Setup <#Setup>`__
3. `Loading the Data <#Loading-the-Data>`__
4. `Training with K-Means and Hosting a
   Model <#Training-with-K-Means-and-Hosting-a-Model>`__
5. `Inference <#Inference>`__
6. `Re-using existing endpoints or models to create a
   SageMakerModel <#Re-using-existing-endpoints-or-models-to-create-SageMakerModel>`__
7. `Clean-up <#Clean-up>`__
8. `More on SageMaker Spark <#More-on-SageMaker-Spark>`__

Introduction
------------

This notebook will show how to cluster handwritten digits through the
SageMaker PySpark library.

We will manipulate data through Spark using a SparkSession, and then use
the SageMaker Spark library to interact with SageMaker for training and
inference. We will first train on SageMaker using K-Means clustering on
the MNIST dataset. Then, we will see how to re-use models from existing
endpoints and from a model stored on S3 in order to only run inference.

You can visit SageMaker Spark’s GitHub repository at
https://github.com/aws/sagemaker-spark to learn more about SageMaker
Spark.

This notebook was created and tested on an ml.m4.xlarge notebook
instance.

Setup
-----

First, we import the necessary modules and create the ``SparkSession``
with the SageMaker-Spark dependencies attached.

.. code:: ipython3

    import os
    import boto3
    
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    
    import sagemaker
    from sagemaker import get_execution_role
    import sagemaker_pyspark
    
    role = get_execution_role()
    
    # Configure Spark to use the SageMaker Spark dependency jars
    jars = sagemaker_pyspark.classpath_jars()
    
    classpath = ":".join(sagemaker_pyspark.classpath_jars())
    
    # See the SageMaker Spark Github to learn how to connect to EMR from a notebook instance
    spark = SparkSession.builder.config("spark.driver.extraClassPath", classpath)\
        .master("local[*]").getOrCreate()
        
    spark

Loading the Data
----------------

Now, we load the MNIST dataset into a Spark Dataframe, which dataset is
available in LibSVM format at

``s3://sagemaker-sample-data-[region]/spark/mnist/``

where ``[region]`` is replaced with a supported AWS region, such as
us-east-1.

In order to train and make inferences our input DataFrame must have a
column of Doubles (named “label” by default) and a column of Vectors of
Doubles (named “features” by default).

Spark’s LibSVM DataFrameReader loads a DataFrame already suitable for
training and inference.

Here, we load into a DataFrame in the SparkSession running on the local
Notebook Instance, but you can connect your Notebook Instance to a
remote Spark cluster for heavier workloads. Starting from EMR 5.11.0,
SageMaker Spark is pre-installed on EMR Spark clusters. For more on
connecting your SageMaker Notebook Instance to a remote EMR cluster,
please see `this blog
post <https://aws.amazon.com/blogs/machine-learning/build-amazon-sagemaker-notebooks-backed-by-spark-in-amazon-emr/>`__.

.. code:: ipython3

    import boto3
    
    cn_regions = ['cn-north-1', 'cn-northwest-1']
    region = boto3.Session().region_name
    endpoint_domain = 'com.cn' if region in cn_regions else 'com'
    spark._jsc.hadoopConfiguration().set('fs.s3a.endpoint', 's3.{}.amazonaws.{}'.format(region, endpoint_domain))
    
    trainingData = spark.read.format('libsvm')\
        .option('numFeatures', '784')\
        .load('s3a://sagemaker-sample-data-{}/spark/mnist/train/'.format(region))
    
    testData = spark.read.format('libsvm')\
        .option('numFeatures', '784')\
        .load('s3a://sagemaker-sample-data-{}/spark/mnist/test/'.format(region))
    
    trainingData.show()

MNIST images are 28x28, resulting in 784 pixels. The dataset consists of
images of digits going from 0 to 9, representing 10 classes.

In each row: \* The ``label`` column identifies the image’s label. For
example, if the image of the handwritten number is the digit 5, the
label value is 5. \* The ``features`` column stores a vector
(``org.apache.spark.ml.linalg.Vector``) of ``Double`` values. The length
of the vector is 784, as each image consists of 784 pixels. Those pixels
are the features we will use.

As we are interested in clustering the images of digits, the number of
pixels represents the feature vector, while the number of classes
represents the number of clusters we want to find.

Training with K-Means and Hosting a Model
-----------------------------------------

Now we create a KMeansSageMakerEstimator, which uses the KMeans Amazon
SageMaker Algorithm to train on our input data, and uses the KMeans
Amazon SageMaker model image to host our model.

Calling fit() on this estimator will train our model on Amazon
SageMaker, and then create an Amazon SageMaker Endpoint to host our
model.

We can then use the SageMakerModel returned by this call to fit() to
transform Dataframes using our hosted model.

The following cell runs a training job and creates an endpoint to host
the resulting model, so this cell can take up to twenty minutes to
complete.

.. code:: ipython3

    from sagemaker_pyspark import IAMRole
    from sagemaker_pyspark.algorithms import KMeansSageMakerEstimator
    from sagemaker_pyspark import RandomNamePolicyFactory
    
    # Create K-Means Estimator
    kmeans_estimator = KMeansSageMakerEstimator(
        sagemakerRole = IAMRole(role),
        trainingInstanceType = 'ml.m4.xlarge', # Instance type to train K-means on SageMaker
        trainingInstanceCount = 1,
        endpointInstanceType = 'ml.t2.large', # Instance type to serve model (endpoint) for inference
        endpointInitialInstanceCount = 1,
        namePolicyFactory = RandomNamePolicyFactory("sparksm-1a-")) # All the resources created are prefixed with sparksm-1
    
    # Set parameters for K-Means
    kmeans_estimator.setFeatureDim(784)
    kmeans_estimator.setK(10)
    
    # Train
    initialModel = kmeans_estimator.fit(trainingData)

To put this ``KMeansSageMakerEstimator`` back into context, let’s look
at the below architecture that shows what actually runs on the notebook
instance and on SageMaker.

.. figure:: img/sagemaker-spark-kmeans-architecture.png
   :alt: Hey

   Hey

We’ll need the name of the SageMaker endpoint hosting the K-Means model
later on. This information can be accessed directly within the
``SageMakerModel``.

.. code:: ipython3

    initialModelEndpointName = initialModel.endpointName
    print(initialModelEndpointName)

Inference
---------

Now we transform our DataFrame. To do this, we serialize each row’s
“features” Vector of Doubles into a Protobuf format for inference
against the Amazon SageMaker Endpoint. We deserialize the Protobuf
responses back into our DataFrame. This serialization and
deserialization is handled automatically by the ``transform()`` method:

.. code:: ipython3

    # Run inference on the test data and show some results
    transformedData = initialModel.transform(testData)
    
    transformedData.show()

How well did the algorithm perform? Let us display the digits from each
of the clusters and manually inspect the results:

.. code:: ipython3

    from pyspark.sql.types import DoubleType
    import matplotlib.pyplot as plt
    import numpy as np
    import string
    
    # Helper function to display a digit
    def showDigit(img, caption='', xlabel='', subplot=None):
        if subplot==None:
            _,(subplot)=plt.subplots(1,1)
        imgr=img.reshape((28,28))
        subplot.axes.get_xaxis().set_ticks([])
        subplot.axes.get_yaxis().set_ticks([])
        plt.title(caption)
        plt.xlabel(xlabel)
        subplot.imshow(imgr, cmap='gray')
        
    def displayClusters(data):
        images = np.array(data.select("features").cache().take(250))
        clusters = data.select("closest_cluster").cache().take(250)
    
        for cluster in range(10):
            print('\n\n\nCluster {}:'.format(string.ascii_uppercase[cluster]))
            digits = [ img for l, img in zip(clusters, images) if int(l.closest_cluster) == cluster ]
            height=((len(digits)-1)//5)+1
            width=5
            plt.rcParams["figure.figsize"] = (width,height)
            _, subplots = plt.subplots(height, width)
            subplots=np.ndarray.flatten(subplots)
            for subplot, image in zip(subplots, digits):
                showDigit(image, subplot=subplot)
            for subplot in subplots[len(digits):]:
                subplot.axis('off')
    
            plt.show()
            
    displayClusters(transformedData)

Now that we’ve seen how to use Spark to load data and SageMaker to train
and infer on it, we will look into creating pipelines consisting of
multiple algorithms, both from SageMaker-provided algorithms as well as
from Spark MLlib.

Re-using existing endpoints or models to create ``SageMakerModel``
------------------------------------------------------------------

SageMaker Spark supports connecting a ``SageMakerModel`` to an existing
SageMaker endpoint, or to an Endpoint created by reference to model data
in S3, or to a previously completed Training Job.

This allows you to use SageMaker Spark just for model hosting and
inference on Spark-scale DataFrames without running a new Training Job.

Endpoint re-use
~~~~~~~~~~~~~~~

Here we will connect to the initial endpoint we created by using it’s
unique name. The endpoint name can either be retrieved by the console or
in in the ``endpointName`` parameter of the model you created. In our
case, we saved this early on in a variable by accessing the parameter.

.. code:: ipython3

    ENDPOINT_NAME = initialModelEndpointName
    print(ENDPOINT_NAME)

Once you have the name of the endpoint, we need to make sure that no
endpoint will be created as we are attaching to an existing endpoint.
This is done using ``endpointCreationPolicy`` field with a value of
``EndpointCreationPolicy.DO_NOT_CREATE``. As we are using an endpoint
serving a K-Means model, we also need to use the
``KMeansProtobufResponseRowDeserializer`` so that the output of the
endpoint on SageMaker will be deserialized in the right way and passed
on back to Spark in a DataFrame with the right columns.

.. code:: ipython3

    from sagemaker_pyspark import SageMakerModel
    from sagemaker_pyspark import EndpointCreationPolicy
    from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
    from sagemaker_pyspark.transformation.deserializers import KMeansProtobufResponseRowDeserializer
    
    attachedModel = SageMakerModel(
        existingEndpointName = ENDPOINT_NAME,
        endpointCreationPolicy = EndpointCreationPolicy.DO_NOT_CREATE,
        endpointInstanceType = None, # Required
        endpointInitialInstanceCount = None, # Required
        requestRowSerializer = ProtobufRequestRowSerializer(featuresColumnName = "features"), # Optional: already default value
        responseRowDeserializer = KMeansProtobufResponseRowDeserializer( # Optional: already default values
          distance_to_cluster_column_name = "distance_to_cluster",
          closest_cluster_column_name = "closest_cluster")
    )

As the data we are passing through the model is using the default
columns naming for both the input to the model (``features``) and for
the ouput of the model (``distance_to_cluster_column_name`` and
``closest_cluster_column_name``), we do not need to specify the names of
the columns in the serializer and deserializer. If your column naming is
different, it’s possible to define the name of the columns as shown
above in the ``requestRowSerializer`` and ``responseRowDeserializer``.

It is also possible to use the ``SageMakerModel.fromEndpoint`` method to
perform the same as above.

.. code:: ipython3

    transformedData2 = attachedModel.transform(testData)
    transformedData2.show()

Create model and endpoint from model data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create a SageMakerModel and an Endpoint by referring directly to
your model data in S3. To do this, you need the path to where the model
is saved (in our case on S3), as well as the role and the inference
image to use. In our case, we use the model data from the initial model,
consisting of a simple K-Means model. We can retrieve the necessary
information from the model variable, or through the console.

.. code:: ipython3

    from sagemaker_pyspark import S3DataPath
    
    MODEL_S3_PATH = S3DataPath(initialModel.modelPath.bucket, initialModel.modelPath.objectPath)
    MODEL_ROLE_ARN = initialModel.modelExecutionRoleARN
    MODEL_IMAGE_PATH = initialModel.modelImage
    
    print(MODEL_S3_PATH.bucket + MODEL_S3_PATH.objectPath)
    print(MODEL_ROLE_ARN)
    print(MODEL_IMAGE_PATH)

Similar to how we created a model from a running endpoint, we specify
the model data information using ``modelPath``,
``modelExecutionRoleARN``, ``modelImage``. This method is more akin to
creating a ``SageMakerEstimator``, where among others you specify the
endpoint information.

.. code:: ipython3

    from sagemaker_pyspark import RandomNamePolicy
    
    retrievedModel = SageMakerModel(
        modelPath = MODEL_S3_PATH,
        modelExecutionRoleARN = MODEL_ROLE_ARN,
        modelImage = MODEL_IMAGE_PATH,
        endpointInstanceType = "ml.t2.medium",
        endpointInitialInstanceCount = 1,
        requestRowSerializer = ProtobufRequestRowSerializer(), 
        responseRowDeserializer = KMeansProtobufResponseRowDeserializer(),
        namePolicy = RandomNamePolicy("sparksm-1b-"), 
        endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM
    )

It is also possible to use the ``SageMakerModel.fromModelS3Path`` method
that takes the same parameters and produces the same model.

.. code:: ipython3

    transformedData3 = retrievedModel.transform(testData)
    transformedData3.show()

Create model and endpoint from job training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create a SageMakerModel and an Endpoint by referring to a
previously-completed training job. Only difference with the model data
from S3 is that instead of providing the model data, you provide the
``trainingJobName``.

.. code:: ipython3

    TRAINING_JOB_NAME = "<YOUR_TRAINING_JOB_NAME>"
    MODEL_ROLE_ARN = initialModel.modelExecutionRoleARN
    MODEL_IMAGE_PATH = initialModel.modelImage

.. code:: ipython3

    modelFromJob = SageMakerModel.fromTrainingJob(
        trainingJobName = TRAINING_JOB_NAME,
        modelExecutionRoleARN = MODEL_ROLE_ARN,
        modelImage = MODEL_IMAGE_PATH,
        endpointInstanceType = "ml.t2.medium",
        endpointInitialInstanceCount = 1,
        requestRowSerializer = ProtobufRequestRowSerializer(), 
        responseRowDeserializer = KMeansProtobufResponseRowDeserializer(),
        namePolicy = RandomNamePolicy("sparksm-1c-"),
        endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM
    )

.. code:: ipython3

    transformedData4 = modelFromJob.transform(testData)
    transformedData4.show()

Clean-up
--------

Since we don’t need to make any more inferences, now we delete the
resources (endpoints, models, configurations, etc):

.. code:: ipython3

    # Delete the resources
    from sagemaker_pyspark import SageMakerResourceCleanup
    
    def cleanUp(model):
        resource_cleanup = SageMakerResourceCleanup(model.sagemakerClient)
        resource_cleanup.deleteResources(model.getCreatedResources())
    
    # Don't forget to include any models or pipeline models that you created in the notebook
    models = [initialModel, retrievedModel, modelFromJob]
    
    # Delete regular SageMakerModels
    for m in models:
        cleanUp(m)

More on SageMaker Spark
-----------------------

The SageMaker Spark Github repository has more about SageMaker Spark,
including how to use SageMaker Spark using the Scala SDK:
https://github.com/aws/sagemaker-spark
