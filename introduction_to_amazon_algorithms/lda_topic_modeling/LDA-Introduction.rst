An Introduction to SageMaker LDA
================================

**Finding topics in synthetic document data using Spectral LDA
algorithms.**

--------------

1. `Introduction <#Introduction>`__
2. `Setup <#Setup>`__
3. `Training <#Training>`__
4. `Inference <#Inference>`__
5. `Epilogue <#Epilogue>`__

Introduction
============

--------------

Amazon SageMaker LDA is an unsupervised learning algorithm that attempts
to describe a set of observations as a mixture of distinct categories.
Latent Dirichlet Allocation (LDA) is most commonly used to discover a
user-specified number of topics shared by documents within a text
corpus. Here each observation is a document, the features are the
presence (or occurrence count) of each word, and the categories are the
topics. Since the method is unsupervised, the topics are not specified
up front, and are not guaranteed to align with how a human may naturally
categorize documents. The topics are learned as a probability
distribution over the words that occur in each document. Each document,
in turn, is described as a mixture of topics.

In this notebook we will use the Amazon SageMaker LDA algorithm to train
an LDA model on some example synthetic data. We will then use this model
to classify (perform inference on) the data. The main goals of this
notebook are to,

-  learn how to obtain and store data for use in Amazon SageMaker,
-  create an AWS SageMaker training job on a data set to produce an LDA
   model,
-  use the LDA model to perform inference with an Amazon SageMaker
   endpoint.

The following are **not** goals of this notebook:

-  understand the LDA model,
-  understand how the Amazon SageMaker LDA algorithm works,
-  interpret the meaning of the inference output

If you would like to know more about these things take a minute to run
this notebook and then check out the SageMaker LDA Documentation and the
**LDA-Science.ipynb** notebook.

.. code:: ipython3

    !conda install -y scipy

.. code:: ipython3

    %matplotlib inline
    
    import os, re
    
    import boto3
    import matplotlib.pyplot as plt
    import numpy as np
    np.set_printoptions(precision=3, suppress=True)
    
    # some helpful utility functions are defined in the Python module
    # "generate_example_data" located in the same directory as this
    # notebook
    from generate_example_data import generate_griffiths_data, plot_lda, match_estimated_topics
    
    # accessing the SageMaker Python SDK
    import sagemaker
    from sagemaker.amazon.common import numpy_to_record_serializer
    from sagemaker.predictor import csv_serializer, json_deserializer

Setup
=====

--------------

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Before we do anything at all, we need data! We also need to setup our
AWS credentials so that AWS SageMaker can store and access data. In this
section we will do four things:

1. `Setup AWS Credentials <#SetupAWSCredentials>`__
2. `Obtain Example Dataset <#ObtainExampleDataset>`__
3. `Inspect Example Data <#InspectExampleData>`__
4. `Store Data on S3 <#StoreDataonS3>`__

Setup AWS Credentials
---------------------

We first need to specify some AWS credentials; specifically data
locations and access roles. This is the only cell of this notebook that
you will need to edit. In particular, we need the following data:

-  ``bucket`` - An S3 bucket accessible by this account.

   -  Used to store input training data and model data output.
   -  Should be within the same region as this notebook instance,
      training, and hosting.

-  ``prefix`` - The location in the bucket where this notebook’s input
   and and output data will be stored. (The default value is
   sufficient.)
-  ``role`` - The IAM Role ARN used to give training and hosting access
   to your data.

   -  See documentation on how to create these.
   -  The script below will try to determine an appropriate Role ARN.

.. code:: ipython3

    from sagemaker import get_execution_role
    
    role = get_execution_role()
    bucket = '<your_s3_bucket_name_here>'
    prefix = 'sagemaker/DEMO-lda-introduction'
    
    print('Training input/output will be stored in {}/{}'.format(bucket, prefix))
    print('\nIAM Role: {}'.format(role))

Obtain Example Data
-------------------

We generate some example synthetic document data. For the purposes of
this notebook we will omit the details of this process. All we need to
know is that each piece of data, commonly called a *“document”*, is a
vector of integers representing *“word counts”* within the document. In
this particular example there are a total of 25 words in the
*“vocabulary”*.

.. math::


   \underbrace{w}_{\text{document}} = \overbrace{\big[ w_1, w_2, \ldots, w_V \big] }^{\text{word counts}},
   \quad
   V = \text{vocabulary size}

These data are based on that used by Griffiths and Steyvers in their
paper `Finding Scientific
Topics <http://psiexp.ss.uci.edu/research/papers/sciencetopics.pdf>`__.
For more information, see the **LDA-Science.ipynb** notebook.

.. code:: ipython3

    print('Generating example data...')
    num_documents = 6000
    num_topics = 5
    known_alpha, known_beta, documents, topic_mixtures = generate_griffiths_data(
        num_documents=num_documents, num_topics=num_topics)
    vocabulary_size = len(documents[0])
    
    # separate the generated data into training and tests subsets
    num_documents_training = int(0.9*num_documents)
    num_documents_test = num_documents - num_documents_training
    
    documents_training = documents[:num_documents_training]
    documents_test = documents[num_documents_training:]
    
    topic_mixtures_training = topic_mixtures[:num_documents_training]
    topic_mixtures_test = topic_mixtures[num_documents_training:]
    
    print('documents_training.shape = {}'.format(documents_training.shape))
    print('documents_test.shape = {}'.format(documents_test.shape))

Inspect Example Data
--------------------

*What does the example data actually look like?* Below we print an
example document as well as its corresponding known *topic-mixture*. A
topic-mixture serves as the “label” in the LDA model. It describes the
ratio of topics from which the words in the document are found.

For example, if the topic mixture of an input document
:math:`\mathbf{w}` is,

.. math:: \theta = \left[ 0.3, 0.2, 0, 0.5, 0 \right]

then :math:`\mathbf{w}` is 30% generated from the first topic, 20% from
the second topic, and 50% from the fourth topic. For more information
see **How LDA Works** in the SageMaker documentation as well as the
**LDA-Science.ipynb** notebook.

Below, we compute the topic mixtures for the first few training
documents. As we can see, each document is a vector of word counts from
the 25-word vocabulary and its topic-mixture is a probability
distribution across the five topics used to generate the sample dataset.

.. code:: ipython3

    print('First training document =\n{}'.format(documents[0]))
    print('\nVocabulary size = {}'.format(vocabulary_size))

.. code:: ipython3

    print('Known topic mixture of first document =\n{}'.format(topic_mixtures_training[0]))
    print('\nNumber of topics = {}'.format(num_topics))
    print('Sum of elements = {}'.format(topic_mixtures_training[0].sum()))

Later, when we perform inference on the training data set we will
compare the inferred topic mixture to this known one.

--------------

Human beings are visual creatures, so it might be helpful to come up
with a visual representation of these documents. In the below plots,
each pixel of a document represents a word. The greyscale intensity is a
measure of how frequently that word occurs. Below we plot the first few
documents of the training set reshaped into 5x5 pixel grids.

.. code:: ipython3

    %matplotlib inline
    
    fig = plot_lda(documents_training, nrows=3, ncols=4, cmap='gray_r', with_colorbar=True)
    fig.suptitle('Example Document Word Counts')
    fig.set_dpi(160)

Store Data on S3
----------------

A SageMaker training job needs access to training data stored in an S3
bucket. Although training can accept data of various formats we convert
the documents MXNet RecordIO Protobuf format before uploading to the S3
bucket defined at the beginning of this notebook. We do so by making use
of the SageMaker Python SDK utility ``numpy_to_record_serializer``.

.. code:: ipython3

    # convert documents_training to Protobuf RecordIO format
    recordio_protobuf_serializer = numpy_to_record_serializer()
    fbuffer = recordio_protobuf_serializer(documents_training)
    
    # upload to S3 in bucket/prefix/train
    fname = 'lda.data'
    s3_object = os.path.join(prefix, 'train', fname)
    boto3.Session().resource('s3').Bucket(bucket).Object(s3_object).upload_fileobj(fbuffer)
    
    s3_train_data = 's3://{}/{}'.format(bucket, s3_object)
    print('Uploaded data to S3: {}'.format(s3_train_data))

Training
========

--------------

Once the data is preprocessed and available in a recommended format the
next step is to train our model on the data. There are number of
parameters required by SageMaker LDA configuring the model and defining
the computational environment in which training will take place.

First, we specify a Docker container containing the SageMaker LDA
algorithm. For your convenience, a region-specific container is
automatically chosen for you to minimize cross-region data
communication. Information about the locations of each SageMaker
algorithm is available in the documentation.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    # select the algorithm container based on this notebook's current location
    
    region_name = boto3.Session().region_name
    container = get_image_uri(region_name, 'lda')
    
    print('Using SageMaker LDA container: {} ({})'.format(container, region_name))

Particular to a SageMaker LDA training job are the following
hyperparameters:

-  **``num_topics``** - The number of topics or categories in the LDA
   model.

   -  Usually, this is not known a priori.
   -  In this example, howevever, we know that the data is generated by
      five topics.

-  **``feature_dim``** - The size of the *“vocabulary”*, in LDA
   parlance.

   -  In this example, this is equal 25.

-  **``mini_batch_size``** - The number of input training documents.

-  **``alpha0``** - *(optional)* a measurement of how “mixed” are the
   topic-mixtures.

   -  When ``alpha0`` is small the data tends to be represented by one
      or few topics.
   -  When ``alpha0`` is large the data tends to be an even combination
      of several or many topics.
   -  The default value is ``alpha0 = 1.0``.

In addition to these LDA model hyperparameters, we provide additional
parameters defining things like the EC2 instance type on which training
will run, the S3 bucket containing the data, and the AWS access role.
Note that,

-  Recommended instance type: ``ml.c4``
-  Current limitations:

   -  SageMaker LDA *training* can only run on a single instance.
   -  SageMaker LDA does not take advantage of GPU hardware.
   -  (The Amazon AI Algorithms team is working hard to provide these
      capabilities in a future release!)

.. code:: ipython3

    session = sagemaker.Session()
    
    # specify general training job information
    lda = sagemaker.estimator.Estimator(
        container,
        role,
        output_path='s3://{}/{}/output'.format(bucket, prefix),
        train_instance_count=1,
        train_instance_type='ml.c4.2xlarge',
        sagemaker_session=session,
    )
    
    # set algorithm-specific hyperparameters
    lda.set_hyperparameters(
        num_topics=num_topics,
        feature_dim=vocabulary_size,
        mini_batch_size=num_documents_training,
        alpha0=1.0,
    )
    
    # run the training job on input data stored in S3
    lda.fit({'train': s3_train_data})

If you see the message

   ``===== Job Complete =====``

at the bottom of the output logs then that means training sucessfully
completed and the output LDA model was stored in the specified output
path. You can also view information about and the status of a training
job using the AWS SageMaker console. Just click on the “Jobs” tab and
select training job matching the training job name, below:

.. code:: ipython3

    print('Training job name: {}'.format(lda.latest_training_job.job_name))

Inference
=========

--------------

A trained model does nothing on its own. We now want to use the model we
computed to perform inference on data. For this example, that means
predicting the topic mixture representing a given document.

We create an inference endpoint using the SageMaker Python SDK
``deploy()`` function from the job we defined above. We specify the
instance type where inference is computed as well as an initial number
of instances to spin up.

.. code:: ipython3

    lda_inference = lda.deploy(
        initial_instance_count=1,
        instance_type='ml.m4.xlarge',  # LDA inference may work better at scale on ml.c4 instances
    )

Congratulations! You now have a functioning SageMaker LDA inference
endpoint. You can confirm the endpoint configuration and status by
navigating to the “Endpoints” tab in the AWS SageMaker console and
selecting the endpoint matching the endpoint name, below:

.. code:: ipython3

    print('Endpoint name: {}'.format(lda_inference.endpoint))

With this realtime endpoint at our fingertips we can finally perform
inference on our training and test data.

We can pass a variety of data formats to our inference endpoint. In this
example we will demonstrate passing CSV-formatted data. Other available
formats are JSON-formatted, JSON-sparse-formatter, and RecordIO
Protobuf. We make use of the SageMaker Python SDK utilities
``csv_serializer`` and ``json_deserializer`` when configuring the
inference endpoint.

.. code:: ipython3

    lda_inference.content_type = 'text/csv'
    lda_inference.serializer = csv_serializer
    lda_inference.deserializer = json_deserializer

We pass some test documents to the inference endpoint. Note that the
serializer and deserializer will atuomatically take care of the datatype
conversion from Numpy NDArrays.

.. code:: ipython3

    results = lda_inference.predict(documents_test[:12])
    
    print(results)

It may be hard to see but the output format of SageMaker LDA inference
endpoint is a Python dictionary with the following format.

::

   {
     'predictions': [
       {'topic_mixture': [ ... ] },
       {'topic_mixture': [ ... ] },
       {'topic_mixture': [ ... ] },
       ...
     ]
   }

We extract the topic mixtures, themselves, corresponding to each of the
input documents.

.. code:: ipython3

    computed_topic_mixtures = np.array([prediction['topic_mixture'] for prediction in results['predictions']])
    
    print(computed_topic_mixtures)

If you decide to compare these results to the known topic mixtures
generated in the `Obtain Example Data <#ObtainExampleData>`__ Section
keep in mind that SageMaker LDA discovers topics in no particular order.
That is, the approximate topic mixtures computed above may be
permutations of the known topic mixtures corresponding to the same
documents.

.. code:: ipython3

    print(topic_mixtures_test[0])      # known test topic mixture
    print(computed_topic_mixtures[0])  # computed topic mixture (topics permuted)

Stop / Close the Endpoint
-------------------------

Finally, we should delete the endpoint before we close the notebook.

To do so execute the cell below. Alternately, you can navigate to the
“Endpoints” tab in the SageMaker console, select the endpoint with the
name stored in the variable ``endpoint_name``, and select “Delete” from
the “Actions” dropdown menu.

.. code:: ipython3

    sagemaker.Session().delete_endpoint(lda_inference.endpoint)

Epilogue
========

--------------

In this notebook we,

-  generated some example LDA documents and their corresponding
   topic-mixtures,
-  trained a SageMaker LDA model on a training set of documents,
-  created an inference endpoint,
-  used the endpoint to infer the topic mixtures of a test input.

There are several things to keep in mind when applying SageMaker LDA to
real-word data such as a corpus of text documents. Note that input
documents to the algorithm, both in training and inference, need to be
vectors of integers representing word counts. Each index corresponds to
a word in the corpus vocabulary. Therefore, one will need to “tokenize”
their corpus vocabulary.

.. math::


   \text{"cat"} \mapsto 0, \; \text{"dog"} \mapsto 1 \; \text{"bird"} \mapsto 2, \ldots

Each text document then needs to be converted to a “bag-of-words” format
document.

.. math::


   w = \text{"cat bird bird bird cat"} \quad \longmapsto \quad w = [2, 0, 3, 0, \ldots, 0]

Also note that many real-word applications have large vocabulary sizes.
It may be necessary to represent the input documents in sparse format.
Finally, the use of stemming and lemmatization in data preprocessing
provides several benefits. Doing so can improve training and inference
compute time since it reduces the effective vocabulary size. More
importantly, though, it can improve the quality of learned topic-word
probability matrices and inferred topic mixtures. For example, the words
*“parliament”*, *“parliaments”*, *“parliamentary”*, *“parliament’s”*,
and *“parliamentarians”* are all essentially the same word,
*“parliament”*, but with different conjugations. For the purposes of
detecting topics, such as a *“politics”* or *governments"* topic, the
inclusion of all five does not add much additional value as they all
essentiall describe the same feature.

