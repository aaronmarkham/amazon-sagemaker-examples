Introduction
------------

Text Classification can be used to solve various use-cases like
sentiment analysis, spam detection, hashtag prediction etc. This
notebook demonstrates the use of SageMaker BlazingText to perform
supervised binary/multi class with single or multi label text
classification. BlazingText can train the model on more than a billion
words in a couple of minutes using a multi-core CPU or a GPU, while
achieving performance on par with the state-of-the-art deep learning
text classification algorithms. BlazingText extends the fastText text
classifier to leverage GPU acceleration using custom CUDA kernels.

Setup
-----

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting. If you don’t specify a bucket, SageMaker SDK
   will create a default bucket following a pre-defined naming
   convention in the same region.
-  The IAM role ARN used to give SageMaker access to your data. It can
   be fetched using the **get_execution_role** method from sagemaker
   python SDK.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    import json
    import boto3
    
    sess = sagemaker.Session()
    
    role = get_execution_role()
    print(role) # This is the role that SageMaker would use to leverage AWS resources (S3, CloudWatch) on your behalf
    
    bucket = sess.default_bucket() # Replace with your own bucket name if needed
    print(bucket)
    prefix = 'blazingtext/supervised' #Replace with the prefix under which you want to store the data if needed

Data Preparation
~~~~~~~~~~~~~~~~

Now we’ll download a dataset from the web on which we want to train the
text classification model. BlazingText expects a single preprocessed
text file with space separated tokens and each line of the file should
contain a single sentence and the corresponding label(s) prefixed by
“\_\ *label\_*”.

In this example, let us train the text classification model on the
`DBPedia Ontology
Dataset <https://wiki.dbpedia.org/services-resources/dbpedia-data-set-2014#2>`__
as done by `Zhang et al <https://arxiv.org/pdf/1509.01626.pdf>`__. The
DBpedia ontology dataset is constructed by picking 14 nonoverlapping
classes from DBpedia 2014. It has 560,000 training samples and 70,000
testing samples. The fields we used for this dataset contain title and
abstract of each Wikipedia article.

.. code:: ipython3

    !wget https://github.com/saurabh3949/Text-Classification-Datasets/raw/master/dbpedia_csv.tar.gz

.. code:: ipython3

    !tar -xzvf dbpedia_csv.tar.gz

Let us inspect the dataset and the classes to get some understanding
about how the data and the label is provided in the dataset.

.. code:: ipython3

    !head dbpedia_csv/train.csv -n 3

As can be seen from the above output, the CSV has 3 fields - Label
index, title and abstract. Let us first create a label index to label
name mapping and then proceed to preprocess the dataset for ingestion by
BlazingText.

Next we will print the labels file (``classes.txt``) to see all possible
labels followed by creating an index to label mapping.

.. code:: ipython3

    !cat dbpedia_csv/classes.txt

The following code creates the mapping from integer indices to class
label which will later be used to retrieve the actual class name during
inference.

.. code:: ipython3

    index_to_label = {} 
    with open("dbpedia_csv/classes.txt") as f:
        for i,label in enumerate(f.readlines()):
            index_to_label[str(i+1)] = label.strip()
    print(index_to_label)

Data Preprocessing
------------------

We need to preprocess the training data into **space separated tokenized
text** format which can be consumed by ``BlazingText`` algorithm. Also,
as mentioned previously, the class label(s) should be prefixed with
``__label__`` and it should be present in the same line along with the
original sentence. We’ll use ``nltk`` library to tokenize the input
sentences from DBPedia dataset.

Download the nltk tokenizer and other libraries

.. code:: ipython3

    from random import shuffle
    import multiprocessing
    from multiprocessing import Pool
    import csv
    import nltk
    nltk.download('punkt')

.. code:: ipython3

    def transform_instance(row):
        cur_row = []
        label = "__label__" + index_to_label[row[0]]  #Prefix the index-ed label with __label__
        cur_row.append(label)
        cur_row.extend(nltk.word_tokenize(row[1].lower()))
        cur_row.extend(nltk.word_tokenize(row[2].lower()))
        return cur_row

The ``transform_instance`` will be applied to each data instance in
parallel using python’s multiprocessing module

.. code:: ipython3

    def preprocess(input_file, output_file, keep=1):
        all_rows = []
        with open(input_file, 'r') as csvinfile:
            csv_reader = csv.reader(csvinfile, delimiter=',')
            for row in csv_reader:
                all_rows.append(row)
        shuffle(all_rows)
        all_rows = all_rows[:int(keep*len(all_rows))]
        pool = Pool(processes=multiprocessing.cpu_count())
        transformed_rows = pool.map(transform_instance, all_rows)
        pool.close() 
        pool.join()
        
        with open(output_file, 'w') as csvoutfile:
            csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
            csv_writer.writerows(transformed_rows)

.. code:: ipython3

    %%time
    
    # Preparing the training dataset
    
    # Since preprocessing the whole dataset might take a couple of mintutes,
    # we keep 20% of the training dataset for this demo.
    # Set keep to 1 if you want to use the complete dataset
    preprocess('dbpedia_csv/train.csv', 'dbpedia.train', keep=.2)
            
    # Preparing the validation dataset        
    preprocess('dbpedia_csv/test.csv', 'dbpedia.validation')

The data preprocessing cell might take a minute to run. After the data
preprocessing is complete, we need to upload it to S3 so that it can be
consumed by SageMaker to execute training jobs. We’ll use Python SDK to
upload these two files to the bucket and prefix location that we have
set above.

.. code:: ipython3

    %%time
    
    train_channel = prefix + '/train'
    validation_channel = prefix + '/validation'
    
    sess.upload_data(path='dbpedia.train', bucket=bucket, key_prefix=train_channel)
    sess.upload_data(path='dbpedia.validation', bucket=bucket, key_prefix=validation_channel)
    
    s3_train_data = 's3://{}/{}'.format(bucket, train_channel)
    s3_validation_data = 's3://{}/{}'.format(bucket, validation_channel)

Next we need to setup an output location at S3, where the model artifact
will be dumped. These artifacts are also the output of the algorithm’s
traning job.

.. code:: ipython3

    s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)

Training
--------

Now that we are done with all the setup that is needed, we are ready to
train our object detector. To begin, let us create a
``sageMaker.estimator.Estimator`` object. This estimator will launch the
training job.

.. code:: ipython3

    region_name = boto3.Session().region_name

.. code:: ipython3

    container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")
    print('Using SageMaker BlazingText container: {} ({})'.format(container, region_name))

Training the BlazingText model for supervised text classification
-----------------------------------------------------------------

Similar to the original implementation of
`Word2Vec <https://arxiv.org/pdf/1301.3781.pdf>`__, SageMaker
BlazingText provides an efficient implementation of the continuous
bag-of-words (CBOW) and skip-gram architectures using Negative Sampling,
on CPUs and additionally on GPU[s]. The GPU implementation uses highly
optimized CUDA kernels. To learn more, please refer to `BlazingText:
Scaling and Accelerating Word2Vec using Multiple
GPUs <https://dl.acm.org/citation.cfm?doid=3146347.3146354>`__.

Besides skip-gram and CBOW, SageMaker BlazingText also supports the
“Batch Skipgram” mode, which uses efficient mini-batching and
matrix-matrix operations (`BLAS Level 3
routines <https://software.intel.com/en-us/mkl-developer-reference-fortran-blas-level-3-routines>`__).
This mode enables distributed word2vec training across multiple CPU
nodes, allowing almost linear scale up of word2vec computation to
process hundreds of millions of words per second. Please refer to
`Parallelizing Word2Vec in Shared and Distributed
Memory <https://arxiv.org/pdf/1604.04661.pdf>`__ to learn more.

BlazingText also supports a *supervised* mode for text classification.
It extends the FastText text classifier to leverage GPU acceleration
using custom CUDA kernels. The model can be trained on more than a
billion words in a couple of minutes using a multi-core CPU or a GPU,
while achieving performance on par with the state-of-the-art deep
learning text classification algorithms. For more information, please
refer to the `algorithm
documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html>`__.

To summarize, the following modes are supported by BlazingText on
different types instances:

+-----------------------+-----+---------+---------------+---------------+
| Modes                 | cbo | skipgra | batch_skipgra | supervised    |
|                       | w   | m       | m             |               |
|                       | (su | (suppor |               |               |
|                       | ppo | ts      |               |               |
|                       | rts | subword |               |               |
|                       | sub | s       |               |               |
|                       | wor | trainin |               |               |
|                       | ds  | g)      |               |               |
|                       | tra |         |               |               |
|                       | ini |         |               |               |
|                       | ng) |         |               |               |
+=======================+=====+=========+===============+===============+
| Single CPU instance   | ✔   | ✔       | ✔             | ✔             |
+-----------------------+-----+---------+---------------+---------------+
| Single GPU instance   | ✔   | ✔       |               | ✔ (Instance   |
|                       |     |         |               | with 1 GPU    |
|                       |     |         |               | only)         |
+-----------------------+-----+---------+---------------+---------------+
| Multiple CPU          |     |         | ✔             |               |
| instances             |     |         |               |               |
+-----------------------+-----+---------+---------------+---------------+

Now, let’s define the SageMaker ``Estimator`` with resource
configurations and hyperparameters to train Text Classification on
*DBPedia* dataset, using “supervised” mode on a ``c4.4xlarge`` instance.

.. code:: ipython3

    bt_model = sagemaker.estimator.Estimator(container,
                                             role, 
                                             train_instance_count=1, 
                                             train_instance_type='ml.c4.4xlarge',
                                             train_volume_size = 30,
                                             train_max_run = 360000,
                                             input_mode= 'File',
                                             output_path=s3_output_location,
                                             sagemaker_session=sess)

Please refer to `algorithm
documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html>`__
for the complete list of hyperparameters.

.. code:: ipython3

    bt_model.set_hyperparameters(mode="supervised",
                                epochs=10,
                                min_count=2,
                                learning_rate=0.05,
                                vector_dim=10,
                                early_stopping=True,
                                patience=4,
                                min_epochs=5,
                                word_ngrams=2)

Now that the hyper-parameters are setup, let us prepare the handshake
between our data channels and the algorithm. To do this, we need to
create the ``sagemaker.session.s3_input`` objects from our data
channels. These objects are then put in a simple dictionary, which the
algorithm consumes.

.. code:: ipython3

    train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                            content_type='text/plain', s3_data_type='S3Prefix')
    validation_data = sagemaker.session.s3_input(s3_validation_data, distribution='FullyReplicated', 
                                 content_type='text/plain', s3_data_type='S3Prefix')
    data_channels = {'train': train_data, 'validation': validation_data}

We have our ``Estimator`` object, we have set the hyper-parameters for
this object and we have our data channels linked with the algorithm. The
only remaining thing to do is to train the algorithm. The following
command will train the algorithm. Training the algorithm involves a few
steps. Firstly, the instance that we requested while creating the
``Estimator`` classes is provisioned and is setup with the appropriate
libraries. Then, the data from our channels are downloaded into the
instance. Once this is done, the training job begins. The provisioning
and data downloading will take some time, depending on the size of the
data. Therefore it might be a few minutes before we start getting
training logs for our training jobs. The data logs will also print out
Accuracy on the validation data for every epoch after training job has
executed ``min_epochs``. This metric is a proxy for the quality of the
algorithm.

Once the job has finished a “Job complete” message will be printed. The
trained model can be found in the S3 bucket that was setup as
``output_path`` in the estimator.

.. code:: ipython3

    bt_model.fit(inputs=data_channels, logs=True)

Hosting / Inference
-------------------

Once the training is done, we can deploy the trained model as an Amazon
SageMaker real-time hosted endpoint. This will allow us to make
predictions (or inference) from the model. Note that we don’t have to
host on the same type of instance that we used to train. Because
instance endpoints will be up and running for long, it’s advisable to
choose a cheaper instance for inference.

.. code:: ipython3

    text_classifier = bt_model.deploy(initial_instance_count = 1,instance_type = 'ml.m4.xlarge')

Use JSON format for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BlazingText supports ``application/json`` as the content-type for
inference. The payload should contain a list of sentences with the key
as “**instances**” while being passed to the endpoint.

.. code:: ipython3

    sentences = ["Convair was an american aircraft manufacturing company which later expanded into rockets and spacecraft.",
                "Berwick secondary college is situated in the outer melbourne metropolitan suburb of berwick ."]
    
    # using the same nltk tokenizer that we used during data preparation for training
    tokenized_sentences = [' '.join(nltk.word_tokenize(sent)) for sent in sentences]
    
    payload = {"instances" : tokenized_sentences}
    
    response = text_classifier.predict(json.dumps(payload))
    
    predictions = json.loads(response)
    print(json.dumps(predictions, indent=2))

By default, the model will return only one prediction, the one with the
highest probability. For retrieving the top k predictions, you can set
``k`` in the configuration as shown below:

.. code:: ipython3

    payload = {"instances" : tokenized_sentences,
              "configuration": {"k": 2}}
    
    response = text_classifier.predict(json.dumps(payload))
    
    predictions = json.loads(response)
    print(json.dumps(predictions, indent=2))

Stop / Close the Endpoint (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we should delete the endpoint before we close the notebook if
we don’t need to keep the endpoint running for serving realtime
predictions.

.. code:: ipython3

    sess.delete_endpoint(text_classifier.endpoint)
