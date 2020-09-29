Introduction
------------

Word2Vec is a popular algorithm used for generating dense vector
representations of words in large corpora using unsupervised learning.
These representations are useful for many natural language processing
(NLP) tasks like sentiment analysis, named entity recognition and
machine translation.

Popular models that learn such representations ignore the morphology of
words, by assigning a distinct vector to each word. This is a
limitation, especially for languages with large vocabularies and many
rare words. *SageMaker BlazingText* can learn vector representations
associated with character n-grams; representing words as the sum of
these character n-grams representations [1]. This method enables
*BlazingText* to generate vectors for out-of-vocabulary (OOV) words, as
demonstrated in this notebook.

Popular tools like
`FastText <https://github.com/facebookresearch/fastText>`__ learn
subword embeddings to generate OOV word representations, but scale
poorly as they can run only on CPUs. BlazingText extends the FastText
model to leverage GPUs, thus providing more than 10x speedup, depending
on the hardware.

[1] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, `Enriching Word
Vectors with Subword
Information <https://arxiv.org/pdf/1607.04606.pdf>`__

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
    import boto3
    import json
    
    sess = sagemaker.Session()
    
    role = get_execution_role()
    print(role) # This is the role that SageMaker would use to leverage AWS resources (S3, CloudWatch) on your behalf
    
    bucket = sess.default_bucket() # Replace with your own bucket name if needed
    print(bucket)
    prefix = 'blazingtext/subwords' #Replace with the prefix under which you want to store the data if needed

Data Ingestion
~~~~~~~~~~~~~~

Next, we download a dataset from the web on which we want to train the
word vectors. BlazingText expects a single preprocessed text file with
space separated tokens and each line of the file should contain a single
sentence.

In this example, let us train the vectors on
`text8 <http://mattmahoney.net/dc/textdata.html>`__ dataset (100 MB),
which is a small (already preprocessed) version of Wikipedia dump.

.. code:: ipython3

    !wget http://mattmahoney.net/dc/text8.zip -O text8.gz

.. code:: ipython3

    # Uncompressing
    !gzip -d text8.gz -f

After the data downloading and uncompressing is complete, we need to
upload it to S3 so that it can be consumed by SageMaker to execute
training jobs. We’ll use Python SDK to upload these two files to the
bucket and prefix location that we have set above.

.. code:: ipython3

    train_channel = prefix + '/train'
    
    sess.upload_data(path='text8', bucket=bucket, key_prefix=train_channel)
    
    s3_train_data = 's3://{}/{}'.format(bucket, train_channel)

Next we need to setup an output location at S3, where the model artifact
will be dumped. These artifacts are also the output of the algorithm’s
training job.

.. code:: ipython3

    s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)

Training Setup
--------------

Now that we are done with all the setup that is needed, we are ready to
train our object detector. To begin, let us create a
``sageMaker.estimator.Estimator`` object. This estimator will launch the
training job.

.. code:: ipython3

    region_name = boto3.Session().region_name

.. code:: ipython3

    container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")
    print('Using SageMaker BlazingText container: {} ({})'.format(container, region_name))

Training the BlazingText model for generating word vectors
----------------------------------------------------------

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
refer to `algorithm
documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html>`__
or `the text classification
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia/blazingtext_text_classification_dbpedia.ipynb>`__.

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

Now, let’s define the resource configuration and hyperparameters to
train word vectors on *text8* dataset, using “skipgram” mode on a
``c4.2xlarge`` instance.

.. code:: ipython3

    bt_model = sagemaker.estimator.Estimator(container,
                                             role, 
                                             train_instance_count=1, 
                                             train_instance_type='ml.c4.2xlarge', # Use of ml.p3.2xlarge is highly recommended for highest speed and cost efficiency
                                             train_volume_size = 30,
                                             train_max_run = 360000,
                                             input_mode= 'File',
                                             output_path=s3_output_location,
                                             sagemaker_session=sess)

Please refer to `algorithm
documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html>`__
for the complete list of hyperparameters.

.. code:: ipython3

    bt_model.set_hyperparameters(mode="skipgram",
                                 epochs=5,
                                 min_count=5,
                                 sampling_threshold=0.0001,
                                 learning_rate=0.05,
                                 window_size=5,
                                 vector_dim=100,
                                 negative_samples=5,
                                 subwords=True, # Enables learning of subword embeddings for OOV word vector generation
                                 min_char=3, # min length of char ngrams
                                 max_char=6, # max length of char ngrams
                                 batch_size=11, #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
                                 evaluation=True)# Perform similarity evaluation on WS-353 dataset at the end of training

Now that the hyper-parameters are setup, let us prepare the handshake
between our data channels and the algorithm. To do this, we need to
create the ``sagemaker.session.s3_input`` objects from our data
channels. These objects are then put in a simple dictionary, which the
algorithm consumes.

.. code:: ipython3

    train_data = sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', 
                            content_type='text/plain', s3_data_type='S3Prefix')
    data_channels = {'train': train_data}

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
``Spearman's Rho`` on some pre-selected validation datasets after the
training job has executed. This metric is a proxy for the quality of the
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

    bt_endpoint = bt_model.deploy(initial_instance_count = 1,instance_type = 'ml.m4.xlarge')

Getting vector representations for words [including out-of-vocabulary (OOV) words]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Since, we trained with **``subwords = "True"``**, we can get vector
  representations for any word - including misspelled words or words
  which were not there in the training dataset.
| If we train without the subwords flag, the training will be much
  faster but the model won’t be able to generate vectors for OOV words.
  Instead, it will return a vector of zeros for such words.

Use JSON format for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The payload should contain a list of words with the key as
“**instances**”. BlazingText supports content-type ``application/json``.

.. code:: ipython3

    words = ["awesome", "awweeesome"]
    
    payload = {"instances" : words}
    
    response = bt_endpoint.predict(json.dumps(payload))
    
    vecs = json.loads(response)
    print(vecs)

As expected, we get an n-dimensional vector (where n is vector_dim as
specified in hyperparameters) for each of the words.

Evaluation
~~~~~~~~~~

We can evaluate the quality of these representations on the task of word
similarity / relatedness. We do so by computing Spearman’s rank
correlation coefficient (Spearman, 1904) between human judgement and the
cosine similarity between the vector representations. For English, we
can use the `rare word dataset
(RW) <https://nlp.stanford.edu/~lmthang/morphoNLM/>`__, introduced by
Luong et al. (2013).

.. code:: ipython3

    !wget http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip
    !unzip "rw.zip"
    !cut -f 1,2 rw/rw.txt | awk '{print tolower($0)}' | tr '\t' '\n' > query_words.txt

The above command downloads the RW dataset and dumps all the words for
which we need vectors in query_words.txt. Let’s read this file and hit
the endpoint to get the vectors in batches of 500 words `to respect the
5MB limit of SageMaker
hosting. <https://docs.aws.amazon.com/sagemaker/latest/dg/API_runtime_InvokeEndpoint.html#API_runtime_InvokeEndpoint_RequestSyntax>`__

.. code:: ipython3

    query_words = []
    with open("query_words.txt") as f:
        for line in f.readlines():
            query_words.append(line.strip())

.. code:: ipython3

    query_words = list(set(query_words))
    total_words = len(query_words)
    vectors = {}

.. code:: ipython3

    import numpy as np
    import math
    from scipy import stats
    
    batch_size = 500
    batch_start = 0
    batch_end = batch_start + batch_size
    while len(vectors) != total_words:
        batch_end = min(batch_end, total_words)
        subset_words = query_words[batch_start:batch_end]
        payload = {"instances" : subset_words}
        response = bt_endpoint.predict(json.dumps(payload))
        vecs = json.loads(response)
        for i in vecs:
            arr = np.array(i["vector"], dtype=float)
            if np.linalg.norm(arr) == 0:
                continue
            vectors[i["word"]] = arr
        batch_start += batch_size
        batch_end += batch_size

Now that we have gotten all the vectors, we can compute the Spearman’s
rank correlation coefficient between human judgement and the cosine
similarity between the vector representations.

.. code:: ipython3

    mysim = []
    gold = []
    dropped = 0
    nwords = 0
    
    def similarity(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / n1 / n2
    
    fin = open("rw/rw.txt", 'rb')
    for line in fin:
        tline = line.decode('utf8').split()
        word1 = tline[0].lower()
        word2 = tline[1].lower()
        nwords += 1
    
        if (word1 in vectors) and (word2 in vectors):
            v1 = vectors[word1]
            v2 = vectors[word2]
            d = similarity(v1, v2)
            mysim.append(d)
            gold.append(float(tline[2]))
        else:
            dropped += 1
    fin.close()
    
    corr = stats.spearmanr(mysim, gold)
    print("Correlation: %s, Dropped words: %s%%" % (corr[0] * 100, math.ceil(dropped / nwords * 100.0)))


We can expect a Correlation coefficient of ~40, which is pretty good for
a small training dataset like text8. For more details, please refer to
`Enriching Word Vectors with Subword
Information <https://arxiv.org/pdf/1607.04606.pdf>`__

Stop / Close the Endpoint (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we should delete the endpoint before we close the notebook.

.. code:: ipython3

    sess.delete_endpoint(bt_endpoint.endpoint)

