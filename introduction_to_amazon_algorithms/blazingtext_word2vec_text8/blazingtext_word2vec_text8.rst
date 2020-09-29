Introduction
------------

Word2Vec is a popular algorithm used for generating dense vector
representations of words in large corpora using unsupervised learning.
The resulting vectors have been shown to capture semantic relationships
between the corresponding words and are used extensively for many
downstream natural language processing (NLP) tasks like sentiment
analysis, named entity recognition and machine translation.

SageMaker BlazingText which provides efficient implementations of
Word2Vec on

-  single CPU instance
-  single instance with multiple GPUs - P2 or P3 instances
-  multiple CPU instances (Distributed training)

In this notebook, we demonstrate how BlazingText can be used for
distributed training of word2vec using multiple CPU instances.

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
    prefix = 'sagemaker/DEMO-blazingtext-text8' #Replace with the prefix under which you want to store the data if needed

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
BlazingText also supports learning of subword embeddings with CBOW and
skip-gram modes. This enables BlazingText to generate vectors for
out-of-vocabulary (OOV) words, as demonstrated in this
`notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_word2vec_subwords_text8/blazingtext_word2vec_subwords_text8.ipynb>`__.

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
train word vectors on *text8* dataset, using “batch_skipgram” mode on
two c4.2xlarge instances.

.. code:: ipython3

    bt_model = sagemaker.estimator.Estimator(container,
                                             role, 
                                             train_instance_count=2, 
                                             train_instance_type='ml.c4.2xlarge',
                                             train_volume_size = 5,
                                             train_max_run = 360000,
                                             input_mode= 'File',
                                             output_path=s3_output_location,
                                             sagemaker_session=sess)

Please refer to `algorithm
documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html>`__
for the complete list of hyperparameters.

.. code:: ipython3

    bt_model.set_hyperparameters(mode="batch_skipgram",
                                 epochs=5,
                                 min_count=5,
                                 sampling_threshold=0.0001,
                                 learning_rate=0.05,
                                 window_size=5,
                                 vector_dim=100,
                                 negative_samples=5,
                                 batch_size=11, #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
                                 evaluation=True,# Perform similarity evaluation on WS-353 dataset at the end of training
                                 subwords=False) # Subword embedding learning is not supported by batch_skipgram

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

Getting vector representations for words
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use JSON format for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The payload should contain a list of words with the key as
“**instances**”. BlazingText supports content-type ``application/json``.

.. code:: ipython3

    words = ["awesome", "blazing"]
    
    payload = {"instances" : words}
    
    response = bt_endpoint.predict(json.dumps(payload))
    
    vecs = json.loads(response)
    print(vecs)

As expected, we get an n-dimensional vector (where n is vector_dim as
specified in hyperparameters) for each of the words. If the word is not
there in the training dataset, the model will return a vector of zeros.

Evaluation
~~~~~~~~~~

Let us now download the word vectors learned by our model and visualize
them using a
`t-SNE <https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding>`__
plot.

.. code:: ipython3

    s3 = boto3.resource('s3')
    
    key = bt_model.model_data[bt_model.model_data.find("/", 5)+1:]
    s3.Bucket(bucket).download_file(key, 'model.tar.gz')

Uncompress ``model.tar.gz`` to get ``vectors.txt``

.. code:: ipython3

    !tar -xvzf model.tar.gz

If you set “evaluation” as “true” in the hyperparameters, then
“eval.json” will be there in the model artifacts.

The quality of trained model is evaluated on word similarity task. We
use `WS-353 <http://alfonseca.org/eng/research/wordsim353.html>`__,
which is one of the most popular test datasets used for this purpose. It
contains word pairs together with human-assigned similarity judgments.

The word representations are evaluated by ranking the pairs according to
their cosine similarities, and measuring the Spearmans rank correlation
coefficient with the human judgments.

Let’s look at the evaluation scores which are there in eval.json. For
embeddings trained on the text8 dataset, scores above 0.65 are pretty
good.

.. code:: ipython3

    !cat eval.json

Now, let us do a 2D visualization of the word vectors

.. code:: ipython3

    import numpy as np
    from sklearn.preprocessing import normalize
    
    # Read the 400 most frequent word vectors. The vectors in the file are in descending order of frequency.
    num_points = 400
    
    first_line = True
    index_to_word = []
    with open("vectors.txt","r") as f:
        for line_num, line in enumerate(f):
            if first_line:
                dim = int(line.strip().split()[1])
                word_vecs = np.zeros((num_points, dim), dtype=float)
                first_line = False
                continue
            line = line.strip()
            word = line.split()[0]
            vec = word_vecs[line_num-1]
            for index, vec_val in enumerate(line.split()[1:]):
                vec[index] = float(vec_val)
            index_to_word.append(word)
            if line_num >= num_points:
                break
    word_vecs = normalize(word_vecs, copy=False, return_norm=False)

.. code:: ipython3

    from sklearn.manifold import TSNE
    
    tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000)
    two_d_embeddings = tsne.fit_transform(word_vecs[:num_points])
    labels = index_to_word[:num_points]

.. code:: ipython3

    from matplotlib import pylab
    %matplotlib inline
    
    def plot(embeddings, labels):
        pylab.figure(figsize=(20,20))
        for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()
    
    plot(two_d_embeddings, labels)

Running the code above might generate a plot like the one below. t-SNE
and Word2Vec are stochastic, so although when you run the code the plot
won’t look exactly like this, you can still see clusters of similar
words such as below where ‘british’, ‘american’, ‘french’, ‘english’ are
near the bottom-left, and ‘military’, ‘army’ and ‘forces’ are all
together near the bottom.

.. figure:: ./tsne.png
   :alt: tsne plot of embeddings

   tsne plot of embeddings

Stop / Close the Endpoint (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we should delete the endpoint before we close the notebook.

.. code:: ipython3

    sess.delete_endpoint(bt_endpoint.endpoint)
