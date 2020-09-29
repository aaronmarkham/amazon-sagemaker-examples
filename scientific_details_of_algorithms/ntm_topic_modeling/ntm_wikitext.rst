Amazon SageMaker Neural Topic Model now supports auxiliary vocabulary channel, new topic evaluation metrics, and training subsampling
=====================================================================================================================================

Amazon SageMaker Neural Topic Model(NTM) is an unsupervised learning
algorithm that learns the topic distributions of large collections of
document corpus. With SageMaker NTM, you can build machine learning
solutions for use cases such as document classification, information
retrieval, and content recommendation. SageMaker provides a rich set of
model training configuration options such as network architecture,
automatic early stopping, as well as hyperparameters to fine-tune
between a magnitude of metrics such as document modeling accuracy, human
interpretability, and granularity of the learned topics. see
Introduction to the Amazon SageMaker Neural Topic Model
(https://aws.amazon.com/blogs/machine-learning/introduction-to-the-amazon-sagemaker-neural-topic-model/)
if you are not already familiar with SageMaker and SageMaker NTM.

If you are new to machine learning, or want to free up time to focus on
other tasks, then the fully automated Amazon Comprehend topic modeling
API is the best option. If you are a data science specialist looking for
finer control over the various layers of building and tuning your own
topic modeling model, then the Amazon SageMaker NTM might work better
for you. For example, let’s say you are building a document topic
tagging application that needs a customized vocabulary, and you need the
ability to adjust the algorithm hyperparameters, such as the number of
layers of the neural network, so you can train a topic model that meets
the target accuracy in terms of coherence and uniqueness scores. In this
case, the Amazon SageMaker NTM would be the appropriate tool to use.

In this blog, we want to introduce 3 new features of the SageMaker NTM
that would help improve productivity, enhance topic coherence evaluation
capability, and speed up model training.

-  Auxiliary vocabulary channel
-  Word Embedding Topic Coherence (WETC) and Topic Uniqueness (TU)
   metrics
-  Subsampling data during training

In addition to these new features, by optimizing sparse operations and
the parameter server, we have improved the speed of the algorithm by 2x
for training and 4x for evaluation on a single GPU. The speedup is even
more significant for multi-gpu training.

--------------

Auxiliary vocabulary channel
----------------------------

When an NTM training job runs, it outputs the training status and
evaluation metrics to the CloudWatch logs. Among the outputs are lists
of top words detected for each of the learned topics. Prior to the
availability of auxiliary vocabulary channel support, the top words were
represented as integers, and customers needed to map the integers to an
external custom vocabulary lookup table in order to know what the actual
words were. With the support of auxiliary vocabulary channel, users can
now add a vocabulary file as an additional data input channel, and
SageMaker NTM will output the actual words in a topic instead of
integers. This feature eliminates the manual effort needed to map
integers to the actual vocabulary. Below is a sample of what a custom
vocabulary text file look like. The text file will simply contain a list
of words, one word per row, in the order corresponding to the integer
IDs provided in the data.

::

   absent
   absentee
   absolute
   absolutely

To include an auxiliary vocabulary for a training job, you should name
the vocabulary file ``vocab.txt`` and place it in the auxiliary channel.
See the code example below for how to add the auxiliary vocabulary file.
In this release we only support the UTF-8 encoding for the vocabulary
file.

Word Embedding Topic Coherence metrics
--------------------------------------

To evaluate the performance of an trained SageMaker NTM model, customers
can examine the perplexity metric emitted by the training job.
Sometimes, however customers also want to evaluate the topic coherence
of a model that measures the closeness of the words in a topic. A good
topic should have semantically similar words in it. Traditional methods
like the Normalized Point-wise Mutual Information(NPMI), while widely
accepted, require a large external corpus. The new WETC metric measures
the similarity of words in a topic by using a pre-trained word
embedding,
`Glove-6B-400K-50d <https://nlp.stanford.edu/projects/glove/>`__.

Intuitively, each word in the vocabulary is given a vector
representation (embedding). We compute the WETC of a topic by averaging
the pair-wise cosine similarities between the vectors corresponding to
the top words of the topic. Finally, we average the WETC for all the
topics to obtain a single score for the model.

Our tests have shown that WETC correlates very well with NPMI as an
effective surrogate. For details about the pair-wise WETC computation
and its correlation to NPMI, please refer to our paper `Coherence-Aware
Neural Topic Modeling, Ding et. al. 2018 (Accepted for EMNLP
2018) <https://arxiv.org/pdf/1809.02687.pdf>`__

WETC ranges between 0 and 1, higher is better. Typical value would be in
the range of 0.2 to 0.8. The WETC metric is evaluated whenever the
vocabulary file is provided. The average WETC score over the topics is
displayed in the log above the top words of all topics. The WETC metric
for each topic is also displayed along with the top words of each topic.
Please refer to the screenshot below for an example.

   Note in case many of the words in the supplied vocabulary cannot be
   found in the pre-trained word embedding, the WETC score can be
   misleading. Therefore we provide a warning message to alert the user
   exactly how many words in the vocabulary do not have an embedding:

::

   [09/07/2018 14:18:57 WARNING 140296605947712] 69 out of 16648 in vocabulary do not have embeddings! Default vector used for unknown embedding!

.. figure:: WETC_screenshot.png
   :alt: Log with WETC metrics

   Log with WETC metrics

Topic Uniqueness metric
-----------------------

A good topic modeling algorithm should generate topics that are unique
to avoid topic duplication. Customers who want to understand the topic
uniqueness of a trained Amazon SageMaker NTM model to evaluate its
quality can now use the new TU metric. To understand how TU works,
suppose there are K topics and we extract the top n words for each
topic, the TU for topic k is defined as |TU definition|

The range of the TU value is between 1/K and 1, where K is the number of
topics. A higher TU value represents higher topic uniqueness for the
topics detected.

The TU score is displayed regardless of the existence of a vocabulary
file. Similar to the WETC, the average TU score over the topics is
displayed in the log above the top words of all topics; the TU score for
each topic is also displayed along with the top words of each topic.
Please refer to the screenshot below for an example.

.. |TU definition| image:: TU_definition.png

.. figure:: TU_screenshot.png
   :alt: Log with TU metrics

   Log with TU metrics

Finally, we introduce a new hyperparameter for subsampling the data
during training

Subsampling data during training
--------------------------------

In typical online training, the entire training dataset is fed into the
training algorithm for each epoch. When the corpus is large, this leads
to long training time. With effective subsampling of the training
dataset, we can achieve faster model convergence while maintaining the
model performance. The new subsampling feature of the SageMaker NMT
allows customers to specify a percentage of training data used for
training using a new hyperparameter, ``sub_sample``. For example,
specifying 0.8 for ``sub_sample`` would direct SageMaker NTM to use 80%
of training data randomly for each epoch. As a result, the algorithm
will stochastically cover different subsets of data during different
epochs. You can configure this value in both the SageMaker console or
directly training code. See sample code below on how to set this value
for training.

::

   ntm.set_hyperparameters(num_topics=num_topics, feature_dim=vocab_size, mini_batch_size=128, 
                           epochs=100, sub_sample=0.7)

At the end of this notebook we will demonstrate that using subsampling
can reduce the overall training time for large dataset and potentially
achieve higher topic uniqueness and coherence.

--------------

Finally, to illustrate the new features, we will go through an example
with the public Wikitext launguage modeling dataset.

Data Preparation
----------------

The WikiText language modeling dataset is a collection of over 100
million tokens extracted from the set of verified Good and Featured
articles on Wikipedia. The dataset is available under the Creative
Commons Attribution-ShareAlike License. The dataset can be downloaded
from
`here <https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset>`__.
We will first use the wikitext-2 dataset.

   **Acknowledgements:** Stephen Merity, Caiming Xiong, James Bradbury,
   and Richard Socher. 2016. Pointer Sentinel Mixture Models

Fetching Data Set
~~~~~~~~~~~~~~~~~

First let’s define the folder to hold the data and clean the content in
it which might be from previous experiments.

.. code:: ipython3

    import os
    import shutil
            
    def check_create_dir(dir):
        if os.path.exists(dir):  # cleanup existing data folder
            shutil.rmtree(dir)
        os.mkdir(dir)
    
    dataset = 'wikitext-2'
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir,dataset)
    check_create_dir(data_dir)
    os.chdir(data_dir)
    print('Current directory: ', os.getcwd())

Now we can download and unzip the data. *Please review the following
Acknowledgements, Copyright Information, and Availability notice before
downloading the data.*

.. code:: ipython3

    # **Acknowledgements, Copyright Information, and Availability**
    # This dataset is available under the Creative Commons Attribution-ShareAlike License
    # Source: https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset
    
    !curl -O https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    !unzip wikitext-2-v1.zip

A sample of the ``wiki.valid.tokens`` is shown below. The datasets
contains markdown text with all documents (articles) concatenated.

::


    = Homarus gammarus =

    Homarus gammarus , known as the European lobster or common lobster , is a species of <unk> lobster from the eastern Atlantic Ocean , Mediterranean Sea and parts of the Black Sea . It is closely related to the American lobster , H. americanus . It may grow to a length of 60 cm ( 24 in ) and a mass of 6 kilograms ( 13 lb ) , and bears a conspicuous pair of claws . In life , the lobsters are blue , only becoming " lobster red " on cooking . Mating occurs in the summer , producing eggs which are carried by the females for up to a year before hatching into <unk> larvae . Homarus gammarus is a highly esteemed food , and is widely caught using lobster pots , mostly around the British Isles .

    = = Description = =

    Homarus gammarus is a large <unk> , with a body length up to 60 centimetres ( 24 in ) and weighing up to 5 – 6 kilograms ( 11 – 13 lb ) , although the lobsters caught in lobster pots are usually 23 – 38 cm ( 9 – 15 in ) long and weigh 0 @.@ 7 – 2 @.@ 2 kg ( 1 @.@ 5 – 4 @.@ 9 lb ) . Like other crustaceans , lobsters have a hard <unk> which they must shed in order to grow , in a process called <unk> ( <unk> ) . This may occur several times a year for young lobsters , but decreases to once every 1 – 2 years for larger animals .

Preprocessing
~~~~~~~~~~~~~

We need to first parse the input files into separate documents. We can
identify each document by its title in level-1 heading. Additional care
is taken to check that the line containing the title should be
sandwiched by blank lines to avoid false detection of document titles.

.. code:: ipython3

    def is_document_start(line):
        if len(line) < 4:
            return False
        if line[0] is '=' and line[-1] is '=':
            if line[2] is not '=':
                return True
            else:
                return False
        else:
            return False
    
    
    def token_list_per_doc(input_dir, token_file):
        lines_list = []
        line_prev = ''
        prev_line_start_doc = False
        with open(os.path.join(input_dir, token_file), 'r', encoding='utf-8') as f:
            for l in f:
                line = l.strip()
                if prev_line_start_doc and line:
                    # the previous line should not have been start of a document!
                    lines_list.pop()
                    lines_list[-1] = lines_list[-1] + ' ' + line_prev
    
                if line:
                    if is_document_start(line) and not line_prev:
                        lines_list.append(line)
                        prev_line_start_doc = True
                    else:
                        lines_list[-1] = lines_list[-1] + ' ' + line
                        prev_line_start_doc = False
                else:
                    prev_line_start_doc = False
                line_prev = line
    
        print("{} documents parsed!".format(len(lines_list)))
        return lines_list
    
    input_dir = os.path.join(data_dir, dataset)
    train_file = 'wiki.train.tokens'
    val_file = 'wiki.valid.tokens'
    test_file = 'wiki.test.tokens'
    train_doc_list = token_list_per_doc(input_dir, train_file)
    val_doc_list = token_list_per_doc(input_dir, val_file)
    test_doc_list = token_list_per_doc(input_dir, test_file)

In the following cell, we use a lemmatizer from ``nltk``. In the list
comprehension, we implement a simple rule: only consider words that are
longer than 2 characters, start with a letter and match the
``token_pattern``.

.. code:: ipython3

    !pip install nltk
    import nltk
    # nltk.download('punkt')
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer 
    import re
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in doc.split() if len(t) >= 2 and re.match("[a-z].*",t) 
                    and re.match(token_pattern, t)]

We perform lemmatizing and counting next.

.. code:: ipython3

    import time
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    
    print('Lemmatizing and counting, this may take a few minutes...')
    start_time = time.time()
    vectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english',
                                 tokenizer=LemmaTokenizer(), max_df=0.9, min_df=3)
    
    train_vectors = vectorizer.fit_transform(train_doc_list)
    val_vectors = vectorizer.transform(val_doc_list)
    test_vectors = vectorizer.transform(test_doc_list)
    
    vocab_list = vectorizer.get_feature_names()
    vocab_size = len(vocab_list)
    print('vocab size:', vocab_size)
    print('Done. Time elapsed: {:.2f}s'.format(time.time() - start_time))

Because all the parameters (weights and biases) in the NTM model are
``np.float32`` type we’d need the input data to also be in
``np.float32``. It is better to do this type-casting upfront rather than
repeatedly casting during mini-batch training.

.. code:: ipython3

    import scipy.sparse as sparse
    
    def shuffle_and_dtype(vectors):
        idx = np.arange(vectors.shape[0])
        np.random.shuffle(idx)
        vectors = vectors[idx]
        vectors = sparse.csr_matrix(vectors, dtype=np.float32)
        print(type(vectors), vectors.dtype)
        return vectors
    
    train_vectors = shuffle_and_dtype(train_vectors)
    val_vectors = shuffle_and_dtype(val_vectors)
    test_vectors = shuffle_and_dtype(test_vectors)

The NTM algorithm, as well as other first-party SageMaker algorithms,
accepts data in
`RecordIO <https://mxnet.apache.org/api/python/io/io.html#module-mxnet.recordio>`__
`Protobuf <https://developers.google.com/protocol-buffers/>`__ format.
Here we define a helper function to convert the data to RecordIO
Protobuf format. In addition, we will have the option to split the data
into several parts specified by ``n_parts``.

The algorithm inherently supports multiple files in the training folder
(“channel”), which could be very helpful for large data sets. In
addition, when we use distributed training with multiple workers
(compute instances), having multiple files allows us to distribute
different portions of the training data to different workers
conveniently.

Inside this helper function we use ``write_spmatrix_to_sparse_tensor``
function provided by `SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__ to convert scipy
sparse matrix into RecordIO Protobuf format.

.. code:: ipython3

    def split_convert(sparray, prefix, fname_template='data_part{}.pbr', n_parts=2):
        import io
        import sagemaker.amazon.common as smac
    
        chunk_size = sparray.shape[0] // n_parts
        for i in range(n_parts):
    
            # Calculate start and end indices
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if i + 1 == n_parts:
                end = sparray.shape[0]
    
            # Convert to record protobuf
            buf = io.BytesIO()
            smac.write_spmatrix_to_sparse_tensor(array=sparray[start:end], file=buf, labels=None)
            buf.seek(0)
    
            fname = os.path.join(prefix, fname_template.format(i))
            with open(fname, 'wb') as f:
                f.write(buf.getvalue())
            print('Saved data to {}'.format(fname))
            
    train_data_dir = os.path.join(data_dir, 'train')
    val_data_dir = os.path.join(data_dir, 'validation')
    test_data_dir = os.path.join(data_dir, 'test')
    
    check_create_dir(train_data_dir)
    check_create_dir(val_data_dir)
    check_create_dir(test_data_dir)
    
    split_convert(train_vectors, prefix=train_data_dir, fname_template='train_part{}.pbr', n_parts=4)
    split_convert(val_vectors, prefix=val_data_dir, fname_template='val_part{}.pbr', n_parts=1)
    split_convert(test_vectors, prefix=test_data_dir, fname_template='test_part{}.pbr', n_parts=1)

Save the vocabulary file
~~~~~~~~~~~~~~~~~~~~~~~~

To make use of the auxiliary channel for vocabulary file, we first save
the text file with the name ``vocab.txt`` in the auxiliary directory.

.. code:: ipython3

    aux_data_dir = os.path.join(data_dir, 'auxiliary')
    check_create_dir(aux_data_dir)
    with open(os.path.join(aux_data_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for item in vocab_list:
            f.write(item+'\n')

Store Data on S3
~~~~~~~~~~~~~~~~

Below we upload the data to an Amazon S3 destination for the model to
access it during training.

Setup AWS Credentials
^^^^^^^^^^^^^^^^^^^^^

We first need to specify data locations and access roles. **This is the
only cell of this notebook that you will need to edit.** In particular,
we need the following data:

-  The S3 ``bucket`` and ``prefix`` that you want to use for training
   and model data. This should be within the same region as the Notebook
   Instance, training, and hosting.
-  The IAM ``role`` is used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the boto regexp with a the appropriate full
   IAM role arn string(s).

.. code:: ipython3

    import os
    import sagemaker
    
    role = sagemaker.get_execution_role()
    
    bucket = sagemaker.Session().default_bucket() #<or insert your own bucket name>#
    prefix = 'ntm/' + dataset
    
    train_prefix = os.path.join(prefix, 'train')
    val_prefix = os.path.join(prefix, 'val')
    aux_prefix = os.path.join(prefix, 'auxiliary')
    test_prefix = os.path.join(prefix, 'test')
    output_prefix = os.path.join(prefix, 'output')
    
    s3_train_data = os.path.join('s3://', bucket, train_prefix)
    s3_val_data = os.path.join('s3://', bucket, val_prefix)
    s3_aux_data = os.path.join('s3://', bucket, aux_prefix)
    s3_test_data = os.path.join('s3://', bucket, test_prefix)
    output_path = os.path.join('s3://', bucket, output_prefix)
    print('Training set location', s3_train_data)
    print('Validation set location', s3_val_data)
    print('Auxiliary data location', s3_aux_data)
    print('Test data location', s3_test_data)
    print('Trained model will be saved at', output_path)

Upload the input directories to s3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the ``aws`` command line interface (CLI) to upload the various
input channels.

.. code:: ipython3

    import subprocess
    
    cmd_train = 'aws s3 cp ' + train_data_dir + ' ' + s3_train_data + ' --recursive' 
    p=subprocess.Popen(cmd_train, shell=True,stdout=subprocess.PIPE)
    p.communicate()

.. code:: ipython3

    cmd_val = 'aws s3 cp ' + val_data_dir + ' ' + s3_val_data + ' --recursive' 
    p=subprocess.Popen(cmd_val, shell=True,stdout=subprocess.PIPE)
    p.communicate()

.. code:: ipython3

    cmd_test = 'aws s3 cp ' + test_data_dir + ' ' + s3_test_data + ' --recursive' 
    p=subprocess.Popen(cmd_test, shell=True,stdout=subprocess.PIPE)
    p.communicate()

.. code:: ipython3

    cmd_aux = 'aws s3 cp ' + aux_data_dir + ' ' + s3_aux_data + ' --recursive' 
    p=subprocess.Popen(cmd_aux, shell=True,stdout=subprocess.PIPE)
    p.communicate()

Model Training
~~~~~~~~~~~~~~

We have prepared the train, validation, test and auxiliary input
channels on s3. Next, we configure a SageMaker training job to use the
NTM algorithm on the data we prepared.

SageMaker uses Amazon Elastic Container Registry (ECR) docker container
to host the NTM training image. The following ECR containers are
currently available for SageMaker NTM training in different regions. For
the latest Docker container registry please refer to `Amazon SageMaker:
Common
Parameters <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html>`__.

.. code:: ipython3

    import boto3
    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'ntm')

The code in the cell below automatically chooses an algorithm container
based on the current region. In the API call to
``sagemaker.estimator.Estimator`` we also specify the type and count of
instances for the training job. Because the wikitext-2 data set is
relatively small, we have chosen a CPU only instance (``ml.c4.xlarge``),
but do feel free to change to `other instance
types <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__. NTM
fully takes advantage of GPU hardware and in general trains roughly an
order of magnitude faster on a GPU than on a CPU. Multi-GPU or
multi-instance training further improves training speed roughly linearly
if communication overhead is low compared to compute time.

.. code:: ipython3

    import sagemaker
    sess = sagemaker.Session()
    ntm = sagemaker.estimator.Estimator(container,
                                        role, 
                                        train_instance_count=1, 
                                        train_instance_type='ml.c4.xlarge',
                                        output_path=output_path,
                                        sagemaker_session=sess)

We can specify the hyperparameters, including the newly introduced
``sub_sample``.

.. code:: ipython3

    num_topics = 20
    ntm.set_hyperparameters(num_topics=num_topics, feature_dim=vocab_size, mini_batch_size=60, 
                            epochs=50, sub_sample=0.7)

Next, we need to specify how the data will be distributed to the workers
during training as well as their content type.

.. code:: ipython3

    from sagemaker.session import s3_input
    s3_train = s3_input(s3_train_data, distribution='ShardedByS3Key',
                        content_type='application/x-recordio-protobuf')
    s3_val = s3_input(s3_val_data, distribution='FullyReplicated',
                      content_type='application/x-recordio-protobuf')
    s3_test = s3_input(s3_test_data, distribution='FullyReplicated',
                      content_type='application/x-recordio-protobuf')
    
    s3_aux = s3_input(s3_aux_data, distribution='FullyReplicated', content_type='text/plain')

We are ready to run the training job. Again, we will notice in the log
that the top words are printed together with the WETC and TU scores.

.. code:: ipython3

    ntm.fit({'train': s3_train, 'validation': s3_val, 'auxiliary': s3_aux, 'test': s3_test})

Once the job is completed, you can view information about and the status
of a training job using the AWS SageMaker console. Just click on the
“Jobs” tab and select training job matching the training job name,
below:

.. code:: ipython3

    print('Training job name: {}'.format(ntm.latest_training_job.job_name))

We demonstrate the utility of the ``sub_sample`` hyperparameter by
setting it to 1.0 and 0.2 for training on the wikitext-103 dataset
(simply change the dataset name and download URL, re-start the kernel
and re-run this notebook). In both settings, we set ``epochs = 100`` and
NTM would early-exit training when the loss on validation data does not
improve in 3 consecutive epochs. We report the TU, WETC, NPMI of the
best epoch based on validation loss as well as the total time for both
settings below. Note we ran each training job on a single GPU of a
``p2.8xlarge`` machine. |subsample_result_table| We observe that setting
sub_sample to 0.2 leads to reduced total training time even though it
takes more epochs to converge (49 instead of 18). The increase in the
number of epochs to convergence is expected due to the variance
introduced by training on a random subset of data per epoch. Yet the
overall training time is reduced because training is about 5 times
faster per epoch at the subsampling rate of 0.2. We also note the higher
scores in terms of TU, WETC and NPMI at the end of training with
subsampling.

.. |subsample_result_table| image:: subsample_table.png

Conclusion
==========

In this blog post, we introduced 3 new features of SageMaker NTM
algorithm. Upon reading this blog and completing the new sample
notebook, you should have learned how to add an auxiliary vocabulary
channel to automatically map integer representations of words in the
detected topics to a human understandable vocabulary. You also have
learned to evaluate the quality of the a model using both Word Embedding
Topic Coherence and Topic Uniqueness metrics. Lastly, you learned to use
the subsampling feature to reduce the model training time while
maintaining similar model performance.
