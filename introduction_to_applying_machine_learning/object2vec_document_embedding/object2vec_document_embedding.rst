Document Embedding with Amazon SageMaker Object2Vec
===================================================

1.  `Introduction <#Introduction>`__
2.  `Background <#Background>`__
3.  `Embedding documents using
    Object2Vec <#Embedding-documents-using-Object2Vec>`__
4.  `Download and preprocess Wikipedia
    data <#Download-and-preprocess-Wikipedia-data>`__
5.  `Install and load dependencies <#Install-and-load-dependencies>`__
6.  `Build vocabulary and tokenize
    datasets <#Build-vocabulary-and-tokenize-datasets>`__
7.  `Upload preprocessed data to S3 <#Upload-preprocessed-data-to-S3>`__
8.  `Define SageMaker session, Object2Vec image, S3 input and output
    paths <#Define-SageMaker-session,-Object2Vec-image,-S3-input-and-output-paths>`__
9.  `Train and deploy doc2vec <#Train-and-deploy-doc2vec>`__
10. `Learning performance boost with new
    features <#Learning-performance-boost-with-new-features>`__
11. `Training speedup with sparse gradient
    update <#Training-speedup-with-sparse-gradient-update>`__
12. `Apply learned embeddings to document retrieval
    task <#Apply-learned-embeddings-to-document-retrieval-task>`__
13. `Comparison with the StarSpace
    algorithm <#Comparison-with-the-StarSpace-algorithm>`__

Introduction
------------

In this notebook, we introduce four new features to Object2Vec, a
general-purpose neural embedding algorithm: negative sampling, sparse
gradient update, weight-sharing, and comparator operator customization.
The new features together broaden the applicability of Object2Vec,
improve its training speed and accuracy, and provide users with greater
flexibility. See `Introduction to the Amazon SageMaker
Object2Vec <https://aws.amazon.com/blogs/machine-learning/introduction-to-amazon-sagemaker-object2vec/>`__
if you aren’t already familiar with Object2Vec.

We demonstrate how these new features extend the applicability of
Object2Vec to a new Document Embedding use-case: A customer has a large
collection of documents. Instead of storing these documents in its raw
format or as sparse bag-of-words vectors, to achieve training efficiency
in the various downstream tasks, she would like to instead embed all
documents in a common low-dimensional space, so that the semantic
distance between these documents are preserved.

Background
----------

Object2Vec is a highly customizable multi-purpose algorithm that can
learn embeddings of pairs of objects. The embeddings are learned such
that it preserves their pairwise similarities in the original space.

-  Similarity is user-defined: users need to provide the algorithm with
   pairs of objects that they define as similar (1) or dissimilar (0);
   alternatively, the users can define similarity in a continuous sense
   (provide a real-valued similarity score).

-  The learned embeddings can be used to efficiently compute nearest
   neighbors of objects, as well as to visualize natural clusters of
   related objects in the embedding space. In addition, the embeddings
   can also be used as features of the corresponding objects in
   downstream supervised tasks such as classification or regression.

Embedding documents using Object2Vec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We demonstrate how, with the new features, Object2Vec can be used to
embed a large collection of documents into vectors in the same latent
space.

Similar to the widely used Word2Vec algorithm for word embedding, a
natural approach to document embedding is to preprocess documents as
(sentence, context) pairs, where the sentence and its matching context
come from the same document. The matching context is the entire document
with the given sentence removed. The idea is to embed both sentence and
context into a low dimensional space such that their mutual similarity
is maximized, since they belong to the same document and therefore
should be semantically related. The learned encoder for the context can
then be used to encode new documents into the same embedding space. In
order to train the encoders for sentences and documents, we also need
negative (sentence, context) pairs so that the model can learn to
discriminate between semantically similar and dissimilar pairs. It is
easy to generate such negatives by pairing sentences with documents that
they do not belong to. Since there are many more negative pairs than
positives in naturally occurring data, we typically resort to random
sampling techniques to achieve a balance between positive and negative
pairs in the training data. The figure below shows pictorially how the
positive pairs and negative pairs are generated from unlabeled data for
the purpose of learning embeddings for documents (and sentences).



We show how Object2Vec with the new *negative sampling feature* can be
applied to the document embedding use-case. In addition, we show how the
other new features, namely, *weight-sharing*, *customization of
comparator operator*, and *sparse gradient update*, together enhance the
algorithm’s performance and user-experience in and beyond this use-case.
Sections `Learning performance boost with new
features <#Learning-performance-boost-with-new-features>`__ and
`Training speedup with sparse gradient
update <#Training-speedup-with-sparse-gradient-update>`__ in this
notebook provide a detailed introduction to the new features.

Download and preprocess Wikipedia data
--------------------------------------

Please be aware of the following requirements about the acknowledgment,
copyright and availability, cited from the `data source description
page <https://github.com/facebookresearch/StarSpace/blob/master/LICENSE.md>`__.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   “Software”), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions: The above copyright notice and this
   permission notice shall be included in all copies or substantial
   portions of the Software.

.. code:: bash

    %%bash
    
    DATANAME="wikipedia"
    DATADIR="/tmp/wiki"
    
    mkdir -p "${DATADIR}"
    
    if [ ! -f "${DATADIR}/${DATANAME}_train250k.txt" ]
    then
        echo "Downloading wikipedia data"
        wget --quiet -c "https://dl.fbaipublicfiles.com/starspace/wikipedia_train250k.tgz" -O "${DATADIR}/${DATANAME}_train.tar.gz"
        tar -xzvf "${DATADIR}/${DATANAME}_train.tar.gz" -C "${DATADIR}"
        wget --quiet -c "https://dl.fbaipublicfiles.com/starspace/wikipedia_devtst.tgz" -O "${DATADIR}/${DATANAME}_test.tar.gz"
        tar -xzvf "${DATADIR}/${DATANAME}_test.tar.gz" -C "${DATADIR}"
    fi


.. code:: ipython3

    datadir = '/tmp/wiki'

.. code:: ipython3

    !ls /tmp/wiki

Install and load dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !pip install jsonlines

.. code:: ipython3

    # note: please run on python 3 kernel
    
    import os
    import random
    
    import math
    import scipy
    import numpy as np
    
    import re
    import string
    import json, jsonlines
    
    from collections import defaultdict
    from collections import Counter
    
    from itertools import chain, islice
    
    from nltk.tokenize import TreebankWordTokenizer
    from sklearn.preprocessing import normalize
    
    ## sagemaker api
    import sagemaker, boto3
    from sagemaker.session import s3_input
    from sagemaker.predictor import json_serializer, json_deserializer

Build vocabulary and tokenize datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    BOS_SYMBOL = "<s>"
    EOS_SYMBOL = "</s>"
    UNK_SYMBOL = "<unk>"
    PAD_SYMBOL = "<pad>"
    PAD_ID = 0
    TOKEN_SEPARATOR = " "
    VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]
    
    
    ##### utility functions for preprocessing
    def get_article_iter_from_file(fname):
        with open(fname) as f:
            for article in f:
                yield article
    
    def get_article_iter_from_channel(channel, datadir='/tmp/wiki'):
        if channel == 'train':
            fname = os.path.join(datadir, 'wikipedia_train250k.txt')
            return get_article_iter_from_file(fname)
        else:
            iterlist = []
            suffix_list = ['train250k.txt', 'test10k.txt', 'dev10k.txt', 'test_basedocs.txt']
            for suffix in suffix_list:
                fname = os.path.join(datadir, 'wikipedia_'+suffix)
                iterlist.append(get_article_iter_from_file(fname))
            return chain.from_iterable(iterlist)
    
    
    def readlines_from_article(article):
        return article.strip().split('\t')
    
    
    def sentence_to_integers(sentence, word_dict, trim_size=None):
        """
        Converts a string of tokens to a list of integers
        """
        if not trim_size:
            return [word_dict[token] if token in word_dict else 0 for token in get_tokens_from_sentence(sentence)]
        else:
            integer_list = []
            for token in get_tokens_from_sentence(sentence):
                if len(integer_list) < trim_size:
                    if token in word_dict:
                        integer_list.append(word_dict[token])
                    else:
                        integer_list.append(0)
                else:
                    break
            return integer_list
    
    
    def get_tokens_from_sentence(sent):
        """
        Yields tokens from input string.
    
        :param line: Input string.
        :return: Iterator over tokens.
        """
        for token in sent.split():
            if len(token) > 0:
                yield normalize_token(token)
    
    
    def get_tokens_from_article(article):
        iterlist = []
        for sent in readlines_from_article(article):
            iterlist.append(get_tokens_from_sentence(sent))
        return chain.from_iterable(iterlist)
    
    
    def normalize_token(token):
        token = token.lower()
        if all(s.isdigit() or s in string.punctuation for s in token):
            tok = list(token)
            for i in range(len(tok)):
                if tok[i].isdigit():
                    tok[i] = '0'
            token = "".join(tok)
        return token

.. code:: ipython3

    # function to build vocabulary
    
    def build_vocab(channel, num_words=50000, min_count=1, use_reserved_symbols=True, sort=True):
        """
        Creates a vocabulary mapping from words to ids. Increasing integer ids are assigned by word frequency,
        using lexical sorting as a tie breaker. The only exception to this are special symbols such as the padding symbol
        (PAD).
    
        :param num_words: Maximum number of words in the vocabulary.
        :param min_count: Minimum occurrences of words to be included in the vocabulary.
        :return: word-to-id mapping.
        """
        vocab_symbols_set = set(VOCAB_SYMBOLS)
        raw_vocab = Counter()
        for article in get_article_iter_from_channel(channel):
            article_wise_vocab_list = list()
            for token in get_tokens_from_article(article):
                if token not in vocab_symbols_set:
                    article_wise_vocab_list.append(token)
            raw_vocab.update(article_wise_vocab_list)
    
        print("Initial vocabulary: {} types".format(len(raw_vocab)))
    
        # For words with the same count, they will be ordered reverse alphabetically.
        # Not an issue since we only care for consistency
        pruned_vocab = sorted(((c, w) for w, c in raw_vocab.items() if c >= min_count), reverse=True)
        print("Pruned vocabulary: {} types (min frequency {})".format(len(pruned_vocab), min_count))
    
        # truncate the vocabulary to fit size num_words (only includes the most frequent ones)
        vocab = islice((w for c, w in pruned_vocab), num_words)
    
        if sort:
            # sort the vocabulary alphabetically
            vocab = sorted(vocab)
        if use_reserved_symbols:
            vocab = chain(VOCAB_SYMBOLS, vocab)
    
        word_to_id = {word: idx for idx, word in enumerate(vocab)}
    
        print("Final vocabulary: {} types".format(len(word_to_id)))
    
        if use_reserved_symbols:
            # Important: pad symbol becomes index 0
            assert word_to_id[PAD_SYMBOL] == PAD_ID
        
        return word_to_id

.. code:: ipython3

    # build vocab dictionary
    
    def build_vocabulary_file(vocab_fname, channel, num_words=50000, min_count=1, 
                              use_reserved_symbols=True, sort=True, force=False):
        if not os.path.exists(vocab_fname) or force:
            w_dict = build_vocab(channel, num_words=num_words, min_count=min_count, 
                                 use_reserved_symbols=True, sort=True)
            with open(vocab_fname, "w") as write_file:
                json.dump(w_dict, write_file)
    
    channel = 'train'
    min_count = 5
    vocab_fname = os.path.join(datadir, 'wiki-vocab-{}250k-mincount-{}.json'.format(channel, min_count))
    
    build_vocabulary_file(vocab_fname, channel, num_words=500000, min_count=min_count, force=True)

.. code:: ipython3

    print("Loading vocab file {} ...".format(vocab_fname))
    
    with open(vocab_fname) as f:
        w_dict = json.load(f)
        print("The vocabulary size is {}".format(len(w_dict.keys())))

.. code:: ipython3

    # Functions to build training data 
    # Tokenize wiki articles to (sentence, document) pairs
    def generate_sent_article_pairs_from_single_article(article, word_dict):
        sent_list = readlines_from_article(article)
        art_len = len(sent_list)
        idx = random.randint(0, art_len-1)
        wrapper_text_idx = list(range(idx)) + list(range((idx+1) % art_len, art_len))
        wrapper_text_list = sent_list[:idx] + sent_list[(idx+1) % art_len : art_len]
        wrapper_tokens = []
        for sent1 in wrapper_text_list:
            wrapper_tokens += sentence_to_integers(sent1, word_dict)
        sent_tokens = sentence_to_integers(sent_list[idx], word_dict)
        yield {'in0':sent_tokens, 'in1':wrapper_tokens, 'label':1}
    
    
    def generate_sent_article_pairs_from_single_file(fname, word_dict):
        with open(fname) as reader:
            iter_list = []
            for article in reader:
                iter_list.append(generate_sent_article_pairs_from_single_article(article, word_dict))
        return chain.from_iterable(iter_list)

.. code:: ipython3

    # Build training data
    
    # Generate integer positive labeled data
    train_prefix = 'train250k'
    fname = "wikipedia_{}.txt".format(train_prefix)
    outfname = os.path.join(datadir, '{}_tokenized.jsonl'.format(train_prefix))
    counter = 0
    
    with jsonlines.open(outfname, 'w') as writer:
        for sample in generate_sent_article_pairs_from_single_file(os.path.join(datadir, fname), w_dict):
            writer.write(sample)
            counter += 1
            
    print("Finished generating {} data of size {}".format(train_prefix, counter))

.. code:: ipython3

    # Shuffle training data
    !shuf {outfname} > {train_prefix}_tokenized_shuf.jsonl

.. code:: ipython3

    ## Function to generate dev/test data (with both positive and negative labels)
    
    def generate_pos_neg_samples_from_single_article(word_dict, article_idx, article_buffer, negative_sampling_rate=1):
        sample_list = []
        # generate positive samples
        sent_list = readlines_from_article(article_buffer[article_idx])
        art_len = len(sent_list)
        idx = random.randint(0, art_len-1)
        wrapper_text_idx = list(range(idx)) + list(range((idx+1) % art_len, art_len))
        wrapper_text_list = sent_list[:idx] + sent_list[(idx+1) % art_len : art_len]
        wrapper_tokens = []
        for sent1 in wrapper_text_list:
            wrapper_tokens += sentence_to_integers(sent1, word_dict)
        sent_tokens = sentence_to_integers(sent_list[idx], word_dict)
        sample_list.append({'in0':sent_tokens, 'in1':wrapper_tokens, 'label':1})
        # generate negative sample
        buff_len = len(article_buffer)
        sampled_inds = np.random.choice(list(range(article_idx)) + list(range((article_idx+1) % buff_len, buff_len)), 
                                        size=negative_sampling_rate)
        for n_idx in sampled_inds:
            other_article = article_buffer[n_idx]
            context_list = readlines_from_article(other_article)
            context_tokens = []
            for sent2 in context_list:
                context_tokens += sentence_to_integers(sent2, word_dict)
            sample_list.append({'in0': sent_tokens, 'in1':context_tokens, 'label':0})
        return sample_list

.. code:: ipython3

    # Build dev and test data
    for data in ['dev10k', 'test10k']:
        fname = os.path.join(datadir,'wikipedia_{}.txt'.format(data))
        test_nsr = 5
        outfname = '{}_tokenized-nsr{}.jsonl'.format(data, test_nsr)
        article_buffer = list(get_article_iter_from_file(fname))
        sample_buffer = []
        for article_idx in range(len(article_buffer)):
            sample_buffer += generate_pos_neg_samples_from_single_article(w_dict, article_idx, 
                                                                          article_buffer, 
                                                                          negative_sampling_rate=test_nsr)
        with jsonlines.open(outfname, 'w') as writer:
            writer.write_all(sample_buffer)

Upload preprocessed data to S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    TRAIN_DATA="train250k_tokenized_shuf.jsonl"
    DEV_DATA="dev10k_tokenized-nsr{}.jsonl".format(test_nsr)
    TEST_DATA="test10k_tokenized-nsr{}.jsonl".format(test_nsr)
    
    # NOTE: define your s3 bucket and key here
    bucket = '<YOUR S3 BUCKET>'
    S3_KEY = 'object2vec-doc2vec'
    


.. code:: bash

    %%bash -s "$TRAIN_DATA" "$DEV_DATA" "$TEST_DATA" "$bucket" "$S3_KEY"
    
    aws s3 cp "$1" s3://$4/$5/input/train/
    aws s3 cp "$2" s3://$4/$5/input/validation/
    aws s3 cp "$3" s3://$4/$5/input/test/

Define Sagemaker session, Object2Vec image, S3 input and output paths
---------------------------------------------------------------------

.. code:: ipython3

    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    
    region = boto3.Session().region_name
    print("Your notebook is running on region '{}'".format(region))
    
    sess = sagemaker.Session()
    
     
    role = get_execution_role()
    print("Your IAM role: '{}'".format(role))
    
    container = get_image_uri(region, 'object2vec')
    print("The image uri used is '{}'".format(container))
    
    print("Using s3 buceket: {} and key prefix: {}".format(bucket, S3_KEY))

.. code:: ipython3

    ## define input channels
    
    s3_input_path = os.path.join('s3://', bucket, S3_KEY, 'input')
    
    s3_train = s3_input(os.path.join(s3_input_path, 'train', TRAIN_DATA), 
                        distribution='ShardedByS3Key', content_type='application/jsonlines')
    
    s3_valid = s3_input(os.path.join(s3_input_path, 'validation', DEV_DATA), 
                        distribution='ShardedByS3Key', content_type='application/jsonlines')
    
    s3_test = s3_input(os.path.join(s3_input_path, 'test', TEST_DATA), 
                       distribution='ShardedByS3Key', content_type='application/jsonlines')

.. code:: ipython3

    ## define output path
    output_path = os.path.join('s3://', bucket, S3_KEY, 'models')

Train and deploy doc2vec
------------------------

We combine four new features into our training of Object2Vec:

-  Negative sampling: With the new ``negative_sampling_rate``
   hyperparameter, users of Object2Vec only need to provide positively
   labeled data pairs, and the algorithm automatically samples for
   negative data internally during training.

-  Weight-sharing of embedding layer: The new
   ``tied_token_embedding_weight`` hyperparameter gives user the
   flexibility to share the embedding weights for both encoders, and it
   improves the performance of the algorithm in this use-case

-  The new ``comparator_list`` hyperparameter gives users the
   flexibility to mix-and-match different operators so that they can
   tune the algorithm towards optimal performance for their
   applications.

Learning performance boost with new features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Table 1* below shows the effect of these features on these two metrics
evaluated on a test set obtained from the same data creation process.

We see that when negative sampling and weight-sharing of embedding layer
is on, and when we use a customized comparator operator (Hadamard
product), the model has improved test performance. When all these
features are combined together (last row of Table 1), the algorithm has
the best performance as measured by accuracy and cross-entropy.

Table 1
~~~~~~~

+---------------+------------+--------------+------------+-----------+
| negative_samp | weight-sha | comparator   | Test       | Test      |
| ling_rate     | ring       | operator     | accuracy   | cross-ent |
|               |            |              |            | ropy      |
+===============+============+==============+============+===========+
| off           | off        | default      | 0.167      | 23        |
+---------------+------------+--------------+------------+-----------+
| 3             | off        | default      | 0.92       | 0.21      |
+---------------+------------+--------------+------------+-----------+
| 5             | off        | default      | 0.92       | 0.19      |
+---------------+------------+--------------+------------+-----------+
| off           | on         | default      | 0.167      | 23        |
+---------------+------------+--------------+------------+-----------+
| 3             | on         | default      | 0.93       | 0.18      |
+---------------+------------+--------------+------------+-----------+
| 5             | on         | default      | 0.936      | 0.17      |
+---------------+------------+--------------+------------+-----------+
| off           | on         | customized   | 0.17       | 23        |
+---------------+------------+--------------+------------+-----------+
| 3             | on         | customized   | 0.93       | 0.18      |
+---------------+------------+--------------+------------+-----------+
| 5             | on         | customized   | 0.94       | 0.17      |
+---------------+------------+--------------+------------+-----------+

-  The new ``token_embedding_storage_type`` hyperparameter flags the use
   of sparse gradient update, which takes advantage of the sparse input
   format of Object2Vec. We tested and summarized the training speedup
   with different GPU and ``max_seq_len`` configurations in the table
   below. In a word, we see 2-20 times speed up on different machine and
   algorithm configurations.

Training speedup with sparse gradient update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Table 2* below shows the training speeds up with sparse gradient update
feature turned on, as a function of number of GPUs used for training.

Table 2
~~~~~~~

+---------------+------------+--------------+------------+-----------+
| num_gpus      | Throughput | Throughput   | max_seq_le | Speedup   |
|               | (samples/s | with sparse  | n          | X-times   |
|               | ec)        | storage      | (in0/in1)  |           |
|               | with dense |              |            |           |
|               | storage    |              |            |           |
+===============+============+==============+============+===========+
| 1             | 5k         | 14k          | 50         | 2.8       |
+---------------+------------+--------------+------------+-----------+
| 2             | 2.7k       | 23k          | 50         | 8.5       |
+---------------+------------+--------------+------------+-----------+
| 3             | 2k         | 23~26k       | 50         | 10        |
+---------------+------------+--------------+------------+-----------+
| 4             | 2k         | 23k          | 50         | 10        |
+---------------+------------+--------------+------------+-----------+
| 8             | 1.1k       | 19k~20k      | 50         | 20        |
+---------------+------------+--------------+------------+-----------+
| 1             | 1.1k       | 2k           | 500        | 2         |
+---------------+------------+--------------+------------+-----------+
| 2             | 1.5k       | 3.6k         | 500        | 2.4       |
+---------------+------------+--------------+------------+-----------+
| 4             | 1.6k       | 6k           | 500        | 3.75      |
+---------------+------------+--------------+------------+-----------+
| 6             | 1.3k       | 6.7k         | 500        | 5.15      |
+---------------+------------+--------------+------------+-----------+
| 8             | 1.1k       | 5.6k         | 500        | 5         |
+---------------+------------+--------------+------------+-----------+

.. code:: ipython3

    # Define training hyperparameters
    
    hyperparameters = {
          "_kvstore": "device",
          "_num_gpus": 'auto',
          "_num_kv_servers": "auto",
          "bucket_width": 0,
          "dropout": 0.4,
          "early_stopping_patience": 2,
          "early_stopping_tolerance": 0.01,
          "enc0_layers": "auto",
          "enc0_max_seq_len": 50,
          "enc0_network": "pooled_embedding",
          "enc0_pretrained_embedding_file": "",
          "enc0_token_embedding_dim": 300,
          "enc0_vocab_size": 267522,
          "enc1_network": "enc0",
          "enc_dim": 300,
          "epochs": 20,
          "learning_rate": 0.01,
          "mini_batch_size": 512,
          "mlp_activation": "relu",
          "mlp_dim": 512,
          "mlp_layers": 2,
          "num_classes": 2,
          "optimizer": "adam",
          "output_layer": "softmax",
          "weight_decay": 0
    }
    
    
    hyperparameters['negative_sampling_rate'] = 3
    hyperparameters['tied_token_embedding_weight'] = "true"
    hyperparameters['comparator_list'] = "hadamard"
    hyperparameters['token_embedding_storage_type'] = 'row_sparse'
    
        
    # get estimator
    doc2vec = sagemaker.estimator.Estimator(container,
                                              role, 
                                              train_instance_count=1, 
                                              train_instance_type='ml.p2.xlarge',
                                              output_path=output_path,
                                              sagemaker_session=sess)
    


.. code:: ipython3

    # set hyperparameters
    doc2vec.set_hyperparameters(**hyperparameters)
    
    # fit estimator with data
    doc2vec.fit({'train': s3_train, 'validation':s3_valid, 'test':s3_test})

.. code:: ipython3

    # deploy model
    
    doc2vec_model = doc2vec.create_model(
                            serializer=json_serializer,
                            deserializer=json_deserializer,
                            content_type='application/json')
    
    predictor = doc2vec_model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Apply learned embeddings to document retrieval task
---------------------------------------------------

After training the model, we can use the encoders in Object2Vec to map
new articles and sentences into a shared embedding space. Then we
evaluate the quality of these embeddings with a downstream document
retrieval task.

In the retrieval task, given a sentence query, the trained algorithm
needs to find its best matching document (the ground-truth document is
the one that contains it) from a pool of documents, where the pool
contains 10,000 other non ground-truth documents.

.. code:: ipython3

    def generate_tokenized_articles_from_single_file(fname, word_dict):
        for article in get_article_iter_from_file(fname):
            integer_article = []
            for sent in readlines_from_article(article):
                integer_article += sentence_to_integers(sent, word_dict)
            yield integer_article

.. code:: ipython3

    def read_jsonline(fname):
        """
        Reads jsonline files and returns iterator
        """
        with jsonlines.open(fname) as reader:
            for line in reader:
                yield line
    
    def send_payload(predictor, payload):
        return predictor.predict(payload)
    
    def write_to_jsonlines(data, fname):
        with jsonlines.open(fname, 'a') as writer:
            data = data['predictions']
            writer.write_all(data)
    
    
    def eval_and_write(predictor, fname, to_fname,  batch_size):
        if os.path.exists(to_fname):
            print("Removing exisiting embedding file {}".format(to_fname))
            os.remove(to_fname)
        print("Getting embedding of data in {} and store to {}...".format(fname, to_fname))
        test_data_content = list(read_jsonline(fname))
        n_test = len(test_data_content)
        n_batches = math.ceil(n_test / float(batch_size))
        start = 0
        for idx in range(n_batches):
            if idx % 10 == 0:
                print("Inference on the {}-th batch".format(idx+1))
            end = (start + batch_size) if (start + batch_size) <= n_test else n_test
            payload = {'instances': test_data_content[start:end]}
            data = send_payload(predictor, payload)
            write_to_jsonlines(data, to_fname)
            start = end
    
    def get_embeddings(predictor, test_data_content, batch_size):
        n_test = len(test_data_content)
        n_batches = math.ceil(n_test / float(batch_size))
        start = 0
        embeddings = []
        for idx in range(n_batches):
            if idx % 10 == 0:
                print("Inference the {}-th batch".format(idx+1))
            end = (start + batch_size) if (start + batch_size) <= n_test else n_test
            payload = {'instances': test_data_content[start:end]}
            data = send_payload(predictor, payload)
            embeddings += data['predictions']
            start = end
        return embeddings

.. code:: ipython3

    basedocs_fpath = os.path.join(datadir, 'wikipedia_test_basedocs.txt')
    test_fpath = '{}_tokenized-nsr{}.jsonl'.format('test10k', test_nsr)
    eval_basedocs = 'test_basedocs_tokenized_in0.jsonl'
    basedocs_emb = 'test_basedocs_embeddings.jsonl'
    sent_doc_emb = 'test10k_embeddings_pairs.jsonl'

.. code:: ipython3

    import jsonlines
    import numpy as np
    basedocs_emb = 'test_basedocs_embeddings.jsonl'
    sent_doc_emb = 'test10k_embeddings_pairs.jsonl'

.. code:: ipython3

    batch_size = 100
    
    # tokenize basedocs
    with jsonlines.open(eval_basedocs, 'w') as writer:
        for data in generate_tokenized_articles_from_single_file(basedocs_fpath, w_dict):
            writer.write({'in0': data})
    
    # get basedocs embedding
    eval_and_write(predictor, eval_basedocs, basedocs_emb, batch_size)
    
    
    # get embeddings for sentence and ground-truth article pairs
    sentences = []
    gt_articles = []
    for data in read_jsonline(test_fpath):
        if data['label'] == 1:
            sentences.append({'in0': data['in0']})
            gt_articles.append({'in0': data['in1']})
            
    sent_emb = get_embeddings(predictor, sentences, batch_size)
    doc_emb = get_embeddings(predictor, gt_articles, batch_size)
    
    with jsonlines.open(sent_doc_emb, 'w') as writer:
        for (sent, doc) in zip(sent_emb, doc_emb):
            writer.write({'sent': sent['embeddings'], 'doc': doc['embeddings']})

.. code:: ipython3

    del w_dict
    del sent_emb, doc_emb

The blocks below evaluate the performance of Object2Vec model on the
document retrieval task.

We use two metrics hits@k and mean rank to evaluate the retrieval
performance. Note that the ground-truth documents in the pool have the
query sentence removed from them – else the task would have been
trivial.

-  hits@k: It calculates the fraction of queries where its best-matching
   (ground-truth) document is contained in top k retrieved documents by
   the algorithm.
-  mean rank: It is the average rank of the best-matching documents, as
   determined by the algorithm, over all queries.

.. code:: ipython3

    # Construct normalized basedocs, sentences, and ground-truth docs embedding matrix
    
    basedocs = []
    with jsonlines.open(basedocs_emb) as reader:
        for line in reader:
            basedocs.append(np.array(line['embeddings'])) 
    
    
    sent_embs = []
    gt_doc_embs = []
    
    with jsonlines.open(sent_doc_emb) as reader2:
        for line2 in reader2:
            sent_embs.append(line2['sent'])
            gt_doc_embs.append(line2['doc'])
    
    basedocs_emb_mat = normalize(np.array(basedocs).T, axis=0)
    sent_emb_mat = normalize(np.array(sent_embs), axis=1)
    gt_emb_mat = normalize(np.array(gt_doc_embs).T, axis=0)

.. code:: ipython3

    def get_chunk_query_rank(sent_emb_mat, basedocs_emb_mat, gt_emb_mat, largest_k):
        # this is a memory-consuming step if chunk is large
        dot_with_basedocs = np.matmul(sent_emb_mat, basedocs_emb_mat)
        dot_with_gt = np.diag(np.matmul(sent_emb_mat, gt_emb_mat))
        final_ranking_scores = np.insert(dot_with_basedocs, 0, dot_with_gt, axis=1)
        query_rankings = list()
        largest_k_list = list()
        for row in final_ranking_scores:
            ranking_ind = np.argsort(row) # sorts row in increasing order of similarity score
            num_scores = len(ranking_ind)
            query_rankings.append(num_scores-list(ranking_ind).index(0))
            largest_k_list.append(np.array(ranking_ind[-largest_k:]).astype(int))
        return query_rankings, largest_k_list
        

``Note: We evaluate the learned embeddings on chunks of test sentences-document pairs to save run-time memory; this is to make sure that our code works on the smallest notebook instance *ml.t2.medium*. If you have a larger notebook instance, you can increase the chunk_size to speed up evaluation. For instances larger than ml.t2.xlarge, you can set chunk_size = num_test_samples``

.. code:: ipython3

    chunk_size = 1000
    num_test_samples = len(sent_embs)
    assert num_test_samples%chunk_size == 0, "Chunk_size must be divisible by {}".format(num_test_samples)
    num_chunks = int(num_test_samples / chunk_size)
    k_list = [1, 5, 10, 20, 50]
    largest_k = max(k_list)
    query_all_rankings = list()
    all_largest_k_list = list()
    
    for i in range(0, num_chunks*chunk_size, chunk_size):
        print("Evaluating on the {}-th chunk".format(i))
        j = i+chunk_size
        sent_emb_submat = sent_emb_mat[i:j, :]
        gt_emb_submat = gt_emb_mat[:, i:j]
        query_rankings, largest_k_list = get_chunk_query_rank(sent_emb_submat, basedocs_emb_mat, gt_emb_submat, largest_k)
        query_all_rankings += query_rankings
        all_largest_k_list.append(np.array(largest_k_list).astype(int))
    
    all_largest_k_mat = np.concatenate(all_largest_k_list, axis=0).astype(int)
    
    print("Summary:")
    print("Mean query ranks is {}".format(np.mean(query_all_rankings)))
    print("Percentiles of query ranks is 50%:{}, 80%:{}, 90%:{}, 99%:{}".format(*np.percentile(query_all_rankings, [50, 80, 90, 99])))
    
    for k in k_list:
        top_k_mat = all_largest_k_mat[:, -k:]
        unique, counts = np.unique(top_k_mat, return_counts=True)
        print("The hits at {} score is {}/{}".format(k, counts[0], len(top_k_mat)))

Comparison with the StarSpace algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We compare the performance of Object2Vec with the StarSpace
(https://github.com/facebookresearch/StarSpace) algorithm on the
document retrieval evaluation task, using a set of 250 thousand
Wikipedia documents. The experimental results displayed in the table
below, show that Object2Vec significantly outperforms StarSpace on all
metrics although both models use the same kind of encoders for sentences
and documents.

+------------+--------+---------+---------+-----------+
| Algorithm  | hits@1 | hits@10 | hits@20 | mean rank |
+============+========+=========+=========+===========+
| StarSpace  | 21.98% | 42.77%  | 50.55%  | 303.34    |
+------------+--------+---------+---------+-----------+
| Object2Vec | 26.40% | 47.42%  | 53.83%  | 248.67    |
+------------+--------+---------+---------+-----------+

.. code:: ipython3

    predictor.delete_endpoint()
