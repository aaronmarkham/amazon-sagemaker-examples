An Introduction to SageMaker ObjectToVec model for MovieLens recommendation
===========================================================================

1. `Background <#Background>`__
2. `Data exploration and
   preparation <#Data-exploration-and-preparation>`__
3. `Rating prediction task <#Rating-prediction-task>`__
4. `Recommendation task <#Recommendation-task>`__
5. `Movie retrieval in the embedding
   space <#Movie-retrieval-in-the-embedding-space>`__

Background
==========

ObjectToVec
~~~~~~~~~~~

*Object2Vec* is a highly customizable multi-purpose algorithm that can
learn embeddings of pairs of objects. The embeddings are learned such
that it preserves their pairwise **similarities** in the original space.
- **Similarity** is user-defined: users need to provide the algorithm
with pairs of objects that they define as similar (1) or dissimilar (0);
alternatively, the users can define similarity in a continuous sense
(provide a real-valued similarity score) - The learned embeddings can be
used to efficiently compute nearest neighbors of objects, as well as to
visualize natural clusters of related objects in the embedding space. In
addition, the embeddings can also be used as features of the
corresponding objects in downstream supervised tasks such as
classification or regression

In this notebook example:
-------------------------

We demonstrate how Object2Vec can be used to solve problems arising in
recommendation systems. Specifically,

-  We provide the algorithm with (UserID, MovieID) pairs; for each such
   pair, we also provide a “label” that tells the algorithm whether the
   user and movie are similar or not

   -  When the labels are real-valued, we use the algorithm to predict
      the exact ratings of a movie given a user
   -  When the labels are binary, we use the algorithm to recommendation
      movies to users

-  The diagram below shows the customization of our model to the problem
   of predicting movie ratings, using a dataset that provides
   ``(UserID, ItemID, Rating)`` samples. Here, ratings are real-valued



Dataset
~~~~~~~

-  We use the MovieLens 100k dataset:
   https://grouplens.org/datasets/movielens/100k/

Use cases
~~~~~~~~~

-  Task 1: Rating prediction (regression)
-  Task 2: Movie recommendation (classification)
-  Task 3: Nearest-neighbor movie retrieval in the learned embedding
   space

Before running the notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Please use a Python 3 kernel for the notebook
-  Please make sure you have ``jsonlines`` package installed (if not,
   you can run the command below to install it)

.. code:: ipython3

    !pip install jsonlines

.. code:: ipython3

    import os
    import sys
    import csv, jsonlines
    import numpy as np
    import copy
    import random

.. code:: ipython3

    %matplotlib inline
    import matplotlib.pyplot as plt

Data exploration and preparation
================================

License
~~~~~~~

Please be aware of the following requirements about ackonwledgment,
copyright and availability, cited from the `data set description
page <http://files.grouplens.org/datasets/movielens/ml-100k-README.txt>`__.
>The data set may be used for any research purposes under the following
conditions: \* The user may not state or imply any endorsement from the
University of Minnesota or the GroupLens Research Group. \* The user
must acknowledge the use of the data set in publications resulting from
the use of the data set (see below for citation information). \* The
user may not redistribute the data without separate permission. \* The
user may not use this information for any commercial or revenue-bearing
purposes without first obtaining permission from a faculty member of the
GroupLens Research Project at the University of Minnesota. If you have
any further questions or comments, please contact GroupLens
<grouplens-info@cs.umn.edu>.

.. code:: bash

    %%bash
    
    curl -o ml-100k.zip http://files.grouplens.org/datasets/movielens/ml-100k.zip
    unzip ml-100k.zip
    rm ml-100k.zip

Let’s first create some utility functions for data exploration and
preprocessing

.. code:: ipython3

    ## some utility functions
    
    def load_csv_data(filename, delimiter, verbose=True):
        """
        input: a file readable as csv and separated by a delimiter
        and has format users - movies - ratings - etc
        output: a list, where each row of the list is of the form
        {'in0':userID, 'in1':movieID, 'label':rating}
        """
        to_data_list = list()
        users = list()
        movies = list()
        ratings = list()
        unique_users = set()
        unique_movies = set()
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for count, row in enumerate(reader):
                #if count!=0:
                to_data_list.append({'in0':[int(row[0])], 'in1':[int(row[1])], 'label':float(row[2])})
                users.append(row[0])
                movies.append(row[1])
                ratings.append(float(row[2]))
                unique_users.add(row[0])
                unique_movies.add(row[1])
        if verbose:
            print("In file {}, there are {} ratings".format(filename, len(ratings)))
            print("The ratings have mean: {}, median: {}, and variance: {}".format(
                                                round(np.mean(ratings), 2), 
                                                round(np.median(ratings), 2), 
                                                round(np.var(ratings), 2)))
            print("There are {} unique users and {} unique movies".format(len(unique_users), len(unique_movies)))
        return to_data_list
    
    
    def csv_to_augmented_data_dict(filename, delimiter):
        """
        Input: a file that must be readable as csv and separated by delimiter (to make columns)
        has format users - movies - ratings - etc
        Output:
          Users dictionary: keys as user ID's; each key corresponds to a list of movie ratings by that user
          Movies dictionary: keys as movie ID's; each key corresponds a list of ratings of that movie by different users
        """
        to_users_dict = dict() 
        to_movies_dict = dict()
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for count, row in enumerate(reader):
                #if count!=0:
                if row[0] not in to_users_dict:
                    to_users_dict[row[0]] = [(row[1], row[2])]
                else:
                    to_users_dict[row[0]].append((row[1], row[2]))
                if row[1] not in to_movies_dict:
                    to_movies_dict[row[1]] = list(row[0])
                else:
                    to_movies_dict[row[1]].append(row[0])
        return to_users_dict, to_movies_dict
    
    
    def user_dict_to_data_list(user_dict):
        # turn user_dict format to data list format (acceptable to the algorithm)
        data_list = list()
        for user, movie_rating_list in user_dict.items():
            for movie, rating in movie_rating_list:
                data_list.append({'in0':[int(user)], 'in1':[int(movie)], 'label':float(rating)})
        return data_list
    
    def divide_user_dicts(user_dict, sp_ratio_dict):
        """
        Input: A user dictionary, a ration dictionary
             - format of sp_ratio_dict = {'train':0.8, "test":0.2}
        Output: 
            A dictionary of dictionaries, with key corresponding to key provided by sp_ratio_dict
            and each key corresponds to a subdivded user dictionary
        """
        ratios = [val for _, val in sp_ratio_dict.items()]
        assert np.sum(ratios) == 1, "the sampling ratios must sum to 1!"
        divided_dict = {}
        for user, movie_rating_list in user_dict.items():
            sub_movies_ptr = 0
            sub_movies_list = []
            #movie_list, _ = zip(*movie_rating_list)
            #print(movie_list)
            for i, ratio in enumerate(ratios):
                if i < len(ratios)-1:
                    sub_movies_ptr_end = sub_movies_ptr + int(len(movie_rating_list)*ratio)
                    sub_movies_list.append(movie_rating_list[sub_movies_ptr:sub_movies_ptr_end])
                    sub_movies_ptr = sub_movies_ptr_end
                else:
                    sub_movies_list.append(movie_rating_list[sub_movies_ptr:])
            for subset_name in sp_ratio_dict.keys():
                if subset_name not in divided_dict:
                    divided_dict[subset_name] = {user: sub_movies_list.pop(0)}
                else:
                    #access sub-dictionary
                    divided_dict[subset_name][user] = sub_movies_list.pop(0)
        
        return divided_dict
    
    def write_csv_to_jsonl(jsonl_fname, csv_fname, csv_delimiter):
        """
        Input: a file readable as csv and separated by delimiter (to make columns)
            - has format users - movies - ratings - etc
        Output: a jsonline file converted from the csv file
        """
        with jsonlines.open(jsonl_fname, mode='w') as writer:
            with open(csv_fname, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=csv_delimiter)
                for count, row in enumerate(reader):
                    #print(row)
                    #if count!=0:
                    writer.write({'in0':[int(row[0])], 'in1':[int(row[1])], 'label':float(row[2])})
            print('Created {} jsonline file'.format(jsonl_fname))
                        
        
    def write_data_list_to_jsonl(data_list, to_fname):
        """
        Input: a data list, where each row of the list is a Python dictionary taking form
        {'in0':userID, 'in1':movieID, 'label':rating}
        Output: save the list as a jsonline file
        """
        with jsonlines.open(to_fname, mode='w') as writer:
            for row in data_list:
                #print(row)
                writer.write({'in0':row['in0'], 'in1':row['in1'], 'label':row['label']})
        print("Created {} jsonline file".format(to_fname))
    
    def data_list_to_inference_format(data_list, binarize=True, label_thres=3):
        """
        Input: a data list
        Output: test data and label, acceptable by SageMaker for inference
        """
        data_ = [({"in0":row['in0'], 'in1':row['in1']}, row['label']) for row in data_list]
        data, label = zip(*data_)
        infer_data = {"instances":data}
        if binarize:
            label = get_binarized_label(list(label), label_thres)
        return infer_data, label
    
    
    def get_binarized_label(data_list, thres):
        """
        Input: data list
        Output: a binarized data list for recommendation task
        """
        for i, row in enumerate(data_list):
            if type(row) is dict:
                #if i < 10:
                    #print(row['label'])
                if row['label'] > thres:
                    #print(row)
                    data_list[i]['label'] = 1
                else:
                    data_list[i]['label'] = 0
            else:
                if row > thres:
                    data_list[i] = 1
                else:
                    data_list[i] = 0
        return data_list


.. code:: ipython3

    ## Load data and shuffle
    prefix = 'ml-100k'
    train_path = os.path.join(prefix, 'ua.base')
    valid_path = os.path.join(prefix, 'ua.test')
    test_path = os.path.join(prefix, 'ub.test')
    
    train_data_list = load_csv_data(train_path, '\t')
    random.shuffle(train_data_list)
    validation_data_list = load_csv_data(valid_path, '\t')
    random.shuffle(validation_data_list)

.. code:: ipython3

    to_users_dict, to_movies_dict = csv_to_augmented_data_dict(train_path, '\t')

We perform some data exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ## Calculate min, max, median of number of movies per user
    movies_per_user = [len(val) for key, val in to_users_dict.items()]
    
    print("The min, max, and median 'movies per user' is {}, {}, and {}".format(np.amin(movies_per_user),
                                                                             np.amax(movies_per_user),
                                                                             np.median(movies_per_user)))
    users_per_movie = [len(val) for key, val in to_movies_dict.items()]
    print("The min, max, and median 'users per movie' is {}, {}, and {}".format(np.amin(users_per_movie),
                                                                             np.amax(users_per_movie),
                                                                              np.median(users_per_movie)))
    
    
    count = 0
    n_movies_lower_bound = 20
    for n_movies in movies_per_user:
        if n_movies <= n_movies_lower_bound:
            count += 1
    print("In the training set")
    print('There are {} users with no more than {} movies'.format(count, n_movies_lower_bound))
    #
    count = 0
    n_users_lower_bound = 2
    for n_users in users_per_movie:
        if n_users <= n_users_lower_bound:
            count += 1
    print('There are {} movies with no more than {} user'.format(count, n_users_lower_bound))
    
    
    ## figures
    
    f = plt.figure(1)
    plt.hist(movies_per_user)
    plt.title("Movies per user")
    ##
    g = plt.figure(2)
    plt.hist(users_per_movie)
    plt.title("Users per movie")

Since the number of movies with an extremely small number of users (<3)
is negligible compared to the total number of movies, we will not remove
movies from the data set (same applies for users)

.. code:: ipython3

    ## Save training and validation data locally for rating-prediction (regression) task
    
    write_data_list_to_jsonl(copy.deepcopy(train_data_list), 'train_r.jsonl')
    write_data_list_to_jsonl(copy.deepcopy(validation_data_list), 'validation_r.jsonl')

.. code:: ipython3

    ## Save training and validation data locally for recommendation (classification) task
    
    ### binarize the data 
    
    train_c = get_binarized_label(copy.deepcopy(train_data_list), 3.0)
    valid_c = get_binarized_label(copy.deepcopy(validation_data_list), 3.0)
    
    write_data_list_to_jsonl(train_c, 'train_c.jsonl')
    write_data_list_to_jsonl(valid_c, 'validation_c.jsonl')

**We check whether the two classes are balanced after binarization**

.. code:: ipython3

    train_c_label = [row['label'] for row in train_c]
    valid_c_label = [row['label'] for row in valid_c]
    
    print("There are {} fraction of positive ratings in train_c.jsonl".format(
                                    np.count_nonzero(train_c_label)/len(train_c_label)))
    print("There are {} fraction of positive ratings in validation_c.jsonl".format(
                                    np.sum(valid_c_label)/len(valid_c_label)))

Rating prediction task
======================

.. code:: ipython3

    def get_mse_loss(res, labels):
        if type(res) is dict:
            res = res['predictions']
        assert len(res)==len(labels), 'result and label length mismatch!'
        loss = 0
        for row, label in zip(res, labels):
            if type(row)is dict:
                loss += (row['scores'][0]-label)**2
            else:
                loss += (row-label)**2
        return round(loss/float(len(labels)), 2)

.. code:: ipython3

    valid_r_data, valid_r_label = data_list_to_inference_format(copy.deepcopy(validation_data_list), binarize=False)

We first test the problem on two baseline algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Baseline 1
----------

A naive approach to predict movie ratings on unseen data is to use the
global average of the user predictions in the training data

.. code:: ipython3

    train_r_label = [row['label'] for row in copy.deepcopy(train_data_list)]
    
    bs1_prediction = round(np.mean(train_r_label), 2)
    print('The Baseline 1 (global rating average) prediction is {}'.format(bs1_prediction))
    print("The validation mse loss of the Baseline 1 is {}".format(
                                         get_mse_loss(len(valid_r_label)*[bs1_prediction], valid_r_label)))

Baseline 2
----------

Now we use a better baseline, which is to perform prediction on unseen
data based on the user-averaged ratings of movies on training data

.. code:: ipython3

    def bs2_predictor(test_data, user_dict, is_classification=False, thres=3):
        test_data = copy.deepcopy(test_data['instances'])
        predictions = list()
        for row in test_data:
            userID = str(row["in0"][0])
            # predict movie ID based on local average of user's prediction
            local_movies, local_ratings = zip(*user_dict[userID])
            local_ratings = [float(score) for score in local_ratings]
            predictions.append(np.mean(local_ratings))
            if is_classification:
                predictions[-1] = int(predictions[-1] > 3)
        return predictions

.. code:: ipython3

    bs2_prediction = bs2_predictor(valid_r_data, to_users_dict, is_classification=False)
    print("The validation loss of the Baseline 2 (user-based rating average) is {}".format(
                                         get_mse_loss(bs2_prediction, valid_r_label)))

Next, we will use *Object2Vec* to predict the movie ratings

Model training and inference
----------------------------

Define S3 bucket that hosts data and model, and upload data to S3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    import boto3 
    import os
     
    bucket = '<Your bucket name>' # Customize your own bucket name
    input_prefix = 'object2vec/movielens/input'
    output_prefix = 'object2vec/movielens/output'

Upload data to S3 and make data paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from sagemaker.session import s3_input
    
    s3_client = boto3.client('s3')
    input_paths = {}
    output_path = os.path.join('s3://', bucket, output_prefix)
    
    for data_name in ['train', 'validation']:
        pre_key = os.path.join(input_prefix, 'rating', f'{data_name}')
        fname = '{}_r.jsonl'.format(data_name)
        data_path = os.path.join('s3://', bucket, pre_key, fname)
        s3_client.upload_file(fname, bucket, os.path.join(pre_key, fname))
        input_paths[data_name] = s3_input(data_path, distribution='ShardedByS3Key', content_type='application/jsonlines')
        print('Uploaded {} data to {} and defined input path'.format(data_name, data_path))
    
    print('Trained model will be saved at', output_path)

Get ObjectToVec algorithm image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    
    sess = sagemaker.Session()
    
    role = get_execution_role()
    print(role)
    
    ## Get docker image of ObjectToVec algorithm
    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'object2vec')

Training
~~~~~~~~

We first define training hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    hyperparameters = {
        "_kvstore": "device",
        "_num_gpus": "auto",
        "_num_kv_servers": "auto",
        "bucket_width": 0,
        "early_stopping_patience": 3,
        "early_stopping_tolerance": 0.01,
        "enc0_cnn_filter_width": 3,
        "enc0_layers": "auto",
        "enc0_max_seq_len": 1,
        "enc0_network": "pooled_embedding",
        "enc0_token_embedding_dim": 300,
        "enc0_vocab_size": 944,
        "enc1_layers": "auto",
        "enc1_max_seq_len": 1,
        "enc1_network": "pooled_embedding",
        "enc1_token_embedding_dim": 300,
        "enc1_vocab_size": 1684,
        "enc_dim": 1024,
        "epochs": 20,
        "learning_rate": 0.001,
        "mini_batch_size": 64,
        "mlp_activation": "tanh",
        "mlp_dim": 256,
        "mlp_layers": 1,
        "num_classes": 2,
        "optimizer": "adam",
        "output_layer": "mean_squared_error"
    }

.. code:: ipython3

    ## get estimator
    regressor = sagemaker.estimator.Estimator(container,
                                              role, 
                                              train_instance_count=1, 
                                              train_instance_type='ml.p2.xlarge',
                                              output_path=output_path,
                                              sagemaker_session=sess)
    
    ## set hyperparameters
    regressor.set_hyperparameters(**hyperparameters)
    
    ## train the model
    regressor.fit(input_paths)

We have seen that we can upload train (validation) data through the
input data channel, and the algorithm will print out train (validation)
evaluation metric during training. In addition, the algorithm uses the
validation metric to perform early stopping.

What if we want to send additional unlabeled data to the algorithm and
get predictions from the trained model? This step is called *inference*
in the Sagemaker framework. Next, we demonstrate how to use a trained
model to perform inference on unseen data points.

Inference using trained model
-----------------------------

Create and deploy the model

.. code:: ipython3

    #import numpy as np
    from sagemaker.predictor import json_serializer, json_deserializer
    
    # create a model using the trained algorithm
    regression_model = regressor.create_model(
                            serializer=json_serializer,
                            deserializer=json_deserializer,
                            content_type='application/json')

.. code:: ipython3

    # deploy the model
    predictor = regression_model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Below we send validation data (without labels) to the deployed endpoint
for inference. We will see that the resulting prediction error we get
from post-training inference matches the best validation error from the
training log in the console above (up to floating point error). If you
follow the training instruction and parameter setup, you should get mean
squared error on the validation set approximately 0.91.

.. code:: ipython3

    # Send data to the endpoint to get predictions
    prediction = predictor.predict(valid_r_data)
    
    print("The mean squared error on validation set is %.3f" %get_mse_loss(prediction, valid_r_label))

Comparison against popular libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below we provide a chart that compares the performance of *Object2Vec*
against several algorithms implemented by popular recommendation system
libraries (LibRec https://www.librec.net/ and scikit-surprise
http://surpriselib.com/). The error metric we use in the chart is **root
mean squared** (RMSE) instead of MSE, so that our result can be compared
against the reported results in the aforementioned libraries.



Recommendation task
===================

In this section, we showcase how to use *Object2Vec* to recommend
movies, using the binarized rating labels. Here, if a movie rating label
for a given user is binarized to ``1``, then it means that the movie
should be recommended to the user; otherwise, the label is binarized to
``0``. The binarized data set is already obtained in the preprocessing
section, so we will proceed to apply the algorithm.

We upload the binarized datasets for classification task to S3

.. code:: ipython3

    for data_name in ['train', 'validation']:
        fname = '{}_c.jsonl'.format(data_name)
        pre_key = os.path.join(input_prefix, 'recommendation', f"{data_name}")
        data_path = os.path.join('s3://', bucket, pre_key, fname)
        s3_client.upload_file(fname, bucket, os.path.join(pre_key, fname))
        input_paths[data_name] = s3_input(data_path, distribution='ShardedByS3Key', content_type='application/jsonlines')
        print('Uploaded data to {}'.format(data_path))

Since we already get the algorithm image from the regression task, we
can directly start training

.. code:: ipython3

    from sagemaker.session import s3_input
    
    hyperparameters_c = {
        "_kvstore": "device",
        "_num_gpus": "auto",
        "_num_kv_servers": "auto",
        "bucket_width": 0,
        "early_stopping_patience": 3, 
        "early_stopping_tolerance": 0.01,
        "enc0_cnn_filter_width": 3,
        "enc0_layers": "auto",
        "enc0_max_seq_len": 1,
        "enc0_network": "pooled_embedding",
        "enc0_token_embedding_dim": 300,
        "enc0_vocab_size": 944,
        "enc1_cnn_filter_width": 3,
        "enc1_layers": "auto",
        "enc1_max_seq_len": 1,
        "enc1_network": "pooled_embedding",
        "enc1_token_embedding_dim": 300,
        "enc1_vocab_size": 1684,
        "enc_dim": 2048,
        "epochs": 20,
        "learning_rate": 0.001,
        "mini_batch_size": 2048,
        "mlp_activation": "relu",
        "mlp_dim": 1024,
        "mlp_layers": 1,
        "num_classes": 2,
        "optimizer": "adam",
        "output_layer": "softmax"
    }

.. code:: ipython3

    ## get estimator
    classifier = sagemaker.estimator.Estimator(container,
                                        role, 
                                        train_instance_count=1, 
                                        train_instance_type='ml.p2.xlarge',
                                        output_path=output_path,
                                        sagemaker_session=sess)
    
    ## set hyperparameters
    classifier.set_hyperparameters(**hyperparameters_c)
    
    ## train, tune, and test the model
    classifier.fit(input_paths)

Again, we can create, deploy, and validate the model after training

.. code:: ipython3

    classification_model = classifier.create_model(
                            serializer=json_serializer,
                            deserializer=json_deserializer,
                            content_type='application/json')
    
    predictor_2 = classification_model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

.. code:: ipython3

    valid_c_data, valid_c_label = data_list_to_inference_format(copy.deepcopy(validation_data_list), 
                                                                label_thres=3, binarize=True)
    predictions = predictor_2.predict(valid_c_data)

.. code:: ipython3

    def get_class_accuracy(res, labels, thres):
        if type(res) is dict:
            res = res['predictions']
        assert len(res)==len(labels), 'result and label length mismatch!'
        accuracy = 0
        for row, label in zip(res, labels):
            if type(row) is dict:
                if row['scores'][1] > thres:
                    prediction = 1
                else: 
                    prediction = 0
                if label > thres:
                    label = 1
                else:
                    label = 0
                accuracy += 1 - (prediction - label)**2
        return accuracy / float(len(res))
    
    print("The accuracy on the binarized validation set is %.3f" %get_class_accuracy(predictions, valid_c_label, 0.5))

The accuracy on validation set you would get should be approximately
0.704.

Movie retrieval in the embedding space
--------------------------------------

Since *Object2Vec* transforms user and movie ID’s into embeddings as
part of the training process. After training, it obtains user and movie
embeddings in the left and right encoders, respectively. Intuitively,
the embeddings should be tuned by the algorithm in a way that
facilitates the supervised learning task: since for a specific user,
similar movies should have similar ratings, we expect that similar
movies should be **close-by** in the embedding space.

In this section, we demonstrate how to find the nearest-neighbor (in
Euclidean distance) of a given movie ID, among all movie ID’s.

.. code:: ipython3

    def get_movie_embedding_dict(movie_ids, trained_model):
        input_instances = list()
        for s_id in movie_ids:
            input_instances.append({'in1': [s_id]})
        data = {'instances': input_instances}
        movie_embeddings = trained_model.predict(data)
        embedding_dict = {}
        for s_id, row in zip(movie_ids, movie_embeddings['predictions']):
            embedding_dict[s_id] = np.array(row['embeddings'])
        return embedding_dict
    
    
    def load_movie_id_name_map(item_file):
        movieID_name_map = {}
        with open(item_file, 'r', encoding="ISO-8859-1") as f:
            for row in f.readlines():
                row = row.strip()
                split = row.split('|')
                movie_id = split[0]
                movie_name = split[1]
                sparse_tags = split[-19:]
                movieID_name_map[int(movie_id)] = movie_name 
        return movieID_name_map
    
                
    def get_nn_of_movie(movie_id, candidate_movie_ids, embedding_dict):
        movie_emb = embedding_dict[movie_id]
        min_dist = float('Inf')
        best_id = candidate_movie_ids[0]
        for idx, m_id in enumerate(candidate_movie_ids):
            candidate_emb = embedding_dict[m_id]
            curr_dist = np.linalg.norm(candidate_emb - movie_emb)
            if curr_dist < min_dist:
                best_id = m_id
                min_dist = curr_dist
        return best_id, min_dist
    
    
    def get_unique_movie_ids(data_list):
        unique_movie_ids = set()
        for row in data_list:
            unique_movie_ids.add(row['in1'][0])
        return list(unique_movie_ids)

.. code:: ipython3

    train_data_list = load_csv_data(train_path, '\t', verbose=False)
    unique_movie_ids = get_unique_movie_ids(train_data_list)
    embedding_dict = get_movie_embedding_dict(unique_movie_ids, predictor_2)
    candidate_movie_ids = unique_movie_ids.copy()

Using the script below, you can check out what is the closest movie to
any movie in the data set. Last time we ran it, the closest movie to
``Terminator, The (1984)`` in the embedding space was
``Die Hard (1988)``. Note that, the result will likely differ slightly
across different runs of the algorithm, due to randomness in
initialization of model parameters.

-  Just plug in the movie id you want to examine

   -  For example, the movie ID for Terminator is 195; you can find the
      movie name and ID pair in the ``u.item`` file

-  Note that, the result will likely differ across different runs of the
   algorithm, due to inherent randomness.

.. code:: ipython3

    movie_id_to_examine = '<movie id>' # Customize the movie ID you want to examine

.. code:: ipython3

    candidate_movie_ids.remove(movie_id_to_examine)
    best_id, min_dist = get_nn_of_movie(movie_id_to_examine, candidate_movie_ids, embedding_dict)
    movieID_name_map = load_movie_id_name_map('ml-100k/u.item')
    print('The closest movie to {} in the embedding space is {}'.format(movieID_name_map[movie_id_to_examine],
                                                                      movieID_name_map[best_id]))
    candidate_movie_ids.append(movie_id_to_examine)

It is recommended to always delete the endpoints used for hosting the
model

.. code:: ipython3

    ## clean up
    sess.delete_endpoint(predictor.endpoint)
    sess.delete_endpoint(predictor_2.endpoint)
