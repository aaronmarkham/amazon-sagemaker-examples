Implementing a Recommender System with SageMaker, MXNet, and Gluon
==================================================================

**Making Video Recommendations Using Neural Networks and Embeddings**

--------------

--------------

*This work is based on content from the*\ `Cyrus Vahid’s 2017 re:Invent
Talk <https://github.com/cyrusmvahid/gluontutorials/blob/master/recommendations/MLPMF.ipynb>`__

Contents
--------

1.  `Background <#Background>`__
2.  `Setup <#Setup>`__
3.  `Data <#Data>`__
4.  `Explore <#Explore>`__
5.  `Clean <#Clean>`__
6.  `Prepare <#Prepare>`__
7.  `Train Locally <#Train-Locally>`__
8.  `Define Network <#Define-Network>`__
9.  `Set Parameters <#Set-Parameters>`__
10. `Execute <#Execute>`__
11. `Train with SageMaker <#Train-with-SageMaker>`__
12. `Wrap Code <#Wrap-Code>`__
13. `Move Data <#Move-Data>`__
14. `Submit <#Submit>`__
15. `Host <#Host>`__
16. `Evaluate <#Evaluate>`__
17. `Wrap-up <#Wrap-up>`__

--------------

Background
----------

In many ways, recommender systems were a catalyst for the current
popularity of machine learning. One of Amazon’s earliest successes was
the “Customers who bought this, also bought…” feature, while the million
dollar Netflix Prize spurred research, raised public awareness, and
inspired numerous other data science competitions.

Recommender systems can utilize a multitude of data sources and ML
algorithms, and most combine various unsupervised, supervised, and
reinforcement learning techniques into a holistic framework. However,
the core component is almost always a model which which predicts a
user’s rating (or purchase) for a certain item based on that user’s
historical ratings of similar items as well as the behavior of other
similar users. The minimal required dataset for this is a history of
user item ratings. In our case, we’ll use 1 to 5 star ratings from over
2M Amazon customers on over 160K digital videos. More details on this
dataset can be found at its `AWS Public Datasets
page <https://s3.amazonaws.com/amazon-reviews-pds/readme.html>`__.

Matrix factorization has been the cornerstone of most user-item
prediction models. This method starts with the large, sparse, user-item
ratings in a single matrix, where users index the rows, and items index
the columns. It then seeks to find two lower-dimensional, dense matrices
which, when multiplied together, preserve the information and
relationships in the larger matrix.

.. figure:: https://developers.google.com/machine-learning/recommendation/images/Matrixfactor.svg
   :alt: image

   image

Matrix factorization has been extended and genarlized with deep learning
and embeddings. These techniques allows us to introduce non-linearities
for enhanced performance and flexibility. This notebook will fit a
neural network-based model to generate recommendations for the Amazon
video dataset. It will start by exploring our data in the notebook and
even training a model on a sample of the data. Later we’ll expand to the
full dataset and fit our model using a SageMaker managed training
cluster. We’ll then deploy to an endpoint and check our method.

--------------

Setup
-----

*This notebook was created and tested on an ml.p2.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the ``get_execution_role()`` call with the
   appropriate full IAM role arn string(s).

.. code:: ipython3

    import sagemaker
    import boto3
    
    role = sagemaker.get_execution_role()
    region = boto3.Session().region_name
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket and prefix
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-gluon-recsys'

Now let’s load the Python libraries we’ll need for the remainder of this
example notebook.

.. code:: ipython3

    import os
    import mxnet as mx
    from mxnet import gluon, nd, ndarray
    from mxnet.metric import MSE
    import pandas as pd
    import numpy as np
    from sagemaker.mxnet import MXNet
    import json
    import matplotlib.pyplot as plt

--------------

Data
----

Explore
~~~~~~~

Let’s start by bringing in our dataset from an S3 public bucket. As
mentioned above, this contains 1 to 5 star ratings from over 2M Amazon
customers on over 160K digital videos. More details on this dataset can
be found at its `AWS Public Datasets
page <https://s3.amazonaws.com/amazon-reviews-pds/readme.html>`__.

*Note, because this dataset is over a half gigabyte, the load from S3
may take ~10 minutes. Also, since Amazon SageMaker Notebooks start with
a 5GB persistent volume by default, and we don’t need to keep this data
on our instance for long, we’ll bring it to the temporary volume (which
has up to 20GB of storage).*

.. code:: ipython3

    !mkdir /tmp/recsys/
    !aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz /tmp/recsys/

Let’s read the data into a `Pandas
DataFrame <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`__
so that we can begin to understand it.

*Note, we’ll set ``error_bad_lines=False`` when reading the file in as
there appear to be a very small number of records which would create a
problem otherwise.*

.. code:: ipython3

    df = pd.read_csv('/tmp/recsys/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz', delimiter='\t',error_bad_lines=False)
    df.head()

We can see this dataset includes information like:

-  ``marketplace``: 2-letter country code (in this case all “US”).
-  ``customer_id``: Random identifier that can be used to aggregate
   reviews written by a single author.
-  ``review_id``: A unique ID for the review.
-  ``product_id``: The Amazon Standard Identification Number (ASIN).
   ``http://www.amazon.com/dp/<ASIN>`` links to the product’s detail
   page.
-  ``product_parent``: The parent of that ASIN. Multiple ASINs (color or
   format variations of the same product) can roll up into a single
   parent parent.
-  ``product_title``: Title description of the product.
-  ``product_category``: Broad product category that can be used to
   group reviews (in this case digital videos).
-  ``star_rating``: The review’s rating (1 to 5 stars).
-  ``helpful_votes``: Number of helpful votes for the review.
-  ``total_votes``: Number of total votes the review received.
-  ``vine``: Was the review written as part of the
   `Vine <https://www.amazon.com/gp/vine/help>`__ program?
-  ``verified_purchase``: Was the review from a verified purchase?
-  ``review_headline``: The title of the review itself.
-  ``review_body``: The text of the review.
-  ``review_date``: The date the review was written.

For this example, let’s limit ourselves to ``customer_id``,
``product_id``, and ``star_rating``. Including additional features in
our recommendation system could be beneficial, but would require
substantial processing (particularly the text data) which would take us
beyond the scope of this notebook.

*Note: we’ll keep ``product_title`` on the dataset to help verify our
recommendations later in the notebook, but it will not be used in
algorithm training.*

.. code:: ipython3

    df = df[['customer_id', 'product_id', 'star_rating', 'product_title']]

Because most people haven’t seen most videos, and people rate fewer
videos than we actually watch, we’d expect our data to be sparse. Our
algorithm should work well with this sparse problem in general, but we
may still want to clean out some of the long tail. Let’s look at some
basic percentiles to confirm.

.. code:: ipython3

    customers = df['customer_id'].value_counts()
    products = df['product_id'].value_counts()
    
    quantiles = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
    print('customers\n', customers.quantile(quantiles))
    print('products\n', products.quantile(quantiles))

As we can see, only about 5% of customers have rated 5 or more videos,
and only 25% of videos have been rated by 9+ customers.

Clean
~~~~~

Let’s filter out this long tail.

.. code:: ipython3

    customers = customers[customers >= 5]
    products = products[products >= 10]
    
    reduced_df = df.merge(pd.DataFrame({'customer_id': customers.index})).merge(pd.DataFrame({'product_id': products.index}))

Now, we’ll recreate our customer and product lists since there are
customers with more than 5 reviews, but all of their reviews are on
products with less than 5 reviews (and vice versa).

.. code:: ipython3

    customers = reduced_df['customer_id'].value_counts()
    products = reduced_df['product_id'].value_counts()

Next, we’ll number each user and item, giving them their own sequential
index. This will allow us to hold the information in a sparse format
where the sequential indices indicate the row and column in our ratings
matrix.

.. code:: ipython3

    customer_index = pd.DataFrame({'customer_id': customers.index, 'user': np.arange(customers.shape[0])})
    product_index = pd.DataFrame({'product_id': products.index, 
                                  'item': np.arange(products.shape[0])})
    
    reduced_df = reduced_df.merge(customer_index).merge(product_index)
    reduced_df.head()

Prepare
~~~~~~~

Let’s start by splitting in training and test sets. This will allow us
to estimate the model’s accuracy on videos our customers rated, but
wasn’t included in our training.

.. code:: ipython3

    test_df = reduced_df.groupby('customer_id').last().reset_index()
    
    train_df = reduced_df.merge(test_df[['customer_id', 'product_id']], 
                                on=['customer_id', 'product_id'], 
                                how='outer', 
                                indicator=True)
    train_df = train_df[(train_df['_merge'] == 'left_only')]

Now, we can convert our Pandas DataFrames into MXNet NDArrays, use those
to create a member of the SparseMatrixDataset class, and add that to an
MXNet Data Iterator. This process is the same for both test and control.

.. code:: ipython3

    batch_size = 1024
    
    train = gluon.data.ArrayDataset(nd.array(train_df['user'].values, dtype=np.float32),
                                    nd.array(train_df['item'].values, dtype=np.float32),
                                    nd.array(train_df['star_rating'].values, dtype=np.float32))
    test  = gluon.data.ArrayDataset(nd.array(test_df['user'].values, dtype=np.float32),
                                    nd.array(test_df['item'].values, dtype=np.float32),
                                    nd.array(test_df['star_rating'].values, dtype=np.float32))
    
    train_iter = gluon.data.DataLoader(train, shuffle=True, num_workers=4, batch_size=batch_size, last_batch='rollover')
    test_iter = gluon.data.DataLoader(train, shuffle=True, num_workers=4, batch_size=batch_size, last_batch='rollover')

--------------

Train Locally
-------------

Define Network
~~~~~~~~~~~~~~

Let’s start by defining the neural network version of our matrix
factorization task. In this case, our network is quite simple. The main
components are: -
`Embeddings <https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Embedding>`__
which turn our indexes into dense vectors of fixed size. In this case,
64. - `Dense
layers <https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.nn.Dense>`__
with ReLU activation. Each dense layer has the same number of units as
our number of embeddings. Our ReLU activation here also adds some
non-linearity to our matrix factorization. - `Dropout
layers <https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.nn.Dropout>`__
which can be used to prevent over-fitting. - Matrix multiplication of
our user matrix and our item matrix to create an estimate of our rating
matrix.

.. code:: ipython3

    class MFBlock(gluon.HybridBlock):
        def __init__(self, max_users, max_items, num_emb, dropout_p=0.5):
            super(MFBlock, self).__init__()
            
            self.max_users = max_users
            self.max_items = max_items
            self.dropout_p = dropout_p
            self.num_emb = num_emb
            
            with self.name_scope():
                self.user_embeddings = gluon.nn.Embedding(max_users, num_emb)
                self.item_embeddings = gluon.nn.Embedding(max_items, num_emb)
                
                self.dropout_user = gluon.nn.Dropout(dropout_p)
                self.dropout_item = gluon.nn.Dropout(dropout_p)
    
                self.dense_user   = gluon.nn.Dense(num_emb, activation='relu')
                self.dense_item = gluon.nn.Dense(num_emb, activation='relu')
                
        def hybrid_forward(self, F, users, items):
            a = self.user_embeddings(users)
            a = self.dense_user(a)
            
            b = self.item_embeddings(items)
            b = self.dense_item(b)
    
            predictions = self.dropout_user(a) * self.dropout_item(b)     
            predictions = F.sum(predictions, axis=1)
            return predictions

.. code:: ipython3

    num_embeddings = 64
    
    net = MFBlock(max_users=customer_index.shape[0], 
                  max_items=product_index.shape[0],
                  num_emb=num_embeddings,
                  dropout_p=0.5)


Set Parameters
~~~~~~~~~~~~~~

Let’s initialize network weights and set our optimization parameters.

.. code:: ipython3

    # Initialize network parameters
    ctx = mx.gpu()
    net.collect_params().initialize(mx.init.Xavier(magnitude=60),
                                    ctx=ctx,
                                    force_reinit=True)
    net.hybridize()
    
    # Set optimization parameters
    opt = 'sgd'
    lr = 0.02
    momentum = 0.9
    wd = 0.
    
    trainer = gluon.Trainer(net.collect_params(),
                            opt,
                            {'learning_rate': lr,
                             'wd': wd,
                             'momentum': momentum})

Execute
~~~~~~~

Let’s define a function to carry out the training of our neural network.

.. code:: ipython3

    def execute(train_iter, test_iter, net, epochs, ctx):
        
        loss_function = gluon.loss.L2Loss()
        for e in range(epochs):
            
            print("epoch: {}".format(e))
            
            for i, (user, item, label) in enumerate(train_iter):
                    user = user.as_in_context(ctx)
                    item = item.as_in_context(ctx)
                    label = label.as_in_context(ctx)
                    
                    with mx.autograd.record():
                        output = net(user, item)               
                        loss = loss_function(output, label)
                        
                    loss.backward()
                    trainer.step(batch_size)
    
            print("EPOCH {}: MSE ON TRAINING and TEST: {}. {}".format(e,
                                                                       eval_net(train_iter, net, ctx, loss_function),
                                                                       eval_net(test_iter, net, ctx, loss_function)))
        print("end of training")
        return net

Let’s also define a function which evaluates our network on a given
dataset. This is called by our ``execute`` function above to provide
mean squared error values on our training and test datasets.

.. code:: ipython3

    def eval_net(data, net, ctx, loss_function):
        acc = MSE()
        for i, (user, item, label) in enumerate(data):
            
                user = user.as_in_context(ctx)
                item = item.as_in_context(ctx)
                label = label.as_in_context(ctx)
                predictions = net(user, item).reshape((batch_size, 1))
                acc.update(preds=[predictions], labels=[label])
       
        return acc.get()[1]

Now, let’s train for a few epochs.

.. code:: ipython3

    %%time
    
    epochs = 3
    
    trained_net = execute(train_iter, test_iter, net, epochs, ctx)

Early Validation
^^^^^^^^^^^^^^^^

We can see our training error going down, but our validation accuracy
bounces around a bit. Let’s check how our model is predicting for an
individual user. We could pick randomly, but for this case, let’s try
user #6.

.. code:: ipython3

    product_index['u6_predictions'] = trained_net(nd.array([6] * product_index.shape[0]).as_in_context(ctx), 
                                                  nd.array(product_index['item'].values).as_in_context(ctx)).asnumpy()
    product_index.sort_values('u6_predictions', ascending=False)

Now let’s compare this to the predictions for another user (we’ll try
user #7).

.. code:: ipython3

    product_index['u7_predictions'] = trained_net(nd.array([7] * product_index.shape[0]).as_in_context(ctx), 
                                                  nd.array(product_index['item'].values).as_in_context(ctx)).asnumpy()
    product_index.sort_values('u7_predictions', ascending=False)

The predicted ratings are different between the two users, but the same
top (and bottom) items for user #6 appear for #7 as well. Let’s look at
the correlation across the full set of 38K items to see if this
relationship holds.

.. code:: ipython3

    product_index[['u6_predictions', 'u7_predictions']].plot.scatter('u6_predictions', 'u7_predictions')
    plt.show()

We can see that this correlation is nearly perfect. Essentially the
average rating of items dominates across users and we’ll recommend the
same well-reviewed items to everyone. As it turns out, we can add more
embeddings and this relationship will go away since we’re better able to
capture differential preferences across users.

However, with just a 64 dimensional embedding, it took 7 minutes to run
just 3 epochs. If we ran this outside of our Notebook Instance we could
run larger jobs and move on to other work would improve productivity.

--------------

Train with SageMaker
--------------------

Now that we’ve trained on this smaller dataset, we can expand training
in SageMaker’s distributed, managed training environment.

Wrap Code
~~~~~~~~~

To use SageMaker’s pre-built MXNet container, we’ll need to wrap our
code from above into a Python script. There’s a great deal of
flexibility in using SageMaker’s pre-built containers, and detailed
documentation can be found
`here <https://github.com/aws/sagemaker-python-sdk#mxnet-sagemaker-estimators>`__,
but for our example, it consisted of: 1. Wrapping all data preparation
into a ``prepare_train_data`` function (we could name this whatever we
like) 1. Copying and pasting classes and functions from above
word-for-word 1. Defining a ``train`` function that: 1. Adds a bit of
new code to pick up the input TSV dataset on the SageMaker Training
cluster 1. Takes in a dict of hyperparameters (which we specified as
globals above) 1. Creates the net and executes training

.. code:: ipython3

    !cat recommender.py

Test Locally
~~~~~~~~~~~~

Now we can test our train function locally. This helps ensure we don’t
have any bugs before submitting our code to SageMaker’s pre-built MXNet
container.

.. code:: ipython3

    # %%time
    
    # import recommender
    
    # local_test_net, local_customer_index, local_product_index = recommender.train(
    #     {'train': '/tmp/recsys/'}, 
    #     {'num_embeddings': 64, 
    #      'opt': 'sgd', 
    #      'lr': 0.02, 
    #      'momentum': 0.9, 
    #      'wd': 0.,
    #      'epochs': 3},
    #     ['local'],
    #     1)

Move Data
~~~~~~~~~

Holding our data in memory works fine when we’re interactively exploring
a sample of data, but for larger, longer running processes, we’d prefer
to run them in the background with SageMaker Training. To do this, let’s
move the dataset to S3 so that it can be picked up by SageMaker
training. This is perfect for use cases like periodic re-training,
expanding to a larger dataset, or moving production workloads to larger
hardware.

.. code:: ipython3

    boto3.client('s3').copy({'Bucket': 'amazon-reviews-pds', 
                             'Key': 'tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz'},
                            bucket,
                            prefix + '/train/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz')

Submit
~~~~~~

Now, we can create an MXNet estimator from the SageMaker Python SDK. To
do so, we need to pass in: 1. Instance type and count for our SageMaker
Training cluster. SageMaker’s MXNet containers support distributed GPU
training, so we could easily set this to multiple ml.p2 or ml.p3
instances if we wanted. - *Note, this would require some changes to our
recommender.py script as we would need to setup the context an key value
store properly, as well as determining if and how to distribute the
training data.* 1. An S3 path for out model artifacts and a role with
access to S3 input and output paths. 1. Hyperparameters for our neural
network. Since with a 64 dimensional embedding, our recommendations
reverted too closely to the mean, let’s increase this by an order of
magnitude when we train outside of our local instance. We’ll also
increase the epochs to see how our accuracy evolves over time. We’ll
leave all other hyperparameters the same.

Once we use ``.fit()`` this creates a SageMaker Training Job that spins
up instances, loads the appropriate packages and data, runs our
``train`` function from ``recommender.py``, wraps up and saves model
artifacts to S3, and finishes by tearing down the cluster.

.. code:: ipython3

    m = MXNet('recommender.py', 
              py_version='py3',
              role=role, 
              train_instance_count=1, 
              train_instance_type="ml.p2.xlarge",
              output_path='s3://{}/{}/output'.format(bucket, prefix),
              hyperparameters={'num_embeddings': 512, 
                               'opt': opt, 
                               'lr': lr, 
                               'momentum': momentum, 
                               'wd': wd,
                               'epochs': 10},
             framework_version='1.1')
    
    m.fit({'train': 's3://{}/{}/train/'.format(bucket, prefix)})

--------------

Host
----

Now that we’ve trained our model, deploying it to a real-time,
production endpoint is easy.

.. code:: ipython3

    predictor = m.deploy(initial_instance_count=1, 
                         instance_type='ml.m4.xlarge')
    predictor.serializer = None

Now that we have an endpoint, let’s test it out. We’ll predict user #6’s
ratings for the top and bottom ASINs from our local model.

*This could be done by sending HTTP POST requests from a separate web
service, but to keep things easy, we’ll just use the ``.predict()``
method from the SageMaker Python SDK.*

.. code:: ipython3

    predictor.predict(json.dumps({'customer_id': customer_index[customer_index['user'] == 6]['customer_id'].values.tolist(), 
                                  'product_id': ['B00KH1O9HW', 'B00M5KODWO']}))

*Note, some of our predictions are actually greater than 5, which is to
be expected as we didn’t do anything special to account for ratings
being capped at that value. Since we are only looking to ranking by
predicted rating, this won’t create problems for our specific use case.*

Evaluate
~~~~~~~~

Let’s start by calculating a naive baseline to approximate how well our
model is doing. The simplest estimate would be to assume every user item
rating is just the average rating over all ratings.

*Note, we could do better by using each individual video’s average,
however, in this case it doesn’t really matter as the same conclusions
would hold.*

.. code:: ipython3

    print('Naive MSE:', np.mean((test_df['star_rating'] - np.mean(train_df['star_rating'])) ** 2))

Now, we’ll calculate predictions for our test dataset.

*Note, this will align closely to our CloudWatch output above, but may
differ slightly due to skipping partial mini-batches in our eval_net
function.*

.. code:: ipython3

    test_preds = []
    for array in np.array_split(test_df[['customer_id', 'product_id']].values, 40):
        test_preds += predictor.predict(json.dumps({'customer_id': array[:, 0].tolist(), 
                                                    'product_id': array[:, 1].tolist()}))
    
    test_preds = np.array(test_preds)
    print('MSE:', np.mean((test_df['star_rating'] - test_preds) ** 2))

We can see that our neural network and embedding model produces
substantially better results (~1.27 vs 1.65 on mean square error).

For recommender systems, subjective accuracy also matters. Let’s get
some recommendations for a random user to see if they make intuitive
sense.

.. code:: ipython3

    reduced_df[reduced_df['user'] == 6].sort_values(['star_rating', 'item'], ascending=[False, True])

As we can see, user #6 seems to like sprawling dramamtic television
series and sci-fi, but they dislike silly comedies.

Now we’ll loop through and predict user #6’s ratings for every common
video in the catalog, to see which ones we’d recommend and which ones we
wouldn’t.

.. code:: ipython3

    predictions = []
    for array in np.array_split(product_index['product_id'].values, 40):
        predictions += predictor.predict(json.dumps({'customer_id': customer_index[customer_index['user'] == 6]['customer_id'].values.tolist() * array.shape[0], 
                                                     'product_id': array.tolist()}))
    
    predictions = pd.DataFrame({'product_id': product_index['product_id'],
                                'prediction': predictions})

.. code:: ipython3

    titles = reduced_df.groupby('product_id')['product_title'].last().reset_index()
    predictions_titles = predictions.merge(titles)
    predictions_titles.sort_values(['prediction', 'product_id'], ascending=[False, True])

Indeed, our predicted highly rated shows have some well-reviewed TV
dramas and some sci-fi. Meanwhile, our bottom rated shows include
goofball comedies.

*Note, because of random initialization in the weights, results on
subsequent runs may differ slightly.*

Let’s confirm that we no longer have almost perfect correlation in
recommendations with user #7.

.. code:: ipython3

    predictions_user7 = []
    for array in np.array_split(product_index['product_id'].values, 40):
        predictions_user7 += predictor.predict(json.dumps({'customer_id': customer_index[customer_index['user'] == 7]['customer_id'].values.tolist() * array.shape[0], 
                                                           'product_id': array.tolist()}))
    plt.scatter(predictions['prediction'], np.array(predictions_user7))
    plt.show()

--------------

Wrap-up
-------

In this example, we developed a deep learning model to predict customer
ratings. This could serve as the foundation of a recommender system in a
variety of use cases. However, there are many ways in which it could be
improved. For example we did very little with: - hyperparameter tuning -
controlling for overfitting (early stopping, dropout, etc.) - testing
whether binarizing our target variable would improve results - including
other information sources (video genres, historical ratings, time of
review) - adjusting our threshold for user and item inclusion

In addition to improving the model, we could improve the engineering by:
- Setting the context and key value store up for distributed training -
Fine tuning our data ingestion (e.g. num_workers on our data iterators)
to ensure we’re fully utilizing our GPU - Thinking about how
pre-processing would need to change as datasets scale beyond a single
machine

Beyond that, recommenders are a very active area of research and
techniques from active learning, reinforcement learning, segmentation,
ensembling, and more should be investigated to deliver well-rounded
recommendations.

Clean-up (optional)
~~~~~~~~~~~~~~~~~~~

Let’s finish by deleting our endpoint to avoid stray hosting charges.

.. code:: ipython3

    sagemaker.Session().delete_endpoint(predictor.endpoint)
