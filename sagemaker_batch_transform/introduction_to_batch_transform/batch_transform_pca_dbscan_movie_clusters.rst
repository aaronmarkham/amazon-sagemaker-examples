Amazon SageMaker Batch Transform
================================

**Generating Machine Learning Model Predictions from a Batch Transformer
versus from a Real Time Endpoint**

--------------

--------------

Contents
--------

1.  `Background <#Background>`__
2.  `Setup <#Setup>`__
3.  `Data <#Data>`__
4.  `Dimensionality reduction <#Dimensionality-reduction>`__
5.  `Train PCA <#Train-PCA>`__
6.  `Batch prediction PCA <#Batch-prediction-PCA>`__
7.  `Real-time prediction
    comparison <#Real-time-prediction-comparison>`__
8.  `Batch prediction on new data <#Batch-prediction-on-new-data>`__
9.  `Clustering <#Clustering>`__
10. `Prepare BYO <#Prepare-BYO>`__
11. `Train DBSCAN <#Train-DBSCAN>`__
12. `Batch prediction DBSCAN <#Batch-prediction-DBSCAN>`__
13. `Evaluate <#Evaluate>`__
14. `Wrap-up <#Wrap-up>`__

--------------

Background
----------

This notebook provides an introduction to the Amazon SageMaker batch
transform functionality. Deploying a trained model to a hosted endpoint
has been available in SageMaker since launch and is a great way to
provide real-time predictions to a service like a website or mobile app.
But, if the goal is to generate predictions from a trained model on a
large dataset where minimizing latency isn’t a concern, then the batch
transform functionality may be easier, more scalable, and more
appropriate. This can be especially useful for cases like:

-  **One-off evaluations of model fit:** For example, we may want to
   compare accuracy of our trained model on new validation data that we
   collected after our initial training job.
-  **Using outputs from one model as the inputs to another:** For
   example, we may want use a pre-processing step like word embeddings,
   principal components, clustering, or TF-IDF, before training a second
   model to generate predictions from that information.
-  **When predictions will ultimately be served outside of Amazon
   SageMaker:** For example, we may have a large, but finite, set of
   predictions to generate which we then store in a fast-lookup
   datastore for serving.

Functionally, batch transform uses the same mechanics as real-time
hosting to generate predictions. It requires a web server that takes in
HTTP POST requests a single observation, or mini-batch, at a time.
However, unlike real-time hosted endpoints which have persistent
hardware (instances stay running until you shut them down), batch
transform clusters are torn down when the job completes.

The example we’ll walk through in this notebook starts with Amazon movie
review
`data <https://s3.amazonaws.com/amazon-reviews-pds/readme.html>`__,
performs on principal components on the large user-item review matrix,
and then uses DBSCAN to cluster movies in the reduced dimensional space.
This allows us to split the notebook into two parts as well as
showcasing how to use batch with SageMaker built-in algorithms, and the
bring your own algorithm use case.

If you are only interested in understanding how SageMaker batch
transform compares to hosting a real-time endpoint, you can stop running
the notebook before the clustering portion of the notebook.

--------------

Setup
-----

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting. We’ve specified the default SageMaker bucket,
   but you can change this.
-  The IAM role arn used to give training and hosting access to your
   data. See the AWS SageMaker documentation for information on how to
   setup an IAM role. Note, if more than one role is required for
   notebook instances, training, and/or hosting, please replace
   ``sagemaker.get_execution_role()`` with the appropriate full IAM role
   arn string(s).

.. code:: ipython3

    import sagemaker
    sess = sagemaker.Session()
    
    bucket = sess.default_bucket()
    prefix = 'sagemaker/DEMO-batch-transform'
    
    role = sagemaker.get_execution_role()

Now we’ll import the Python libraries we’ll need.

.. code:: ipython3

    import boto3
    import sagemaker
    import sagemaker.amazon.common as smac
    from sagemaker.amazon.amazon_estimator import get_image_uri
    from sagemaker.transformer import Transformer
    from sagemaker.predictor import csv_serializer, json_deserializer
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import scipy.sparse
    import os
    import json

Permissions
~~~~~~~~~~~

Running the clustering portion of this notebook requires permissions in
addition to the normal ``SageMakerFullAccess`` permissions. This is
because we’ll be creating a new repository in Amazon ECR. The easiest
way to add these permissions is simply to add the managed policy
``AmazonEC2ContainerRegistryFullAccess`` to the role that you used to
start your notebook instance. There’s no need to restart your notebook
instance when you do this, the new permissions will be available
immediately.

--------------

Data
----

Let’s start by bringing in our dataset from an S3 public bucket. The
Amazon review dataset contains 1 to 5 star ratings from over 2M Amazon
customers on over 160K digital videos. More details on this dataset can
be found at its `AWS Public Datasets
page <https://s3.amazonaws.com/amazon-reviews-pds/readme.html>`__.

*Note, because this dataset is over a half gigabyte, the load from S3
may take ~10 minutes. Also, since Amazon SageMaker Notebooks start with
a 5GB persistent volume by default, and we don’t need to keep this data
on our instance for long, we’ll bring it to the temporary volume (which
has up to 20GB of storage).*

.. code:: ipython3

    !mkdir /tmp/reviews/
    !aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz /tmp/reviews/

Let’s read the data into a `Pandas
DataFrame <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`__
so that we can begin to understand it.

*Note, we’ll set ``error_bad_lines=False`` when reading the file in as
there appear to be a very small number of records which would create a
problem otherwise.*

.. code:: ipython3

    df = pd.read_csv('/tmp/reviews/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz', delimiter='\t',error_bad_lines=False)
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

To keep the problem tractable and get started on batch transform
quickly, we’ll make a few simplifying transformations on the data. Let’s
start by reducing our dataset to users, items, and start ratings. We’ll
keep product title on the dataset for evaluating our clustering at the
end.

.. code:: ipython3

    df = df[['customer_id', 'product_id', 'star_rating', 'product_title']]

Now, because most users don’t rate most products, and there’s a long
tail of products that are almost never rated, we’ll tabulate common
percentiles.

.. code:: ipython3

    customers = df['customer_id'].value_counts()
    products = df['product_id'].value_counts()
    
    quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 1]
    print('customers\n', customers.quantile(quantiles))
    print('products\n', products.quantile(quantiles))

As we can see, only 0.1% of users have rated more than 36 movies. And,
only 25% of movies have been rated more than 8 times. For the purposes
of our analysis, we’d like to keep a large sample of popular movies for
our clustering, but base that only on heavy reviewers. So, we’ll limit
to customers who have reviewed 35+ movies and movies that have been
reviewed 20+ times.

.. code:: ipython3

    customers = customers[customers >= 35]
    products = products[products >= 20]
    
    reduced_df = df.merge(pd.DataFrame({'customer_id': customers.index})).merge(pd.DataFrame({'product_id': products.index}))

.. code:: ipython3

    customers = reduced_df['customer_id'].value_counts()
    products = reduced_df['product_id'].value_counts()

Now, we’ll setup to split our dataset into train and test.
Dimensionality reduction and clustering don’t always require a holdout
set to test accuracy, but it will allow us to illustrate how batch
prediction might be used when new data arrives. In this case, our test
dataset will be a simple 10% sample of items.

.. code:: ipython3

    test_products = products.sample(frac=0.1)
    train_products = products[~(products.index.isin(test_products.index))]

Now, to build our matrix, we’ll give each of our customers and products
their own unique, sequential index. This will allow us to easily hold
the data as a sparse matrix, and then write that out to S3 as a dense
matrix, which will serve as the input to our PCA algorithm.

.. code:: ipython3

    customer_index = pd.DataFrame({'customer_id': customers.index, 'user': np.arange(customers.shape[0])})
    train_product_index = pd.DataFrame({'product_id': train_products.index, 
                                        'item': np.arange(train_products.shape[0])})
    test_product_index = pd.DataFrame({'product_id': test_products.index, 
                                       'item': np.arange(test_products.shape[0])})
    
    train_df = reduced_df.merge(customer_index).merge(train_product_index)
    test_df = reduced_df.merge(customer_index).merge(test_product_index)

Next, we’ll create sparse matrices for the train and test datasets from
the indices we just created and an indicator for whether the customer
gave the rating 4 or more stars. Note that this inherently implies a
star rating below for all movies that a customer has not yet reviewed.
Although this isn’t strictly true (it’s possible the customer would
review it highly but just hasn’t seen it yet), our purpose is not to
predict ratings, just to understand how movies may cluster together, so
we use this simplification.

.. code:: ipython3

    train_sparse = scipy.sparse.csr_matrix((np.where(train_df['star_rating'].values >= 4, 1, 0), 
                                            (train_df['item'].values, train_df['user'].values)),
                                           shape=(train_df['item'].nunique(), customers.count()))
    
    test_sparse = scipy.sparse.csr_matrix((np.where(test_df['star_rating'].values >= 4, 1, 0), 
                                           (test_df['item'].values, test_df['user'].values)),
                                          shape=(test_df['item'].nunique(), customers.count()))

Now, we’ll save these files to dense CSVs. This will create a dense
matrix of movies by customers, with reviews as the entries, similar to:

+-----------+-------+-------+-------+---+-------+
| Item      | User1 | User2 | User3 | … | UserN |
+===========+=======+=======+=======+===+=======+
| **Item1** | 1     | 0     | 0     | … | 0     |
+-----------+-------+-------+-------+---+-------+
| **Item2** | 0     | 0     | 1     | … | 1     |
+-----------+-------+-------+-------+---+-------+
| **Item3** | 1     | 0     | 0     | … | 0     |
+-----------+-------+-------+-------+---+-------+
| **…**     | …     | …     | …     | … | …     |
+-----------+-------+-------+-------+---+-------+
| **ItemM** | 0     | 1     | 1     | … | 1     |
+-----------+-------+-------+-------+---+-------+

Which translates to User1 positively reviewing Items 1 and 3, User2
positively reviewing ItemM, and so on.

.. code:: ipython3

    np.savetxt('/tmp/reviews/train.csv',
               train_sparse.todense(),
               delimiter=',',
               fmt='%i')
    
    np.savetxt('/tmp/reviews/test.csv',
               test_sparse.todense(),
               delimiter=',',
               fmt='%i')

And upload them to S3. Note, we’ll keep them in separate prefixes to
ensure the test dataset isn’t picked up for training.

.. code:: ipython3

    train_s3 = sess.upload_data('/tmp/reviews/train.csv', 
                                bucket=bucket, 
                                key_prefix='{}/pca/train'.format(prefix))
    
    test_s3 = sess.upload_data('/tmp/reviews/test.csv',
                               bucket=bucket,
                               key_prefix='{}/pca/test'.format(prefix))

Finally, we’ll create an input which can be passed to our SageMaker
training estimator.

.. code:: ipython3

    train_inputs = sagemaker.s3_input(train_s3, content_type='text/csv;label_size=0')

--------------

Dimensionality reduction
------------------------

Now that we have our item user positive review matrix, we want to
perform Principal Components Analysis (PCA) on it. This can serve as an
effective pre-processing technique prior to clustering. Even though we
filtered out customers with very few reviews, we still have 2348 users.
If we wanted to cluster directly on this data, we would be in a very
high dimensional space. This runs the risk of the curse of
dimensionality. Essentially, because we have such a high dimensional
feature space, every point looks far away from all of the others on at
least some of those dimensions. So, We’ll use PCA to generate a much
smaller number of uncorrelated components. This should make finding
clusters easier.

Train PCA
~~~~~~~~~

Let’s start by creating a PCA estimator. We’ll define: - Algorithm
container path - IAM role for data permissions and - Harware setup
(instance count and type) - Output path (where our PCA model artifact
will be saved)

.. code:: ipython3

    container = get_image_uri(boto3.Session().region_name, 'pca', 'latest')

.. code:: ipython3

    pca = sagemaker.estimator.Estimator(container,
                                        role,
                                        train_instance_count=1,
                                        train_instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/pca/output'.format(bucket, prefix),
                                        sagemaker_session=sess)

Then we can define hyperparameters like: - ``feature_dim``: The number
of features (in this case users) in our input dataset. -
``num_components``: The number of features we want in our output dataset
(which we’ll pass to our clustering algorithm as input). -
``subtract_mean``: Debiases our features before running PCA. -
``algorithm_mode``: Since our dataset is rather large, we’ll use
randomized, which scales better.

See the
`documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/PCA-reference.html>`__
for more detail.

.. code:: ipython3

    pca.set_hyperparameters(feature_dim=customers.count(),
                            num_components=100,
                            subtract_mean=True,
                            algorithm_mode='randomized',
                            mini_batch_size=500)

And finally, we’ll use ``.fit()`` to start the training job.

.. code:: ipython3

    pca.fit({'train': train_inputs})

Batch prediction PCA
~~~~~~~~~~~~~~~~~~~~

Now that our PCA training job has finished, let’s generate some
predictions from it. We’ll start by creating a batch transformer. For
this, we need to specify: - Hardware specification (instance count and
type). Prediction is embarassingly parallel, so feel free to test this
with multiple instances, but since our dataset is not enormous, we’ll
stick to one. - ``strategy``: Which determines how records should be
batched into each prediction request within the batch transform job.
‘MultiRecord’ may be faster, but some use cases may require
‘SingleRecord’. - ``assemble_with``: Which controls how predictions are
output. ‘None’ does not perform any special processing, ‘Line’ places
each prediction on it’s own line. - ``output_path``: The S3 location for
batch transform to be output. Note, file(s) will be named with ‘.out’
suffixed to the input file(s) names. In our case this will be
‘train.csv.out’. Note that in this case, multiple batch transform runs
will overwrite existing values unless this is updated appropriately.

.. code:: ipython3

    pca_transformer = pca.transformer(instance_count=1,
                                      instance_type='ml.m4.xlarge',
                                      strategy='MultiRecord',
                                      assemble_with='Line',
                                      output_path='s3://{}/{}/pca/transform/train'.format(bucket, prefix))

Now, we’ll pass our training data in to get predictions from batch
transformer. A critical parameter to set properly here is
``split_type``. Since we are using CSV, we’ll specify ‘Line’, which
ensures we only pass one line at a time to our algorithm for prediction.
Had we not specified this, we’d attempt to pass all lines in our file,
which would exhaust our transformer instance’s memory.

*Note: Here we pass the S3 path as input rather than input we use in
``.fit()``.*

.. code:: ipython3

    pca_transformer.transform(train_s3, content_type='text/csv', split_type='Line')
    pca_transformer.wait()

Now that our batch transform job has completed, let’s take a look at the
output. Since we’ve reduced the dimensionality so much, the output is
reasonably small and we can just download it locally.

.. code:: ipython3

    !aws s3 cp --recursive $pca_transformer.output_path ./

.. code:: ipython3

    !head train.csv.out

We can see the records are output as JSON, which is typical for Amazon
SageMaker built-in algorithms. It’s the same format we’d see if we
performed real-time prediction. However, here, we didn’t have to stand
up a persistent endpoint, and we didn’t have to write code to loop
through our training dataset and invoke the endpoint one mini-batch at a
time. Just for the sake of comparison, we’ll show what that would look
like here.

Real-time prediction comparison (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we’ll deploy PCA to a real-time endpoint. As mentioned above, if our
use-case required individual predictions in near real-time, SageMaker
endpoints make sense. They can also be used for pseudo-batch prediction,
but the process is more involved than simply using SageMaker batch
transform.

We’ll start by deploying our PCA estimator.

.. code:: ipython3

    pca_predictor = pca.deploy(initial_instance_count=1,
                               instance_type='ml.m4.xlarge')

Now we need to specify our content type and how we serialize our request
data (which needs to be help in local memory) to that type.

.. code:: ipython3

    pca_predictor.content_type = 'text/csv'
    pca_predictor.serializer = csv_serializer
    pca_predictor.deserializer = json_deserializer

Then, we setup a loop to: 1. Cycle through our training dataset a 5MB or
less mini-batch at a time. 2. Invoke our endpoint. 3. Collect our
results.

Importantly, If we wanted to do this: 1. On a very large dataset, then
we’d need to work out a means of reading just some of the dataset into
memory at a time. 2. In parallel, then we’d need to monitor and
recombine the separate threads properly.

.. code:: ipython3

    components = []
    for array in np.array_split(np.array(train_sparse.todense()), 500):
        result = pca_predictor.predict(array)
        components += [r['projection'] for r in result['projections']]
    components = np.array(components)

.. code:: ipython3

    components[:5, ]

In order to use these values in a subsequent model, we would also have
to output ``components`` to a local file and then save that file to S3.
And, of course we wouldn’t want to forget to delete our endpoint.

.. code:: ipython3

    sess.delete_endpoint(pca_predictor.endpoint)

Batch prediction on new data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you may acquire more data after initially training your model.
SageMaker batch transform can be used in cases like these as well. We
can start by creating a model and getting it’s name.

.. code:: ipython3

    pca_model = sess.create_model_from_job(pca._current_job_name, name='{}-test'.format(pca._current_job_name))

Now, we can create a transformer starting from the SageMaker model. Our
arguments are the same as when we created the transformer from the
estimator except for the additional model name argument.

.. code:: ipython3

    pca_test_transformer = Transformer(pca_model,
                                       1,
                                       'ml.m4.xlarge',
                                       output_path='s3://{}/{}/pca/transform/test'.format(bucket, prefix),
                                       sagemaker_session=sess,
                                       strategy='MultiRecord',
                                       assemble_with='Line')
    pca_test_transformer.transform(test_s3, content_type='text/csv', split_type='Line')
    pca_test_transformer.wait()

Let’s pull this in as well and take a peak to confirm it’s what we
expected. Note, since we used ‘MultiRecord’, the first line in our file
is enormous, so we’ll only print out the first 10,000 bytes.

.. code:: ipython3

    !aws s3 cp --recursive $pca_test_transformer.output_path ./

.. code:: ipython3

    !head -c 10000 test.csv.out

We can see that we have output the reduced dimensional components for
our test dataset, using the model we built from our training dataset.

At this point in time, we’ve shown all of the batch functionality you
need to get started using it in Amazon SageMaker. The second half of the
notebook takes our first set of batch outputs from SageMaker’s PCA
algorithm and passes them to a bring your own container version of the
DBSCAN clustering algorithm. Feel free to continue on for a deep dive.

--------------

--------------

Clustering (Optional)
---------------------

For the second half of this notebook we’ll show you how you can use
batch transform with a container that you’ve created yourself. This uses
`R <https://www.r-project.org/>`__ to run the DBSCAN clustering
algorithm on the reduced dimensional space which was output from
SageMaker PCA.

We’ll start by walking through the three scripts we’ll need for bringing
our DBSCAN container to SageMaker.

Prepare BYO
~~~~~~~~~~~

Dockerfile
^^^^^^^^^^

``Dockerfile`` defines what libraries should be in our container. We
start with an Ubuntu base, and install R, dbscan, and plumber libraries.
Then we add ``dbscan.R`` and ``plumber.R`` files from our local
filesystem to our container. Finally, we set it to run ``dbscan.R`` as
the entrypoint when launched.

*Note: Smaller containers are preferred for Amazon SageMaker as they
lead to faster spin up times in training and endpoint creation, so we
keep the Dockerfile minimal.*

.. code:: ipython3

    !cat Dockerfile

dbscan.R
^^^^^^^^

``dbscan.R`` is the script that runs when the container starts. It looks
for either ‘train’ or ‘serve’ arguments to determine if we are training
our algorithm or serving predictions, and it contains two functions
``train()`` and ``serve()``, which are executed when appropriate. It
also includes some setup at the top to create shortcut paths so our
algorithm can use the container directories as they are setup by
SageMaker
(`documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html>`__).

The ``train()`` function reads in training data, which is actually the
output from the SageMaker PCA batch transform job. Appropriate
transformations to read the file’s JSON into a data frame are made. And
then takes in hyperparameters for DBSCAN. In this case, that consists of
``eps`` (size of the neighborhood fo assess density) and ``minPts``
(minimum number of points needed in the ``eps`` region). The DBSCAN
model is fit, and model artifacts are output.

The ``serve()`` function sets up a
`plumber <https://www.rplumber.io/>`__ API. In this case, most of the
work of generating predictions is done in the ``plumber.R`` script.

.. code:: ipython3

    !cat dbscan.R

plumber.R
^^^^^^^^^

This script functions to generate predictions for both real-time
prediction from a SageMaker hosted endpoint and batch transform. So, we
return an empty message body on ``/ping`` and we load our model and
generate predictions for requests sent to ``/invocations``. We’re
inherently expecting scoring input to come in the same SageMaker PCA
output JSON format as we did in training. This assumption may not be
valid if we were making real-time requests rather than batch requests.
But, we could include additional logic to accommodate multiple input
formats as needed.

.. code:: ipython3

    !cat plumber.R

Publish
~~~~~~~

In the next step we’ll build our container and publish it to ECR where
SageMaker can access it.

This command will take several minutes to run the first time.

.. code:: sh

    %%sh
    
    # The name of our algorithm
    algorithm_name=dbscan
    
    #set -e # stop if anything fails
    
    account=$(aws sts get-caller-identity --query Account --output text)
    
    # Get the region defined in the current configuration (default to us-west-2 if none defined)
    region=$(aws configure get region)
    region=${region:-us-west-2}
    
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
    
    # If the repository doesn't exist in ECR, create it.
    
    aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
    
    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
    fi
    
    # Get the login command from ECR and execute it directly
    $(aws ecr get-login --region ${region} --no-include-email)
    
    # Build the docker image locally with the image name and then push it to ECR
    # with the full name.
    docker build  -t ${algorithm_name} .
    docker tag ${algorithm_name} ${fullname}
    
    docker push ${fullname}

Train DBSCAN
~~~~~~~~~~~~

Now that our container is built, we can create an estimator and use it
to train our DBSCAN clustering algorithm. note, we’re passing in
``pca_transformer.output_path`` as our input training data.

.. code:: ipython3

    region = boto3.Session().region_name
    account = boto3.client('sts').get_caller_identity().get('Account')

.. code:: ipython3

    dbscan = sagemaker.estimator.Estimator('{}.dkr.ecr.{}.amazonaws.com/dbscan:latest'.format(account, region),
                                           role,
                                           train_instance_count=1,
                                           train_instance_type='ml.m4.xlarge',
                                           output_path='s3://{}/{}/dbscan/output'.format(bucket, prefix),
                                           sagemaker_session=sess)
    dbscan.set_hyperparameters(minPts=5)
    dbscan.fit({'train': pca_transformer.output_path})

Batch prediction
~~~~~~~~~~~~~~~~

Next, we’ll kick off batch prediction for DBSCAN. In this case, we’ll
choose to do this on our test output from above. This again illustrates
that although batch transform can be used to generate predictions on the
training data, it can just as easily be used on holdout or future data
as well.

*Note: Here we use strategy ‘SingleRecord’ because each line from our
previous batch output is from a ‘MultiRecord’ output, so we’ll process
all of those at once.*

.. code:: ipython3

    dbscan_transformer = dbscan.transformer(instance_count=1,
                                            instance_type='ml.m4.xlarge',
                                            output_path='s3://{}/{}/dbscan/transform'.format(bucket, prefix),
                                            strategy='SingleRecord',
                                            assemble_with='Line')
    dbscan_transformer.transform(pca_test_transformer.output_path, 
                                 content_type='text/csv', 
                                 split_type='Line')
    dbscan_transformer.wait()

--------------

Evaluate
--------

We’ll start by bringing in the cluster output dataset locally.

.. code:: ipython3

    !aws s3 cp --recursive $dbscan_transformer.output_path ./

Next we’ll read the JSON output in to pick up the cluster membership for
each observation.

.. code:: ipython3

    dbscan_output = []
    with open('test.csv.out.out', 'r') as f:
        for line in f:
            result = json.loads(line)[0].split(',')
            dbscan_output += [r for r in result]

We’ll merge that information back onto our test data frame.

.. code:: ipython3

    dbscan_clusters = pd.DataFrame({'item': np.arange(test_products.shape[0]),
                                    'cluster': dbscan_output})
    
    dbscan_clusters_items = test_df.groupby('item')['product_title'].first().reset_index().merge(dbscan_clusters)

And now we’ll take a look at 5 example movies from each cluster.

.. code:: ipython3

    dbscan_clusters_items.sort_values(['cluster', 'item']).groupby('cluster').head(2)

Our clustering could likely use some tuning as we see some skewed
cluster distributions. But, we do find a few commonalities like
“Charlotte’s Web” and “Wild Kratts Season 3” both showing up in cluster
#32, which may be kid’s videos.

*Note: Due to inherent randomness of the algorithms and data
manipulations, your specific results may differ from those mentioned
above.*

--------------

Wrap-up
-------

In this notebook we showcased how to use Amazon SageMaker batch
transform with built-in algorithms and with a bring your own algorithm
container. This allowed us to set it up so that our custom container
ingested the batch output of the first algorithm. Extensions could
include: - Moving to larger datasets, where batch transform can be
particularly effective. - Using batch transform with the SageMaker
pre-built deep learning framework containers. - Adding more steps or
further automating the machine learning pipeline we’ve started.
