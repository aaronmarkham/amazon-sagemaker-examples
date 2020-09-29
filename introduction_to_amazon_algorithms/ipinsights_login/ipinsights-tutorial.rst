An Introduction to the Amazon SageMaker IP Insights Algorithm
=============================================================

#### Unsupervised anomaly detection for susicipous IP addresses
---------------------------------------------------------------

1. `Introduction <#Introduction>`__
2. `Setup <#Setup>`__
3. `Training <#Training>`__
4. `Inference <#Inference>`__
5. `Epilogue <#Epilogue>`__

## Introduction
---------------

The Amazon SageMaker IP Insights algorithm uses statistical modeling and
neural networks to capture associations between online resources (such
as account IDs or hostnames) and IPv4 addresses. Under the hood, it
learns vector representations for online resources and IP addresses.
This essentially means that if the vector representing an IP address and
an online resource are close together, then it is likey for that IP
address to access that online resource, even if it has never accessed it
before.

In this notebook, we use the Amazon SageMaker IP Insights algorithm to
train a model on synthetic data. We then use this model to perform
inference on the data and show how to discover anomalies. After running
this notebook, you should be able to:

-  obtain, transform, and store data for use in Amazon SageMaker,
-  create an AWS SageMaker training job to produce an IP Insights model,
-  use the model to perform inference with an Amazon SageMaker endpoint.

If you would like to know more, please check out the `SageMaker IP
Inisghts
Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights.html>`__.

## Setup
--------

*This notebook was created and tested on a ml.m4.xlarge notebook
instance.*

Our first step is to setup our AWS credentials so that AWS SageMaker can
store and access training data and model artifacts.

Select Amazon S3 Bucket
~~~~~~~~~~~~~~~~~~~~~~~

We first need to specify the locations where we will store our training
data and trained model artifacts. **This is the only cell of this
notebook that you will need to edit.** In particular, we need the
following data:

-  ``bucket`` - An S3 bucket accessible by this account.
-  ``prefix`` - The location in the bucket where this notebook’s input
   and output data will be stored. (The default value is sufficient.)

.. code:: ipython3

    import boto3
    import botocore
    import os
    import sagemaker
    
    
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/ipinsights-tutorial'
    execution_role = sagemaker.get_execution_role()
    
    
    # check if the bucket exists
    try:
        boto3.Session().client('s3').head_bucket(Bucket=bucket)
    except botocore.exceptions.ParamValidationError as e:
        print('Hey! You either forgot to specify your S3 bucket'
              ' or you gave your bucket an invalid name!')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '403':
            print("Hey! You don't have permission to access the bucket, {}.".format(bucket))
        elif e.response['Error']['Code'] == '404':
            print("Hey! Your bucket, {}, doesn't exist!".format(bucket))
        else:
            raise
    else:
        print('Training input/output will be stored in: s3://{}/{}'.format(bucket, prefix))

Dataset
~~~~~~~

Apache Web Server (“httpd”) is the most popular web server used on the
internet. And luckily for us, it logs all requests processed by the
server - by default. If a web page requires HTTP authentication, the
Apache Web Server will log the IP address and authenticated user name
for each requested resource.

The `access logs <https://httpd.apache.org/docs/2.4/logs.html>`__ are
typically on the server under the file ``/var/log/httpd/access_log``.
From the example log output below, we see which IP addresses each user
has connected with:

::

   192.168.1.100 - user1 [15/Oct/2018:18:58:32 +0000] "GET /login_success?userId=1 HTTP/1.1" 200 476 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"
   192.168.1.102 - user2 [15/Oct/2018:18:58:35 +0000] "GET /login_success?userId=2 HTTP/1.1" 200 - "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"
   ...

If we want to train an algorithm to detect suspicious activity, this
dataset is ideal for SageMaker IP Insights.

First, we determine the resource we want to be analyzing (such as a
login page or access to a protected file). Then, we construct a dataset
containing the history of all past user interactions with the resource.
We extract out each ‘access event’ from the log and store the
corresponding user name and IP address in a headerless CSV file with two
columns. The first column will contain the user identifier string, and
the second will contain the IPv4 address in decimal-dot notation.

::

   user1, 192.168.1.100
   user2, 193.168.1.102
   ...

As a side note, the dataset should include all access events. That means
some ``<user_name, ip_address>`` pairs will be repeated.

User Activity Simulation
^^^^^^^^^^^^^^^^^^^^^^^^

For this example, we are going to simulate our own web-traffic logs. We
mock up a toy website example and simulate users logging into the
website from mobile devices.

The details of the simulation are explained in the script
`here <./generate_data.py>`__.

.. code:: ipython3

    # Install dependency
    !pip install tqdm

.. code:: ipython3

    from generate_data import generate_dataset
    
    # We simulate traffic for 10,000 users. This should yield about 3 million log lines (~700 MB). 
    NUM_USERS = 10000
    log_file = 'ipinsights_web_traffic.log'
    generate_dataset(NUM_USERS, log_file)
    
    # Visualize a few log lines
    !head $log_file

Prepare the dataset
~~~~~~~~~~~~~~~~~~~

Now that we have our logs, we need to transform them into a format that
IP Insights can use. As we mentioned above, we need to: 1. Choose the
resource which we want to analyze users’ history for 2. Extract our
users’ usage history of IP addresses 3. In addition, we want to separate
our dataset into a training and test set. This will allow us to check
for overfitting by evaluating our model on ‘unseen’ login events.

For the rest of the notebook, we assume that the Apache Access Logs are
in the Common Log Format as defined by the `Apache
documentation <https://httpd.apache.org/docs/2.4/logs.html#accesslog>`__.
We start with reading the logs into a Pandas DataFrame for easy data
exploration and pre-processing.

.. code:: ipython3

    import pandas as pd
    
    df = pd.read_csv(
        log_file,
        sep=" ",
        na_values='-',
        header=None,
        names=['ip_address','rcf_id','user','timestamp','time_zone','request', 'status', 'size', 'referer', 'user_agent']
    )
    df.head()

We convert the log timestamp strings into Python datetimes so that we
can sort and compare the data more easily.

.. code:: ipython3

    # Convert time stamps to DateTime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='[%d/%b/%Y:%H:%M:%S')

We also verify the time zones of all of the time stamps. If the log
contains more than one time zone, we would need to standardize the
timestamps.

.. code:: ipython3

    # Check if they are all in the same timezone
    num_time_zones = len(df['time_zone'].unique())
    num_time_zones

As we see above, there is only one value in the entire ``time_zone``
column. Therefore, all of the timestamps are in the same time zone, and
we do not need to standardize them. We can skip the next cell and go to
`1. Selecting a Resource <#1.-Select-Resource>`__.

If there is more than one time_zone in your dataset, then we parse the
timezone offset and update the corresponding datetime object.

**Note:** The next cell takes about 5-10 minutes to run.

.. code:: ipython3

    from datetime import datetime
    import pytz
    
    
    def apply_timezone(row):
        tz = row[1]
        tz_offset = int(tz[:3]) * 60   # Hour offset
        tz_offset += int(tz[3:5])      # Minutes offset
        return row[0].replace(tzinfo=pytz.FixedOffset(tz_offset))
    
    if num_time_zones > 1:
        df['timestamp'] = df[['timestamp','time_zone']].apply(apply_timezone, axis=1)

1. Select Resource
^^^^^^^^^^^^^^^^^^

Our goal is to train an IP Insights algorithm to analyze the history of
user logins such that we can predict how suspicious a login event is.

In our simulated web server, the server logs a ``GET`` request to the
``/login_success`` page everytime a user successfully logs in. We filter
our Apache logs for ``GET`` requests for ``/login_success``. We also
filter for requests that have a ``status_code == 200``, to ensure that
the page request was well formed.

**Note:** every web server handles logins differently. For your dataset,
determine which resource you will need to be analyzing to correctly
frame this problem. Depending on your usecase, you may need to do more
data exploration and preprocessing.

.. code:: ipython3

    df = df[(df['request'].str.startswith('GET /login_success')) & (df['status'] == 200)]

2. Extract Users and IP address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that our DataFrame only includes log events for the resource we want
to analyze, we extract the relevant fields to construct a IP Insights
dataset.

IP Insights takes in a headerless CSV file with two columns: an entity
(username) ID string and the IPv4 address in decimal-dot notation.
Fortunately, the Apache Web Server Access Logs output IP addresses and
authentcated usernames in their own columns.

**Note:** Each website handles user authentication differently. If the
Access Log does not output an authenticated user, you could explore the
website’s query strings or work with your website developers on another
solution.

.. code:: ipython3

    df = df[['user', 'ip_address', 'timestamp']]

3. Create training and test dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As part of training a model, we want to evaluate how it generalizes to
data it has never seen before.

Typically, you create a test set by reserving a random percentage of
your dataset and evaluating the model after training. However, for
machine learning models that make future predictions on historical data,
we want to use out-of-time testing. Instead of randomly sampling our
dataset, we split our dataset into two contiguous time windows. The
first window is the training set, and the second is the test set.

We first look at the time range of our dataset to select a date to use
as the partition between the training and test set.

.. code:: ipython3

    df['timestamp'].describe()

We have login events for 10 days. Let’s take the first week (7 days) of
data as training and then use the last 3 days for the test set.

.. code:: ipython3

    time_partition = datetime(2018, 11, 11, tzinfo=pytz.FixedOffset(0)) if num_time_zones > 1 else datetime(2018, 11, 11)
    
    train_df = df[df['timestamp'] <= time_partition]
    test_df = df[df['timestamp'] > time_partition]

Now that we have our training dataset, we shuffle it.

Shuffling improves the model’s performance since SageMaker IP Insights
uses stochastic gradient descent. This ensures that login events for the
same user are less likely to occur in the same mini batch. This allows
the model to improve its performance in between predictions of the same
user, which will improve training convergence.

.. code:: ipython3

    # Shuffle train data 
    train_df = train_df.sample(frac=1)
    train_df.head()

Store Data on S3
~~~~~~~~~~~~~~~~

Now that we have simulated (or scraped) our datasets, we have to prepare
and upload it to S3.

We will be doing local inference, therefore we don’t need to upload our
test dataset.

.. code:: ipython3

    # Output dataset as headerless CSV 
    train_data = train_df.to_csv(index=False, header=False, columns=['user', 'ip_address'])

.. code:: ipython3

    # Upload data to S3 key
    train_data_file = 'train.csv'
    key = os.path.join(prefix, 'train', train_data_file)
    s3_train_data = 's3://{}/{}'.format(bucket, key)
    
    print('Uploading data to: {}'.format(s3_train_data))
    boto3.resource('s3').Bucket(bucket).Object(key).put(Body=train_data)
    
    # Configure SageMaker IP Insights Input Channels
    input_data = {
        'train': sagemaker.session.s3_input(s3_train_data, distribution='FullyReplicated', content_type='text/csv')
    }

## Training
-----------

Once the data is preprocessed and available in the necessary format, the
next step is to train our model on the data. There are number of
parameters required by the SageMaker IP Insights algorithm to configure
the model and define the computational environment in which training
will take place. The first of these is to point to a container image
which holds the algorithms training and hosting code:

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    image = get_image_uri(boto3.Session().region_name, 'ipinsights')

Then, we need to determine the training cluster to use. The IP Insights
algorithm supports both CPU and GPU training. We recommend using GPU
machines as they will train faster. However, when the size of your
dataset increases, it can become more economical to use multiple CPU
machines running with distributed training. See `Recommended Instance
Types <https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights.html#ip-insights-instances>`__
for more details.

Training Job Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **train_instance_type**: the instance type to train on. We recommend
   ``p3.2xlarge`` for single GPU, ``p3.8xlarge`` for multi-GPU, and
   ``m5.2xlarge`` if using distributed training with CPU;
-  **train_instance_count**: the number of worker nodes in the training
   cluster.

We need to also configure SageMaker IP Insights-specific hypeparameters:

Model Hyperparameters
~~~~~~~~~~~~~~~~~~~~~

-  **num_entity_vectors**: the total number of embeddings to train. We
   use an internal hashing mechanism to map the entity ID strings to an
   embedding index; therefore, using an embedding size larger than the
   total number of possible values helps reduce the number of hash
   collisions. We recommend this value to be 2x the total number of
   unique entites (i.e. user names) in your dataset;
-  **vector_dim**: the size of the entity and IP embedding vectors. The
   larger the value, the more information can be encoded using these
   representations but using too large vector representations may cause
   the model to overfit, especially for small training data sets;
-  **num_ip_encoder_layers**: the number of layers in the IP encoder
   network. The larger the number of layers, the higher the model
   capacity to capture patterns among IP addresses. However, large
   number of layers increases the chance of overfitting.
   ``num_ip_encoder_layers=1`` is a good value to start experimenting
   with;
-  **random_negative_sampling_rate**: the number of randomly generated
   negative samples to produce per 1 positive sample;
   ``random_negative_sampling_rate=1`` is a good value to start
   experimenting with;

   -  Random negative samples are produced by drawing each octet from a
      uniform distributed of [0, 255];

-  **shuffled_negative_sampling_rate**: the number of shuffled negative
   samples to produce per 1 positive sample;
   ``shuffled_negative_sampling_rate=1`` is a good value to start
   experimenting with;

   -  Shuffled negative samples are produced by shuffling the accounts
      within a batch;

Training Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~

-  **epochs**: the number of epochs to train. Increase this value if you
   continue to see the accuracy and cross entropy improving over the
   last few epochs;
-  **mini_batch_size**: how many examples in each mini_batch. A smaller
   number improves convergence with stochastic gradient descent. But a
   larger number is necessary if using shuffled_negative_sampling to
   avoid sampling a wrong account for a negative sample;
-  **learning_rate**: the learning rate for the Adam optimizer (try
   ranges in [0.001, 0.1]). Too large learning rate may cause the model
   to diverge since the training would be likely to overshoot minima. On
   the other hand, too small learning rate slows down the convergence;
-  **weight_decay**: L2 regularization coefficient. Regularization is
   required to prevent the model from overfitting the training data. Too
   large of a value will prevent the model from learning anything;

For more details, see `Amazon SageMaker IP Insights
(Hyperparameters) <https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights-hyperparameters.html>`__.
Additionally, most of these hyperparameters can be found using SageMaker
Automatic Model Tuning; see `Amazon SageMaker IP Insights (Model
Tuning) <https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights-tuning.html>`__
for more details.

.. code:: ipython3

    # Set up the estimator with training job configuration
    ip_insights = sagemaker.estimator.Estimator(
        image, 
        execution_role, 
        train_instance_count=1, 
        train_instance_type='ml.p3.2xlarge',
        output_path='s3://{}/{}/output'.format(bucket, prefix),
        sagemaker_session=sagemaker.Session())
    
    # Configure algorithm-specific hyperparameters
    ip_insights.set_hyperparameters(
        num_entity_vectors='20000',
        random_negative_sampling_rate='5',
        vector_dim='128', 
        mini_batch_size='1000',
        epochs='5',
        learning_rate='0.01',
    )
    
    # Start the training job (should take about ~1.5 minute / epoch to complete)  
    ip_insights.fit(input_data)

If you see the message

::

   > Completed - Training job completed

at the bottom of the output logs then that means training successfully
completed and the output of the SageMaker IP Insights model was stored
in the specified output path. You can also view information about and
the status of a training job using the AWS SageMaker console. Just click
on the “Jobs” tab and select training job matching the training job
name, below:

.. code:: ipython3

    print('Training job name: {}'.format(ip_insights.latest_training_job.job_name))

## Inference
------------

Now that we have trained a SageMaker IP Insights model, we can deploy
the model to an endpoint to start performing inference on data. In this
case, that means providing it a ``<user, IP address>`` pair and
predicting their compatability scores.

We can create an inference endpoint using the SageMaker Python SDK
``deploy()``\ function from the job we defined above. We specify the
instance type where inference will be performed, as well as the initial
number of instnaces to spin up. We recommend using the ``ml.m5``
instance as it provides the most memory at the lowest cost. Verify how
large your model is in S3 and pick the instance type with the
appropriate amount of memory.

.. code:: ipython3

    predictor = ip_insights.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge'
    )

Congratulations, you now have a SageMaker IP Insights inference
endpoint! You could start integrating this endpoint with your production
services to start querying incoming requests for abnormal behavior.

You can confirm the endpoint configuration and status by navigating to
the “Endpoints” tab in the AWS SageMaker console and selecting the
endpoint matching the endpoint name below:

.. code:: ipython3

    print('Endpoint name: {}'.format(predictor.endpoint))

Data Serialization/Deserialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can pass data in a variety of formats to our inference endpoint. In
this example, we will pass CSV-formmated data. Other available formats
are JSON-formated and JSON Lines-formatted. We make use of the SageMaker
Python SDK utilities: ``csv_serializer`` and ``json_deserializer`` when
configuring the inference endpoint

.. code:: ipython3

    from sagemaker.predictor import csv_serializer, json_deserializer
    
    predictor.content_type = 'text/csv'
    predictor.serializer = csv_serializer
    predictor.accept = 'application/json'
    predictor.deserializer = json_deserializer

Now that the predictor is configured, it is as easy as passing in a
matrix of inference data. We can take a few samples from the simulated
dataset above, so we can see what the output looks like.

.. code:: ipython3

    inference_data = [(data[0], data[1]) for data in train_df[:5].values]
    predictor.predict(inference_data)

By default, the predictor will only output the ``dot_product`` between
the learned IP address and the online resource (in this case, the user
ID). The dot product summarizes the compatibility between the IP address
and online resource. The larger the value, the more the algorithm thinks
the IP address is likely to be used by the user. This compatability
score is sufficient for most applications, as we can define a threshold
for what we constitute as an anomalous score.

However, more advanced users may want to inspect the learned embeddings
and use them in further applications. We can configure the predictor to
provide the learned embeddings by specifing the ``verbose=True``
parameter to the Accept heading. You should see that each ‘prediction’
object contains three keys: ``ip_embedding``, ``entity_embedding``, and
``dot_product``.

.. code:: ipython3

    predictor.accept = 'application/json; verbose=True'
    predictor.predict(inference_data)

## Compute Anomaly Scores
-------------------------

The ``dot_product`` output of the model provides a good measure of how
compatible an IP address and online resource are. However, the range of
the dot_product is unbounded. This means to be able to consider an event
as anomolous we need to define a threshold. Such that when we score an
event, if the dot_product is above the threshold we can flag the
behavior as anomolous.However, picking a threshold can be more of an
art, and a good threshold depends on the specifics of your problem and
dataset.

In the following section, we show how to pick a simple threshold by
comparing the score distributions between known normal and malicious
traffic: 1. We construct a test set of ‘Normal’ traffic; 2. Inject
‘Malicious’ traffic into the dataset; 3. Plot the distribution of
dot_product scores for the model on ‘Normal’ trafic and the ‘Malicious’
traffic. 3. Select a threshold value which separates the normal
distribution from the malicious traffic threshold. This value is based
on your false-positive tolerance.

1. Construct ‘Normal’ Traffic Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We previously `created a test
set <#3.-Create-training-and-test-dataset>`__ from our simulated Apache
access logs dataset. We use this test dataset as the ‘Normal’ traffic in
the test case.

.. code:: ipython3

    test_df.head()

2. Inject Malicious Traffic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we had a dataset with enough real malicious activity, we would use
that to determine a good threshold. Those are hard to come by. So
instead, we simulate malicious web traffic that mimics a realistic
attack scenario.

We take a set of user accounts from the test set and randomly generate
IP addresses. The users should not have used these IP addresses during
training. This simulates an attacker logging in to a user account
without knowledge of their IP history.

.. code:: ipython3

    import numpy as np
    from generate_data import draw_ip
    
    # We only need the dot product. Let's reset the predictor output type.
    predictor.accept = 'application/json; verbose=False'
    
    
    def score_ip_insights(predictor, df):
        
        def get_score(result):
            """Return the negative to the dot product of the predictions from the model."""
            return [-prediction["dot_product"] for prediction in result["predictions"]]
        
        df = df[['user', 'ip_address']]
        result = predictor.predict(df.values)
        return get_score(result)
    
    
    def create_test_case(train_df, test_df, num_samples, attack_freq):
        """Creates a test case from provided train and test data frames. 
        
        This generates test case for accounts that are both in training and testing data sets.
    
        :param train_df: (panda.DataFrame with columns ['user', 'ip_address']) training DataFrame
        :param test_df: (panda.DataFrame with columns ['user', 'ip_address']) testing DataFrame
        :param num_samples: (int) number of test samples to use
        :param attack_freq: (float) the ratio of negative_samples:positive_samples to generate for test case 
        :return: DataFrame with both good and bad traffic, with labels
        """
        # Get all possible accounts. The IP Insights model can only make predictions on users it has seen in training
        # Therefore, filter the test dataset for unseen accounts, as their results will not mean anything.
        valid_accounts = set(train_df['user'])
        valid_test_df = test_df[test_df['user'].isin(valid_accounts)]
    
        good_traffic = valid_test_df.sample(num_samples, replace=False)
        good_traffic = good_traffic[['user', 'ip_address']]
        good_traffic['label'] = 0
    
        # Generate malicious traffic
        num_bad_traffic = int(num_samples * attack_freq)
        bad_traffic_accounts = np.random.choice(list(valid_accounts), size=num_bad_traffic, replace=True) 
        bad_traffic_ips = [draw_ip() for i in range(num_bad_traffic)]
        bad_traffic = pd.DataFrame({'user': bad_traffic_accounts, 'ip_address': bad_traffic_ips})
        bad_traffic['label'] = 1
        
        # All traffic labels are: 0 for good traffic; 1 for bad traffic. 
        all_traffic = good_traffic.append(bad_traffic)
    
        return all_traffic

.. code:: ipython3

    NUM_SAMPLES = 100000
    test_case = create_test_case(train_df, test_df, num_samples=NUM_SAMPLES, attack_freq=1)
    test_case.head()

.. code:: ipython3

    test_case_scores = score_ip_insights(predictor, test_case)

3. Plot Distribution
~~~~~~~~~~~~~~~~~~~~

Now, we plot the distribution of scores. Looking at this distribution
will inform us on where we can set a good threshold, based on our risk
tolerance.

.. code:: ipython3

    %matplotlib inline
    import matplotlib.pyplot as plt
    
    n, x = np.histogram(test_case_scores[:NUM_SAMPLES], bins=100, density=True)
    plt.plot(x[1:], n)
    
    n, x = np.histogram(test_case_scores[NUM_SAMPLES:], bins=100, density=True)
    plt.plot(x[1:], n)
    
    plt.legend(["Normal", "Random IP"])
    plt.xlabel("IP Insights Score")
    plt.ylabel("Frequency")
    
    plt.figure()

4. Selecting a Good Threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we see in the figure above, there is a clear separation between
normal traffic and random traffic. We could select a threshold depending
on the application.

-  If we were working with low impact decisions, such as whether to ask
   for another factor or authentication during login, we could use a
   ``threshold = 0.0``. This would result in catching more
   true-positives, at the cost of more false-positives.

-  If our decision system were more sensitive to false positives, we
   could choose a larger threshold, such as ``threshold = 10.0``. That
   way if we were sending the flagged cases to manual investigation, we
   would have a higher confidence that the acitivty was suspicious.

.. code:: ipython3

    threshold = 0.0
    
    flagged_cases = test_case[np.array(test_case_scores) > threshold]
    
    num_flagged_cases = len(flagged_cases)
    num_true_positives = len(flagged_cases[flagged_cases['label'] == 1])
    num_false_positives = len(flagged_cases[flagged_cases['label'] == 0])
    num_all_positives = len(test_case.loc[test_case['label'] == 1])
    
    print("When threshold is set to: {}".format(threshold))
    print("Total of {} flagged cases".format(num_flagged_cases))
    print("Total of {} flagged cases are true positives".format(num_true_positives))
    print("True Positive Rate: {}".format(num_true_positives/float(num_flagged_cases)))
    print("Recall: {}".format(num_true_positives/float(num_all_positives)))
    print("Precision: {}".format(num_true_positives/float(num_flagged_cases)))

## Epilogue
-----------

In this notebook, we have showed how to configure the basic training,
deployment, and usage of the Amazon SageMaker IP Insights algorithm. All
SageMaker algorithms come with support for two additional services that
make optimizing and using the algorithm that much easier: Automatic
Model Tuning and Batch Transform service.

Amazon SageMaker Automatic Model Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The results above were based on using the default hyperparameters of the
SageMaker IP Insights algorithm. If we wanted to improve the model’s
performance even more, we can use `Amazon SageMaker Automatic Model
Tuning <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`__
to automate the process of finding the hyperparameters.

Validation Dataset
^^^^^^^^^^^^^^^^^^

Previously, we separated our dataset into a training and test set to
validate the performance of a single IP Insights model. However, when we
do model tuning, we train many IP Insights models in parallel. If we
were to use the same test dataset to select the best model, we bias our
model selection such that we don’t know if we selected the best model in
general, or just the best model for that particular dateaset.

Therefore, we need to separate our test set into a validation dataset
and a test dataset. The validation dataset is used for model selection.
Then once we pick the model with the best performance, we evaluate it
the winner on a test set just as before.

Validation Metrics
^^^^^^^^^^^^^^^^^^

For SageMaker Automatic Model Tuning to work, we need an objective
metric which determines the performance of the model we want to
optimize. Because SageMaker IP Insights is an usupervised algorithm, we
do not have a clearly defined metric for performance (such as percentage
of fraudulent events discovered).

We allow the user to provide a validation set of sample data (same
format as training data bove) through the ``validation`` channel. We
then fix the negative sampling strategy to use
``random_negative_sampling_rate=1`` and
``shuffled_negative_sampling_rate=0`` and generate a validation dataset
by assigning corresponding labels to the real and simulated data. We
then calculate the model’s ``descriminator_auc`` metric. We do this by
taking the model’s predicted labels and the ‘true’ simulated labels and
compute the Area Under ROC Curve (AUC) on the model’s performance.

We set up the ``HyperParameterTuner`` to maximize the
``discriminator_auc`` on the validation dataset. We also need to set the
search space for the hyperparameters. We give recommended ranges for the
hyperparmaeters in the `Amazon SageMaker IP Insights
(Hyperparameters) <https://docs.aws.amazon.com/sagemaker/latest/dg/ip-insights-hyperparameters.html>`__
documentation.

.. code:: ipython3

    test_df['timestamp'].describe()

The test set we constructed above spans 3 days. We reserve the first day
as the validation set and the subsequent two days for the test set.

.. code:: ipython3

    time_partition = datetime(2018, 11, 13, tzinfo=pytz.FixedOffset(0)) if num_time_zones > 1 else datetime(2018, 11, 13)
    
    validation_df = test_df[test_df['timestamp'] < time_partition]
    test_df = test_df[test_df['timestamp'] >= time_partition]
    
    valid_data = validation_df.to_csv(index=False, header=False, columns=['user', 'ip_address'])

We then upload the validation data to S3 and specify it as the
validation channel.

.. code:: ipython3

    # Upload data to S3 key
    validation_data_file = 'valid.csv'
    key = os.path.join(prefix, 'validation', validation_data_file)
    boto3.resource('s3').Bucket(bucket).Object(key).put(Body=valid_data)
    s3_valid_data = 's3://{}/{}'.format(bucket, key)
    
    print('Validation data has been uploaded to: {}'.format(s3_valid_data))
    
    # Configure SageMaker IP Insights Input Channels
    input_data = {
        'train': s3_train_data,
        'validation': s3_valid_data
    }

.. code:: ipython3

    from sagemaker.tuner import HyperparameterTuner, IntegerParameter
    
    # Configure HyperparameterTuner
    ip_insights_tuner = HyperparameterTuner(
        estimator=ip_insights,  # previously-configured Estimator object
        objective_metric_name='validation:discriminator_auc',
        hyperparameter_ranges={'vector_dim': IntegerParameter(64, 1024)},
        max_jobs=4,
        max_parallel_jobs=2)
    
    # Start hyperparameter tuning job
    ip_insights_tuner.fit(input_data, include_cls_metadata=False)

.. code:: ipython3

    # Wait for all the jobs to finish
    ip_insights_tuner.wait()
    
    # Visualize training job results
    ip_insights_tuner.analytics().dataframe()

.. code:: ipython3

    # Deploy best model
    tuned_predictor = ip_insights_tuner.deploy(
        initial_instance_count=1, 
        instance_type='ml.m4.xlarge',
        content_type='text/csv',
        serializer=csv_serializer,
        accept='application/json',
        deserializer=json_deserializer
    )

.. code:: ipython3

    # Make a prediction against the SageMaker endpoint
    tuned_predictor.predict(inference_data)

We should have the best performing model from the training job! Now we
can determine thresholds and make predictions just like we did with the
inference endpoint `above <#Inference>`__.

Batch Transform
~~~~~~~~~~~~~~~

Let’s say we want to score all of the login events at the end of the day
and aggregate flagged cases for investigators to look at in the morning.
If we store the daily login events in S3, we can use IP Insights with
`Amazon SageMaker Batch
Transform <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html>`__
to run inference and store the IP Insights scores back in S3 for future
analysis.

Below, we take the training job from before and evaluate it on the
validation data we put in S3.

.. code:: ipython3

    transformer = ip_insights.transformer(
        instance_count=1,
        instance_type='ml.m4.xlarge',
    )
    
    transformer.transform(
        s3_valid_data,
        content_type='text/csv',
        split_type='Line'
    )

.. code:: ipython3

    # Wait for Transform Job to finish
    transformer.wait()

.. code:: ipython3

    print("Batch Transform output is at: {}".format(transformer.output_path))

Stop and Delete the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are done with this model, then we should delete the endpoint
before we close the notebook. Or else you will continue to pay for the
endpoint while it is running.

To do so execute the cell below. Alternately, you can navigate to the
“Endpoints” tab in the SageMaker console, select the endpoint with the
name stored in the variable endpoint_name, and select “Delete” from the
“Actions” dropdown menu.

.. code:: ipython3

    ip_insights_tuner.delete_endpoint()
    sagemaker.Session().delete_endpoint(predictor.endpoint)
