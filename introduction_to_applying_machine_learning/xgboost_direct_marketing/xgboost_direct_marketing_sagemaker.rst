Targeting Direct Marketing with Amazon SageMaker XGBoost
========================================================

**Supervised Learning with Gradient Boosted Trees: A Binary Prediction
Problem With Unbalanced Classes**

--------------

--------------

Contents
--------

1. `Background <#Background>`__
2. `Prepration <#Preparation>`__
3. `Data <#Data>`__

   1. `Exploration <#Exploration>`__
   2. `Transformation <#Transformation>`__

4. `Training <#Training>`__
5. `Hosting <#Hosting>`__
6. `Evaluation <#Evaluation>`__
7. `Exentsions <#Extensions>`__

--------------

Background
----------

Direct marketing, either through mail, email, phone, etc., is a common
tactic to acquire customers. Because resources and a customer’s
attention is limited, the goal is to only target the subset of prospects
who are likely to engage with a specific offer. Predicting those
potential customers based on readily available information like
demographics, past interactions, and environmental factors is a common
machine learning problem.

This notebook presents an example problem to predict if a customer will
enroll for a term deposit at a bank, after one or more phone calls. The
steps include:

-  Preparing your Amazon SageMaker notebook
-  Downloading data from the internet into Amazon SageMaker
-  Investigating and transforming the data so that it can be fed to
   Amazon SageMaker algorithms
-  Estimating a model using the Gradient Boosting algorithm
-  Evaluating the effectiveness of the model
-  Setting the model up to make on-going predictions

--------------

Preparation
-----------

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the boto regexp with a the appropriate full
   IAM role arn string(s).

.. code:: ipython3

    bucket = '<your_s3_bucket_name_here>'
    prefix = 'sagemaker/DEMO-xgboost-dm'
     
    # Define IAM role
    import boto3
    import re
    from sagemaker import get_execution_role
    
    role = get_execution_role()

Now let’s bring in the Python libraries that we’ll use throughout the
analysis

.. code:: ipython3

    import numpy as np                                # For matrix operations and numerical processing
    import pandas as pd                               # For munging tabular data
    import matplotlib.pyplot as plt                   # For charts and visualizations
    from IPython.display import Image                 # For displaying images in the notebook
    from IPython.display import display               # For displaying outputs in the notebook
    from time import gmtime, strftime                 # For labeling SageMaker models, endpoints, etc.
    import sys                                        # For writing outputs to notebook
    import math                                       # For ceiling function
    import json                                       # For parsing hosting outputs
    import os                                         # For manipulating filepath names
    import sagemaker                                  # Amazon SageMaker's Python SDK provides many helper functions
    from sagemaker.predictor import csv_serializer    # Converts strings for HTTP POST requests on inference

--------------

Data
----

Let’s start by downloading the `direct marketing
dataset <https://sagemaker-sample-data-us-west-2.s3-us-west-2.amazonaws.com/autopilot/direct_marketing/bank-additional.zip>`__
from the sample data s3 bucket.

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven
Approach to Predict the Success of Bank Telemarketing. Decision Support
Systems, Elsevier, 62:22-31, June 2014

.. code:: ipython3

    !wget https://sagemaker-sample-data-us-west-2.s3-us-west-2.amazonaws.com/autopilot/direct_marketing/bank-additional.zip
    !apt-get install unzip -y
    !unzip -o bank-additional.zip

Now lets read this into a Pandas data frame and take a look.

.. code:: ipython3

    data = pd.read_csv('./bank-additional/bank-additional-full.csv')
    pd.set_option('display.max_columns', 500)     # Make sure we can see all of the columns
    pd.set_option('display.max_rows', 20)         # Keep the output on one page
    data

Let’s talk about the data. At a high level, we can see:

-  We have a little over 40K customer records, and 20 features for each
   customer
-  The features are mixed; some numeric, some categorical
-  The data appears to be sorted, at least by ``time`` and ``contact``,
   maybe more

**Specifics on each of the features:**

*Demographics:* \* ``age``: Customer’s age (numeric) \* ``job``: Type of
job (categorical: ‘admin.’, ‘services’, …) \* ``marital``: Marital
status (categorical: ‘married’, ‘single’, …) \* ``education``: Level of
education (categorical: ‘basic.4y’, ‘high.school’, …)

*Past customer events:* \* ``default``: Has credit in default?
(categorical: ‘no’, ‘unknown’, …) \* ``housing``: Has housing loan?
(categorical: ‘no’, ‘yes’, …) \* ``loan``: Has personal loan?
(categorical: ‘no’, ‘yes’, …)

*Past direct marketing contacts:* \* ``contact``: Contact communication
type (categorical: ‘cellular’, ‘telephone’, …) \* ``month``: Last
contact month of year (categorical: ‘may’, ‘nov’, …) \* ``day_of_week``:
Last contact day of the week (categorical: ‘mon’, ‘fri’, …) \*
``duration``: Last contact duration, in seconds (numeric). Important
note: If duration = 0 then ``y`` = ‘no’.

*Campaign information:* \* ``campaign``: Number of contacts performed
during this campaign and for this client (numeric, includes last
contact) \* ``pdays``: Number of days that passed by after the client
was last contacted from a previous campaign (numeric) \* ``previous``:
Number of contacts performed before this campaign and for this client
(numeric) \* ``poutcome``: Outcome of the previous marketing campaign
(categorical: ‘nonexistent’,‘success’, …)

*External environment factors:* \* ``emp.var.rate``: Employment
variation rate - quarterly indicator (numeric) \* ``cons.price.idx``:
Consumer price index - monthly indicator (numeric) \* ``cons.conf.idx``:
Consumer confidence index - monthly indicator (numeric) \*
``euribor3m``: Euribor 3 month rate - daily indicator (numeric) \*
``nr.employed``: Number of employees - quarterly indicator (numeric)

*Target variable:* \* ``y``: Has the client subscribed a term deposit?
(binary: ‘yes’,‘no’)

Exploration
~~~~~~~~~~~

Let’s start exploring the data. First, let’s understand how the features
are distributed.

.. code:: ipython3

    # Frequency tables for each categorical feature
    for column in data.select_dtypes(include=['object']).columns:
        display(pd.crosstab(index=data[column], columns='% observations', normalize='columns'))
    
    # Histograms for each numeric features
    display(data.describe())
    %matplotlib inline
    hist = data.hist(bins=30, sharey=True, figsize=(10, 10))

Notice that:

-  Almost 90% of the values for our target variable ``y`` are “no”, so
   most customers did not subscribe to a term deposit.
-  Many of the predictive features take on values of “unknown”. Some are
   more common than others. We should think carefully as to what causes
   a value of “unknown” (are these customers non-representative in some
   way?) and how we that should be handled.

   -  Even if “unknown” is included as it’s own distinct category, what
      does it mean given that, in reality, those observations likely
      fall within one of the other categories of that feature?

-  Many of the predictive features have categories with very few
   observations in them. If we find a small category to be highly
   predictive of our target outcome, do we have enough evidence to make
   a generalization about that?
-  Contact timing is particularly skewed. Almost a third in May and less
   than 1% in December. What does this mean for predicting our target
   variable next December?
-  There are no missing values in our numeric features. Or missing
   values have already been imputed.

   -  ``pdays`` takes a value near 1000 for almost all customers. Likely
      a placeholder value signifying no previous contact.

-  Several numeric features have a very long tail. Do we need to handle
   these few observations with extremely large values differently?
-  Several numeric features (particularly the macroeconomic ones) occur
   in distinct buckets. Should these be treated as categorical?

Next, let’s look at how our features relate to the target that we are
attempting to predict.

.. code:: ipython3

    for column in data.select_dtypes(include=['object']).columns:
        if column != 'y':
            display(pd.crosstab(index=data[column], columns=data['y'], normalize='columns'))
    
    for column in data.select_dtypes(exclude=['object']).columns:
        print(column)
        hist = data[[column, 'y']].hist(by='y', bins=30)
        plt.show()

Notice that:

-  Customers who are– “blue-collar”, “married”, “unknown” default
   status, contacted by “telephone”, and/or in “may” are a substantially
   lower portion of “yes” than “no” for subscribing.
-  Distributions for numeric variables are different across “yes” and
   “no” subscribing groups, but the relationships may not be
   straightforward or obvious.

Now let’s look at how our features relate to one another.

.. code:: ipython3

    display(data.corr())
    pd.plotting.scatter_matrix(data, figsize=(12, 12))
    plt.show()

Notice that: \* Features vary widely in their relationship with one
another. Some with highly negative correlation, others with highly
positive correlation. \* Relationships between features is non-linear
and discrete in many cases.

Transformation
~~~~~~~~~~~~~~

Cleaning up data is part of nearly every machine learning project. It
arguably presents the biggest risk if done incorrectly and is one of the
more subjective aspects in the process. Several common techniques
include:

-  Handling missing values: Some machine learning algorithms are capable
   of handling missing values, but most would rather not. Options
   include:
-  Removing observations with missing values: This works well if only a
   very small fraction of observations have incomplete information.
-  Removing features with missing values: This works well if there are a
   small number of features which have a large number of missing values.
-  Imputing missing values: Entire
   `books <https://www.amazon.com/Flexible-Imputation-Missing-Interdisciplinary-Statistics/dp/1439868247>`__
   have been written on this topic, but common choices are replacing the
   missing value with the mode or mean of that column’s non-missing
   values.
-  Converting categorical to numeric: The most common method is one hot
   encoding, which for each feature maps every distinct value of that
   column to its own feature which takes a value of 1 when the
   categorical feature is equal to that value, and 0 otherwise.
-  Oddly distributed data: Although for non-linear models like Gradient
   Boosted Trees, this has very limited implications, parametric models
   like regression can produce wildly inaccurate estimates when fed
   highly skewed data. In some cases, simply taking the natural log of
   the features is sufficient to produce more normally distributed data.
   In others, bucketing values into discrete ranges is helpful. These
   buckets can then be treated as categorical variables and included in
   the model when one hot encoded.
-  Handling more complicated data types: Mainpulating images, text, or
   data at varying grains is left for other notebook templates.

Luckily, some of these aspects have already been handled for us, and the
algorithm we are showcasing tends to do well at handling sparse or oddly
distributed data. Therefore, let’s keep pre-processing simple.

.. code:: ipython3

    data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)                                 # Indicator variable to capture when pdays takes a value of 999
    data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)   # Indicator for individuals not actively employed
    model_data = pd.get_dummies(data)                                                                  # Convert categorical variables to sets of indicators

Another question to ask yourself before building a model is whether
certain features will add value in your final use case. For example, if
your goal is to deliver the best prediction, then will you have access
to that data at the moment of prediction? Knowing it’s raining is highly
predictive for umbrella sales, but forecasting weather far enough out to
plan inventory on umbrellas is probably just as difficult as forecasting
umbrella sales without knowledge of the weather. So, including this in
your model may give you a false sense of precision.

Following this logic, let’s remove the economic features and
``duration`` from our data as they would need to be forecasted with high
precision to use as inputs in future predictions.

Even if we were to use values of the economic indicators from the
previous quarter, this value is likely not as relevant for prospects
contacted early in the next quarter as those contacted later on.

.. code:: ipython3

    model_data = model_data.drop(['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)

When building a model whose primary goal is to predict a target value on
new data, it is important to understand overfitting. Supervised learning
models are designed to minimize error between their predictions of the
target value and actuals, in the data they are given. This last part is
key, as frequently in their quest for greater accuracy, machine learning
models bias themselves toward picking up on minor idiosyncrasies within
the data they are shown. These idiosyncrasies then don’t repeat
themselves in subsequent data, meaning those predictions can actually be
made less accurate, at the expense of more accurate predictions in the
training phase.

The most common way of preventing this is to build models with the
concept that a model shouldn’t only be judged on its fit to the data it
was trained on, but also on “new” data. There are several different ways
of operationalizing this, holdout validation, cross-validation,
leave-one-out validation, etc. For our purposes, we’ll simply randomly
split the data into 3 uneven groups. The model will be trained on 70% of
data, it will then be evaluated on 20% of data to give us an estimate of
the accuracy we hope to have on “new” data, and 10% will be held back as
a final testing dataset which will be used later on.

.. code:: ipython3

    train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])   # Randomly sort the data then split out first 70%, second 20%, and last 10%

Amazon SageMaker’s XGBoost container expects data in the libSVM or CSV
data format. For this example, we’ll stick to CSV. Note that the first
column must be the target variable and the CSV should not include
headers. Also, notice that although repetitive it’s easiest to do this
after the train|validation|test split rather than before. This avoids
any misalignment issues due to random reordering.

.. code:: ipython3

    pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
    pd.concat([validation_data['y_yes'], validation_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)

Now we’ll copy the file to S3 for Amazon SageMaker’s managed training to
pickup.

.. code:: ipython3

    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')

--------------

Training
--------

Now we know that most of our features have skewed distributions, some
are highly correlated with one another, and some appear to have
non-linear relationships with our target variable. Also, for targeting
future prospects, good predictive accuracy is preferred to being able to
explain why that prospect was targeted. Taken together, these aspects
make gradient boosted trees a good candidate algorithm.

There are several intricacies to understanding the algorithm, but at a
high level, gradient boosted trees works by combining predictions from
many simple models, each of which tries to address the weaknesses of the
previous models. By doing this the collection of simple models can
actually outperform large, complex models. Other Amazon SageMaker
notebooks elaborate on gradient boosting trees further and how they
differ from similar algorithms.

``xgboost`` is an extremely popular, open-source package for gradient
boosted trees. It is computationally powerful, fully featured, and has
been successfully used in many machine learning competitions. Let’s
start with a simple ``xgboost`` model, trained using Amazon SageMaker’s
managed, distributed training framework.

First we’ll need to specify the ECR container location for Amazon
SageMaker’s implementation of XGBoost.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'xgboost')

Then, because we’re training with the CSV file format, we’ll create
``s3_input``\ s that our training function can use as a pointer to the
files in S3, which also specify that the content type is CSV.

.. code:: ipython3

    s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket, prefix), content_type='csv')
    s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/validation/'.format(bucket, prefix), content_type='csv')

First we’ll need to specify training parameters to the estimator. This
includes: 1. The ``xgboost`` algorithm container 1. The IAM role to use
1. Training instance type and count 1. S3 location for output data 1.
Algorithm hyperparameters

And then a ``.fit()`` function which specifies: 1. S3 location for
output data. In this case we have both a training and validation set
which are passed in.

.. code:: ipython3

    sess = sagemaker.Session()
    
    xgb = sagemaker.estimator.Estimator(container,
                                        role, 
                                        train_instance_count=1, 
                                        train_instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/output'.format(bucket, prefix),
                                        sagemaker_session=sess)
    xgb.set_hyperparameters(max_depth=5,
                            eta=0.2,
                            gamma=4,
                            min_child_weight=6,
                            subsample=0.8,
                            silent=0,
                            objective='binary:logistic',
                            num_round=100)
    
    xgb.fit({'train': s3_input_train, 'validation': s3_input_validation}) 

--------------

Hosting
-------

Now that we’ve trained the ``xgboost`` algorithm on our data, let’s
deploy a model that’s hosted behind a real-time endpoint.

.. code:: ipython3

    xgb_predictor = xgb.deploy(initial_instance_count=1,
                               instance_type='ml.m4.xlarge')

--------------

Evaluation
----------

There are many ways to compare the performance of a machine learning
model, but let’s start by simply comparing actual to predicted values.
In this case, we’re simply predicting whether the customer subscribed to
a term deposit (``1``) or not (``0``), which produces a simple confusion
matrix.

First we’ll need to determine how we pass data into and receive data
from our endpoint. Our data is currently stored as NumPy arrays in
memory of our notebook instance. To send it in an HTTP POST request,
we’ll serialize it as a CSV string and then decode the resulting CSV.

*Note: For inference with CSV format, SageMaker XGBoost requires that
the data does NOT include the target variable.*

.. code:: ipython3

    xgb_predictor.content_type = 'text/csv'
    xgb_predictor.serializer = csv_serializer

Now, we’ll use a simple function to: 1. Loop over our test dataset 1.
Split it into mini-batches of rows 1. Convert those mini-batches to CSV
string payloads (notice, we drop the target variable from our dataset
first) 1. Retrieve mini-batch predictions by invoking the XGBoost
endpoint 1. Collect predictions and convert from the CSV output our
model provides into a NumPy array

.. code:: ipython3

    def predict(data, rows=500):
        split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
        predictions = ''
        for array in split_array:
            predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])
    
        return np.fromstring(predictions[1:], sep=',')
    
    predictions = predict(test_data.drop(['y_no', 'y_yes'], axis=1).to_numpy())

Now we’ll check our confusion matrix to see how well we predicted versus
actuals.

.. code:: ipython3

    pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions), rownames=['actuals'], colnames=['predictions'])

So, of the ~4000 potential customers, we predicted 136 would subscribe
and 94 of them actually did. We also had 389 subscribers who subscribed
that we did not predict would. This is less than desirable, but the
model can (and should) be tuned to improve this. Most importantly, note
that with minimal effort, our model produced accuracies similar to those
published
`here <http://media.salford-systems.com/video/tutorial/2015/targeted_marketing.pdf>`__.

*Note that because there is some element of randomness in the
algorithm’s subsample, your results may differ slightly from the text
written above.*

--------------

Extensions
----------

This example analyzed a relatively small dataset, but utilized Amazon
SageMaker features such as distributed, managed training and real-time
model hosting, which could easily be applied to much larger problems. In
order to improve predictive accuracy further, we could tweak value we
threshold our predictions at to alter the mix of false-positives and
false-negatives, or we could explore techniques like hyperparameter
tuning. In a real-world scenario, we would also spend more time
engineering features by hand and would likely look for additional
datasets to include which contain customer information not available in
our initial dataset.

(Optional) Clean-up
~~~~~~~~~~~~~~~~~~~

If you are done with this notebook, please run the cell below. This will
remove the hosted endpoint you created and avoid any charges from a
stray instance being left on.

.. code:: ipython3

    sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
