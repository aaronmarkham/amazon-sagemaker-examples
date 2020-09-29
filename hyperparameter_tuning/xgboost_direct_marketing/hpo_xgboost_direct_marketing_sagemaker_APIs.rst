Direct Marketing with Amazon SageMaker XGBoost and Hyperparameter Tuning
========================================================================

**Supervised Learning with Gradient Boosted Trees: A Binary Prediction
Problem With Unbalanced Classes**

--------------

--------------

Kernel ``Python 3 (Data Science)`` works well with this notebook.

Contents
--------

1. `Background <#Background>`__
2. `Prepration <#Preparation>`__
3. `Data Downloading <#Data_Downloading>`__
4. `Data Transformation <#Data_Transformation>`__
5. `Setup Hyperparameter Tuning <#Setup_Hyperparameter_Tuning>`__
6. `Launch Hyperparameter Tuning <#Launch_Hyperparameter_Tuning>`__
7. `Analyze Hyperparameter Tuning
   Results <#Analyze_Hyperparameter_Tuning_Results>`__
8. `Deploy The Best Model <#Deploy_The_Best_Model>`__

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

This notebook will train a model which can be used to predict if a
customer will enroll for a term deposit at a bank, after one or more
phone calls. Hyperparameter tuning will be used in order to try multiple
hyperparameter settings and produce the best model.

--------------

Preparation
-----------

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as SageMaker training.
-  The IAM role used to give training access to your data. See SageMaker
   documentation for how to create these.

.. code:: ipython3

    import sagemaker
    import boto3
    
    import numpy as np                                # For matrix operations and numerical processing
    import pandas as pd                               # For munging tabular data
    from time import gmtime, strftime                 
    import os 
     
    region = boto3.Session().region_name    
    smclient = boto3.Session().client('sagemaker')
    
    role = sagemaker.get_execution_role()
    
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-hpo-xgboost-dm'

--------------

Data_Downloading
----------------

Let’s start by downloading the `direct marketing
dataset <https://archive.ics.uci.edu/ml/datasets/bank+marketing>`__ from
UCI’s ML Repository.

.. code:: ipython3

    !wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
    !unzip -o bank-additional.zip

Now lets read this into a Pandas data frame and take a look.

.. code:: ipython3

    data = pd.read_csv('./bank-additional/bank-additional-full.csv', sep=';')
    pd.set_option('display.max_columns', 500)     # Make sure we can see all of the columns
    pd.set_option('display.max_rows', 50)         # Keep the output on one page
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

Data_Transformation
-------------------

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
   data at varying grains.

Luckily, some of these aspects have already been handled for us, and the
algorithm we are showcasing tends to do well at handling sparse or oddly
distributed data. Therefore, let’s keep pre-processing simple.

First of all, Many records have the value of “999” for pdays, number of
days that passed by after a client was last contacted. It is very likely
to be a magic number to represent that no contact was made before.
Considering that, we create a new column called “no_previous_contact”,
then grant it value of “1” when pdays is 999 and “0” otherwise.

In the “job” column, there are categories that mean the customer is not
working, e.g., “student”, “retire”, and “unemployed”. Since it is very
likely whether or not a customer is working will affect his/her decision
to enroll in the term deposit, we generate a new column to show whether
the customer is working based on “job” column.

Last but not the least, we convert categorical to numeric, as is
suggested above.

.. code:: ipython3

    data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)                                 # Indicator variable to capture when pdays takes a value of 999
    data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)   # Indicator for individuals not actively employed
    model_data = pd.get_dummies(data)                                                                  # Convert categorical variables to sets of indicators
    model_data

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

We’ll then split the dataset into training (70%), validation (20%), and
test (10%) datasets and convert the datasets to the right format the
algorithm expects. We will use training and validation datasets during
training. Test dataset will be used to evaluate model performance after
it is deployed to an endpoint.

Amazon SageMaker’s XGBoost algorithm expects data in the libSVM or CSV
data format. For this example, we’ll stick to CSV. Note that the first
column must be the target variable and the CSV should not include
headers. Also, notice that although repetitive it’s easiest to do this
after the train|validation|test split rather than before. This avoids
any misalignment issues due to random reordering.

.. code:: ipython3

    train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9*len(model_data))])  
    
    pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
    pd.concat([validation_data['y_yes'], validation_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('validation.csv', index=False, header=False)
    pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)

Now we’ll copy the file to S3 for Amazon SageMaker training to pickup.

.. code:: ipython3

    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')

--------------

Setup_Hyperparameter_Tuning
---------------------------

*Note, with the default setting below, the hyperparameter tuning job can
take about 30 minutes to complete.*

Now that we have prepared the dataset, we are ready to train models.
Before we do that, one thing to note is there are algorithm settings
which are called “hyperparameters” that can dramtically affect the
performance of the trained models. For example, XGBoost algorithm has
dozens of hyperparameters and we need to pick the right values for those
hyperparameters in order to achieve the desired model training results.
Since which hyperparameter setting can lead to the best result depends
on the dataset as well, it is almost impossible to pick the best
hyperparameter setting without searching for it, and a good search
algorithm can search for the best hyperparameter setting in an automated
and effective way.

We will use SageMaker hyperparameter tuning to automate the searching
process effectively. Specifically, we specify a range, or a list of
possible values in the case of categorical hyperparameters, for each of
the hyperparameter that we plan to tune. SageMaker hyperparameter tuning
will automatically launch multiple training jobs with different
hyperparameter settings, evaluate results of those training jobs based
on a predefined “objective metric”, and select the hyperparameter
settings for future attempts based on previous results. For each
hyperparameter tuning job, we will give it a budget (max number of
training jobs) and it will complete once that many training jobs have
been executed.

Now we configure the hyperparameter tuning job by defining a JSON object
that specifies following information: \* The ranges of hyperparameters
we want to tune \* Number of training jobs to run in total and how many
training jobs should be run simultaneously. More parallel jobs will
finish tuning sooner, but may sacrifice accuracy. We recommend you set
the parallel jobs value to less than 10% of the total number of training
jobs (we’ll set it higher just for this example to keep it short). \*
The objective metric that will be used to evaluate training results, in
this example, we select *validation:auc* to be the objective metric and
the goal is to maximize the value throughout the hyperparameter tuning
process. One thing to note is the objective metric has to be among the
metrics that are emitted by the algorithm during training. In this
example, the built-in XGBoost algorithm emits a bunch of metrics and
*validation:auc* is one of them. If you bring your own algorithm to
SageMaker, then you need to make sure whatever objective metric you
select, your algorithm actually emits it.

We will tune four hyperparameters in this examples: \* *eta*: Step size
shrinkage used in updates to prevent overfitting. After each boosting
step, you can directly get the weights of new features. The eta
parameter actually shrinks the feature weights to make the boosting
process more conservative. \* *alpha*: L1 regularization term on
weights. Increasing this value makes models more conservative. \*
*min_child_weight*: Minimum sum of instance weight (hessian) needed in a
child. If the tree partition step results in a leaf node with the sum of
instance weight less than min_child_weight, the building process gives
up further partitioning. In linear regression models, this simply
corresponds to a minimum number of instances needed in each node. The
larger the algorithm, the more conservative it is. \* *max_depth*:
Maximum depth of a tree. Increasing this value makes the model more
complex and likely to be overfitted.

.. code:: ipython3

    from time import gmtime, strftime, sleep
    tuning_job_name = 'xgboost-tuningjob-' + strftime("%d-%H-%M-%S", gmtime())
    
    print (tuning_job_name)
    
    tuning_job_config = {
        "ParameterRanges": {
          "CategoricalParameterRanges": [],
          "ContinuousParameterRanges": [
            {
              "MaxValue": "1",
              "MinValue": "0",
              "Name": "eta",
            },
            {
              "MaxValue": "10",
              "MinValue": "1",
              "Name": "min_child_weight",
            },
            {
              "MaxValue": "2",
              "MinValue": "0",
              "Name": "alpha",            
            }
          ],
          "IntegerParameterRanges": [
            {
              "MaxValue": "10",
              "MinValue": "1",
              "Name": "max_depth",
            }
          ]
        },
        "ResourceLimits": {
          "MaxNumberOfTrainingJobs": 20,
          "MaxParallelTrainingJobs": 3
        },
        "Strategy": "Bayesian",
        "HyperParameterTuningJobObjective": {
          "MetricName": "validation:auc",
          "Type": "Maximize"
        }
      }

Then we configure the training jobs the hyperparameter tuning job will
launch by defining a JSON object that specifies following information:
\* The container image for the algorithm (XGBoost) \* The input
configuration for the training and validation data \* Configuration for
the output of the algorithm \* The values of any algorithm
hyperparameters that are not tuned in the tuning job
(StaticHyperparameters) \* The type and number of instances to use for
the training jobs \* The stopping condition for the training jobs

Again, since we are using built-in XGBoost algorithm here, it emits two
predefined metrics: *validation:auc* and *train:auc*, and we elected to
monitor *validation_auc* as you can see above. One thing to note is if
you bring your own algorithm, your algorithm emits metrics by itself. In
that case, you’ll need to add a MetricDefinition object here to define
the format of those metrics through regex, so that SageMaker knows how
to extract those metrics.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    training_image = get_image_uri(region, 'xgboost', repo_version='latest')
         
    s3_input_train = 's3://{}/{}/train'.format(bucket, prefix)
    s3_input_validation ='s3://{}/{}/validation/'.format(bucket, prefix)
        
    training_job_definition = {
        "AlgorithmSpecification": {
          "TrainingImage": training_image,
          "TrainingInputMode": "File"
        },
        "InputDataConfig": [
          {
            "ChannelName": "train",
            "CompressionType": "None",
            "ContentType": "csv",
            "DataSource": {
              "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": s3_input_train
              }
            }
          },
          {
            "ChannelName": "validation",
            "CompressionType": "None",
            "ContentType": "csv",
            "DataSource": {
              "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": s3_input_validation
              }
            }
          }
        ],
        "OutputDataConfig": {
          "S3OutputPath": "s3://{}/{}/output".format(bucket,prefix)
        },
        "ResourceConfig": {
          "InstanceCount": 1,
          "InstanceType": "ml.m4.xlarge",
          "VolumeSizeInGB": 10
        },
        "RoleArn": role,
        "StaticHyperParameters": {
          "eval_metric": "auc",
          "num_round": "100",
          "objective": "binary:logistic",
          "rate_drop": "0.3",
          "tweedie_variance_power": "1.4"
        },
        "StoppingCondition": {
          "MaxRuntimeInSeconds": 43200
        }
    }


Launch_Hyperparameter_Tuning
----------------------------

Now we can launch a hyperparameter tuning job by calling
create_hyper_parameter_tuning_job API. After the hyperparameter tuning
job is created, we can go to SageMaker console to track the progress of
the hyperparameter tuning job until it is completed.

.. code:: ipython3

    smclient.create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name,
                                                HyperParameterTuningJobConfig = tuning_job_config,
                                                TrainingJobDefinition = training_job_definition)

Let’s just run a quick check of the hyperparameter tuning jobs status to
make sure it started successfully.

.. code:: ipython3

    smclient.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name)['HyperParameterTuningJobStatus']

Analyze tuning job results - after tuning job is completed
----------------------------------------------------------

Please refer to “HPO_Analyze_TuningJob_Results.ipynb” to see example
code to analyze the tuning job results.

Deploy the best model
---------------------

Now that we have got the best model, we can deploy it to an endpoint.
Please refer to other SageMaker sample notebooks or SageMaker
documentation to see how to deploy a model.
