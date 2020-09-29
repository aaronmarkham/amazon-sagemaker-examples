Customer Churn Prediction with Amazon SageMaker Autopilot
=========================================================

**Using AutoPilot to Predict Mobile Customer Departure**

--------------

--------------

Kernel ``Python 3 (Data Science)`` works well with this notebook.

Contents
--------

1. `Introduction <#Introduction>`__
2. `Setup <#Setup>`__
3. `Data <#Data>`__
4. `Train <#Settingup>`__
5. `Autopilot Results <#Results>`__
6. `Host <#Host>`__
7. `Cleanup <#Cleanup>`__

--------------

Introduction
------------

Amazon SageMaker Autopilot is an automated machine learning (commonly
referred to as AutoML) solution for tabular datasets. You can use
SageMaker Autopilot in different ways: on autopilot (hence the name) or
with human guidance, without code through SageMaker Studio, or using the
AWS SDKs. This notebook, as a first glimpse, will use the AWS SDKs to
simply create and deploy a machine learning model.

Losing customers is costly for any business. Identifying unhappy
customers early on gives you a chance to offer them incentives to stay.
This notebook describes using machine learning (ML) for the automated
identification of unhappy customers, also known as customer churn
prediction. ML models rarely give perfect predictions though, so this
notebook is also about how to incorporate the relative costs of
prediction mistakes when determining the financial outcome of using ML.

We use an example of churn that is familiar to all of us–leaving a
mobile phone operator. Seems like I can always find fault with my
provider du jour! And if my provider knows that I’m thinking of leaving,
it can offer timely incentives–I can always use a phone upgrade or
perhaps have a new feature activated–and I might just stick around.
Incentives are often much more cost effective than losing and
reacquiring a customer.

--------------

Setup
-----

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

    import sagemaker
    import boto3
    from sagemaker import get_execution_role
    
    region = boto3.Session().region_name
    
    session = sagemaker.Session()
    
    # You can modify the following to use a bucket of your choosing
    bucket = session.default_bucket()
    prefix = 'sagemaker/DEMO-autopilot-churn'
    
    role = get_execution_role()
    
    # This is the client we will use to interact with SageMaker AutoPilot
    sm = boto3.Session().client(service_name='sagemaker',region_name=region)

Next, we’ll import the Python libraries we’ll need for the remainder of
the exercise.

.. code:: ipython3

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import os
    import sys
    import time
    import json
    from IPython.display import display
    from time import strftime, gmtime
    import sagemaker
    from sagemaker.predictor import csv_serializer

--------------

Data
----

Mobile operators have historical records on which customers ultimately
ended up churning and which continued using the service. We can use this
historical information to construct an ML model of one mobile operator’s
churn using a process called training. After training the model, we can
pass the profile information of an arbitrary customer (the same profile
information that we used to train the model) to the model, and have the
model predict whether this customer is going to churn. Of course, we
expect the model to make mistakes–after all, predicting the future is
tricky business! But I’ll also show how to deal with prediction errors.

The dataset we use is publicly available and was mentioned in the book
`Discovering Knowledge in
Data <https://www.amazon.com/dp/0470908742/>`__ by Daniel T. Larose. It
is attributed by the author to the University of California Irvine
Repository of Machine Learning Datasets. Let’s download and read that
dataset in now:

.. code:: ipython3

    !apt-get install unzip
    !wget http://dataminingconsultant.com/DKD2e_data_sets.zip
    !unzip -o DKD2e_data_sets.zip

Upload the dataset to S3
~~~~~~~~~~~~~~~~~~~~~~~~

Before you run Autopilot on the dataset, first perform a check of the
dataset to make sure that it has no obvious errors. The Autopilot
process can take long time, and it’s generally a good practice to
inspect the dataset before you start a job. This particular dataset is
small, so you can inspect it in the notebook instance itself. If you
have a larger dataset that will not fit in a notebook instance memory,
inspect the dataset offline using a big data analytics tool like Apache
Spark. `Deequ <https://github.com/awslabs/deequ>`__ is a library built
on top of Apache Spark that can be helpful for performing checks on
large datasets. Autopilot is capable of handling datasets up to 5 GB.

Read the data into a Pandas data frame and take a look.

.. code:: ipython3

    churn = pd.read_csv('./Data sets/churn.txt')
    pd.set_option('display.max_columns', 500)
    churn

By modern standards, it’s a relatively small dataset, with only 3,333
records, where each record uses 21 attributes to describe the profile of
a customer of an unknown US mobile operator. The attributes are:

-  ``State``: the US state in which the customer resides, indicated by a
   two-letter abbreviation; for example, OH or NJ
-  ``Account Length``: the number of days that this account has been
   active
-  ``Area Code``: the three-digit area code of the corresponding
   customer’s phone number
-  ``Phone``: the remaining seven-digit phone number
-  ``Int’l Plan``: whether the customer has an international calling
   plan: yes/no
-  ``VMail Plan``: whether the customer has a voice mail feature: yes/no
-  ``VMail Message``: presumably the average number of voice mail
   messages per month
-  ``Day Mins``: the total number of calling minutes used during the day
-  ``Day Calls``: the total number of calls placed during the day
-  ``Day Charge``: the billed cost of daytime calls
-  ``Eve Mins, Eve Calls, Eve Charge``: the billed cost for calls placed
   during the evening
-  ``Night Mins``, ``Night Calls``, ``Night Charge``: the billed cost
   for calls placed during nighttime
-  ``Intl Mins``, ``Intl Calls``, ``Intl Charge``: the billed cost for
   international calls
-  ``CustServ Calls``: the number of calls placed to Customer Service
-  ``Churn?``: whether the customer left the service: true/false

The last attribute, ``Churn?``, is known as the target attribute–the
attribute that we want the ML model to predict.

Reserve some data for calling inference on the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Divide the data into training and testing splits. The training split is
used by SageMaker Autopilot. The testing split is reserved to perform
inference using the suggested model.

.. code:: ipython3

    train_data = churn.sample(frac=0.8,random_state=200)
    
    test_data = churn.drop(train_data.index)
    
    test_data_no_target = test_data.drop(columns=['Churn?'])

Now we’ll upload these files to S3.

.. code:: ipython3

    train_file = 'train_data.csv';
    train_data.to_csv(train_file, index=False, header=True)
    train_data_s3_path = session.upload_data(path=train_file, key_prefix=prefix + "/train")
    print('Train data uploaded to: ' + train_data_s3_path)
    
    test_file = 'test_data.csv';
    test_data_no_target.to_csv(test_file, index=False, header=False)
    test_data_s3_path = session.upload_data(path=test_file, key_prefix=prefix + "/test")
    print('Test data uploaded to: ' + test_data_s3_path)

--------------

Setting up the SageMaker Autopilot Job
--------------------------------------

After uploading the dataset to Amazon S3, you can invoke Autopilot to
find the best ML pipeline to train a model on this dataset.

The required inputs for invoking a Autopilot job are: \* Amazon S3
location for input dataset and for all output artifacts \* Name of the
column of the dataset you want to predict (``Churn?`` in this case) \*
An IAM role

Currently Autopilot supports only tabular datasets in CSV format. Either
all files should have a header row, or the first file of the dataset,
when sorted in alphabetical/lexical order by name, is expected to have a
header row.

.. code:: ipython3

    input_data_config = [{
          'DataSource': {
            'S3DataSource': {
              'S3DataType': 'S3Prefix',
              'S3Uri': 's3://{}/{}/train'.format(bucket,prefix)
            }
          },
          'TargetAttributeName': 'Churn?'
        }
      ]
    
    output_data_config = {
        'S3OutputPath': 's3://{}/{}/output'.format(bucket,prefix)
      }

You can also specify the type of problem you want to solve with your
dataset
(``Regression, MulticlassClassification, BinaryClassification``). In
case you are not sure, SageMaker Autopilot will infer the problem type
based on statistics of the target column (the column you want to
predict).

Because the target attribute, ``Churn?``, is binary, our model will be
performing binary prediction, also known as binary classification. In
this example we will let AutoPilot infer the type of problem for us.

You have the option to limit the running time of a SageMaker Autopilot
job by providing either the maximum number of pipeline evaluations or
candidates (one pipeline evaluation is called a ``Candidate`` because it
generates a candidate model) or providing the total time allocated for
the overall Autopilot job. Under default settings, this job takes about
four hours to run. This varies between runs because of the nature of the
exploratory process Autopilot uses to find optimal training parameters.

Launching the SageMaker Autopilot Job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can now launch the Autopilot job by calling the
``create_auto_ml_job`` API. We limit the number of candidates to 20 so
that the job finishes in a few minutes.

.. code:: ipython3

    from time import gmtime, strftime, sleep
    timestamp_suffix = strftime('%d-%H-%M-%S', gmtime())
    
    auto_ml_job_name = 'automl-churn-' + timestamp_suffix
    print('AutoMLJobName: ' + auto_ml_job_name)
    
    sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                          InputDataConfig=input_data_config,
                          OutputDataConfig=output_data_config,
                          AutoMLJobConfig={'CompletionCriteria':
                                           {'MaxCandidates': 20}
                                          },
                          RoleArn=role)

Tracking SageMaker Autopilot job progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker Autopilot job consists of the following high-level steps : \*
Analyzing Data, where the dataset is analyzed and Autopilot comes up
with a list of ML pipelines that should be tried out on the dataset. The
dataset is also split into train and validation sets. \* Feature
Engineering, where Autopilot performs feature transformation on
individual features of the dataset as well as at an aggregate level. \*
Model Tuning, where the top performing pipeline is selected along with
the optimal hyperparameters for the training algorithm (the last stage
of the pipeline).

.. code:: ipython3

    print ('JobStatus - Secondary Status')
    print('------------------------------')
    
    
    describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
    print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
    job_run_status = describe_response['AutoMLJobStatus']
        
    while job_run_status not in ('Failed', 'Completed', 'Stopped'):
        describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
        job_run_status = describe_response['AutoMLJobStatus']
        
        print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
        sleep(30)

--------------

Results
-------

Now use the describe_auto_ml_job API to look up the best candidate
selected by the SageMaker Autopilot job.

.. code:: ipython3

    best_candidate = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)['BestCandidate']
    best_candidate_name = best_candidate['CandidateName']
    print(best_candidate)
    print('\n')
    print("CandidateName: " + best_candidate_name)
    print("FinalAutoMLJobObjectiveMetricName: " + best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName'])
    print("FinalAutoMLJobObjectiveMetricValue: " + str(best_candidate['FinalAutoMLJobObjectiveMetric']['Value']))

Due to some randomness in the algorithms involved, different runs will
provide slightly different results, but accuracy will be around or above
:math:`93\%`, which is a good result.

--------------

Host
----

Now that we’ve trained the algorithm, let’s create a model and deploy it
to a hosted endpoint.

.. code:: ipython3

    timestamp_suffix = strftime('%d-%H-%M-%S', gmtime())
    model_name = best_candidate_name + timestamp_suffix + "-model"
    model_arn = sm.create_model(Containers=best_candidate['InferenceContainers'],
                                ModelName=model_name,
                                ExecutionRoleArn=role)
    
    epc_name = best_candidate_name + timestamp_suffix + "-epc"
    ep_config = sm.create_endpoint_config(EndpointConfigName = epc_name,
                                          ProductionVariants=[{'InstanceType': 'ml.m5.2xlarge',
                                                               'InitialInstanceCount': 1,
                                                               'ModelName': model_name,
                                                               'VariantName': 'main'}])
    
    ep_name = best_candidate_name + timestamp_suffix + "-ep"
    create_endpoint_response = sm.create_endpoint(EndpointName=ep_name,
                                                  EndpointConfigName=epc_name)

.. code:: ipython3

    sm.get_waiter('endpoint_in_service').wait(EndpointName=ep_name)

Evaluate
~~~~~~~~

Now that we have a hosted endpoint running, we can make real-time
predictions from our model very easily, simply by making an http POST
request. But first, we’ll need to setup serializers and deserializers
for passing our ``test_data`` NumPy arrays to the model behind the
endpoint.

.. code:: ipython3

    from io import StringIO
    from sagemaker.predictor import RealTimePredictor
    from sagemaker.content_types import CONTENT_TYPE_CSV
    
    
    predictor = RealTimePredictor(
        endpoint=ep_name,
        sagemaker_session=session,
        content_type=CONTENT_TYPE_CSV,
        accept=CONTENT_TYPE_CSV)
    
    # Remove the target column from the test data
    test_data_inference = test_data.drop('Churn?', axis=1)
    
    # Obtain predictions from SageMaker endpoint
    prediction = predictor.predict(test_data_inference.to_csv(sep=',', header=False, index=False)).decode('utf-8')
    
    # Load prediction in pandas and compare to ground truth
    prediction_df = pd.read_csv(StringIO(prediction), header=None)
    accuracy = (test_data.reset_index()['Churn?'] == prediction_df[0]).sum() / len(test_data_inference)
    print('Accuracy: {}'.format(accuracy))

--------------

Cleanup
-------

The Autopilot job creates many underlying artifacts such as dataset
splits, preprocessing scripts, or preprocessed data, etc. This code,
when un-commented, deletes them. This operation deletes all the
generated models and the auto-generated notebooks as well.

.. code:: ipython3

    #s3 = boto3.resource('s3')
    #s3_bucket = s3.Bucket(bucket)
    
    #job_outputs_prefix = '{}/output/{}'.format(prefix, auto_ml_job_name)
    #s3_bucket.objects.filter(Prefix=job_outputs_prefix).delete()

Finally, we delete the endpoint and associated resources.

.. code:: ipython3

    sm.delete_endpoint(EndpointName=ep_name)
    sm.delete_endpoint_config(EndpointConfigName=epc_name)
    sm.delete_model(ModelName=model_name)
