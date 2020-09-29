Goal: Train a model using AutoML functionality!
===============================================

A popular approach to solve a machine learning problem is to try
multiple approaches for training a model by running multiple algorithms
on a dataset. Based on initial analysis, you can decide which algorithm
to use for training and tuning the actual model. However, each algorithm
can have specific feature requirements such as data must be numeric,
missing values must be addressed before the training, etc. Performing
algorithm specific feature engineering tasks can take time. Such a
project can be shortened by running an AutoML algorithm that performs
feature engineering tasks such as one-hot encoding, generalization,
addressing missing values, automatically and then trains models using
multiple algorithms in parallel.

This notebook demonstrates how to use such an AutoML algorithm offerd by
`H2O.ai <https://aws.amazon.com/marketplace/seller-profile?id=55552124-d41b-4bad-90db-72d427682225>`__
in AWS Marketplace for machine learning. AutoML from H2O.ai trains one
or more of following types of models in parallel: 1. XGBoost GBM
(Gradient Boosting Machine) 2. GLM 3. default Random Forest (DRF) 4.
Extremely Randomized Forest (XRT) 5. Deep Neural Nets

Once these models have been trained, it also creates two stacked
ensemble models: 1. An ensemble model created using all the models. 2.
Best of family ensemble model created using models that performed best
in each class/family.

For more information on how H2O.ai’s AutoML works, see `FAQ section of
H2O.ai’s
documentation. <http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html#faq>`__

Contents:
~~~~~~~~~

-  `Step 1: Subscribe to AutoML algorithm from AWS
   Marketplace <#Step-1:-Subscribe-to-AutoML-algorithm-from-AWS-Marketplace>`__
-  `Step 2: Step 2 : Set up
   environment <#Step-2-:-Set-up-environment>`__
-  `Step 3: Prepare and upload
   data <#Step-3:-Prepare-and-upload-data>`__
-  `Step 4: Train a model <#Step-4:-Train-a-model>`__
-  `Step 5: Deploy the model and perform a real-time
   inference <#Step-5:-Deploy-the-model-and-perform-a-real-time-inference>`__
-  `Step 6: Clean-up <#Step-6:-Clean-up>`__

Compatibility
^^^^^^^^^^^^^

This notebook is compatible only with `H2O-3 Automl
Algorithm <https://aws.amazon.com/marketplace/pp/prodview-vbm2cls5zcnky>`__
from AWS Marketplace and an AWS Marketplace subscription is required to
successfully run this notebook.

Usage instructions
^^^^^^^^^^^^^^^^^^

You can run this notebook one cell at a time (By using Shift+Enter for
running a cell).

.. code:: ipython3

    #Let us install necessary H2O.ai library which you would use to load and inspect the model summary.
    import sys
    !{sys.executable} -m pip install http://h2o-release.s3.amazonaws.com/h2o/rel-wright/10/Python/h2o-3.20.0.10-py2.py3-none-any.whl

.. code:: ipython3

    #Import necessary libraries.
    import boto3
    import re
    import os
    import errno
    import base64
    import time
    import numpy as np
    import pandas as pd
    import urllib
    from sagemaker import get_execution_role
    import json
    import uuid
    import sagemaker
    from time import gmtime, strftime
    import urllib.request
    from sagemaker import AlgorithmEstimator


Step 1: Subscribe to AutoML algorithm from AWS Marketplace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open `H2O-3 Automl Algorithm listing from AWS
   Marketplace <https://aws.amazon.com/marketplace/pp/prodview-vbm2cls5zcnky?qid=1557245796960&sr=0-1&ref_=srh_res_product_title>`__
2. Read the **Highlights** section and then **product overview** section
   of the listing.
3. View **usage information** and then **additional resources**.
4. Note the supported instance types and specify the same in the
   following cell.
5. Next, click on **Continue to subscribe**.
6. Review **End user license agreement**, **support terms**, as well as
   **pricing information**.
7. Next, “Accept Offer” button needs to be clicked only if your
   organization agrees with EULA, pricing information as well as support
   terms. Once **Accept offer** button has been clicked, specify
   compatible training and inference types you wish to use.

**Notes**: 1. If **Continue to configuration** button is active, it
means your account already has a subscription to this listing. 2. Once
you click on **Continue to configuration** button and then choose
region, you will see that a product ARN will appear. This is the
algorithm ARN that you need to specify in your training job. However,
for this notebook, the algorithm ARN has been specified in
**src/algorithm_arns.py** file and you do not need to specify the same
explicitly.

.. code:: ipython3

    compatible_training_instance_type='ml.c5.4xlarge' 
    
    compatible_inference_instance_type='ml.c5.2xlarge' 

Step 2 : Set up environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sagemaker as sage
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    
    # Specify S3 prefixes
    common_prefix = "automl-iris"
    training_input_prefix = common_prefix + "/training-input-data"
    training_output_prefix = common_prefix + "/training-output"
    
    #Create session - The session remembers our connection parameters to Amazon SageMaker. We'll use it to perform all of our Amazon SageMaker operations.
    sagemaker_session = sage.Session()


.. code:: ipython3

    #Specify algorithm ARN for H2O.ai's AutoML algorithm from AWS Marketplace.  However, for this notebook, the algorithm ARN 
    #has been specified in src/scikit_product_arns.py file and you do not need to specify the same explicitly.
    
    from src.algorithm_arns import AlgorithmArnProvider
    
    algorithm_arn = AlgorithmArnProvider.get_algorithm_arn(sagemaker_session.boto_region_name)

Next, configure the S3 bucket name.

.. code:: ipython3

    bucket=sagemaker_session.default_bucket()

Next, specify your name to tag resources you create as part of this
experiment.

.. code:: ipython3

    created_by='your_name'

Step 3: Prepare and upload data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have identified the algorithm you want to run, you need to
prepare data that is compatible with your algorithm. This notebook
demonstrates AutoML using the Iris data set (Dua, D. and Graff, C.
(2019). `UCI Machine Learning
Repository <http://archive.ics.uci.edu/ml>`__. Irvine, CA: University of
California, School of Information and Computer Science). Note that we
will be adding a missing value to the first row to demonstrate that
AutoML would take care of missing values.

Background - The Iris dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Iris data
set <https://en.wikipedia.org/wiki/Iris_flower_data_set>`__ contains 150
rows of data, comprising 50 samples from each of three related Iris
species: *Iris setosa*, *Iris virginica*, and *Iris versicolor*.

|Petal geometry compared for three iris species: Iris setosa, Iris
virginica, and Iris versicolor| **From left to right,**\ `Iris
setosa <https://commons.wikimedia.org/w/index.php?curid=170298>`__\ **(by**\ `Radomil <https://commons.wikimedia.org/wiki/User:Radomil>`__\ **,
CC BY-SA 3.0),**\ `Iris
versicolor <https://commons.wikimedia.org/w/index.php?curid=248095>`__\ **(by**\ `Dlanglois <https://commons.wikimedia.org/wiki/User:Dlanglois>`__\ **,
CC BY-SA 3.0), and**\ `Iris
virginica <https://www.flickr.com/photos/33397993@N05/3352169862>`__\ **(by**\ `Frank
Mayfield <https://www.flickr.com/photos/33397993@N05>`__\ **, CC BY-SA
2.0).**

Each row contains the following data for each flower sample:
`sepal <https://en.wikipedia.org/wiki/Sepal>`__ length, sepal width,
`petal <https://en.wikipedia.org/wiki/Petal>`__ length, petal width, and
flower species.

+--------------+-------------+--------------+-------------+------------+
| Sepal Length | Sepal Width | Petal Length | Petal Width | Species    |
+==============+=============+==============+=============+============+
| 5.1          | 3.5         | 1.4          | 0.2         | setosa     |
+--------------+-------------+--------------+-------------+------------+
| 4.9          | 3.0         | 1.4          | 0.2         | setosa     |
+--------------+-------------+--------------+-------------+------------+
| 4.7          | 3.2         | 1.3          | 0.2         | setosa     |
+--------------+-------------+--------------+-------------+------------+
| …            | …           | …            | …           | …          |
+--------------+-------------+--------------+-------------+------------+
| 7.0          | 3.2         | 4.7          | 1.4         | versicolor |
+--------------+-------------+--------------+-------------+------------+
| 6.4          | 3.2         | 4.5          | 1.5         | versicolor |
+--------------+-------------+--------------+-------------+------------+
| 6.9          | 3.1         | 4.9          | 1.5         | versicolor |
+--------------+-------------+--------------+-------------+------------+
| …            | …           | …            | …           | …          |
+--------------+-------------+--------------+-------------+------------+
| 6.5          | 3.0         | 5.2          | 2.0         | virginica  |
+--------------+-------------+--------------+-------------+------------+
| 6.2          | 3.4         | 5.4          | 2.3         | virginica  |
+--------------+-------------+--------------+-------------+------------+
| 5.9          | 3.0         | 5.1          | 1.8         | virginica  |
+--------------+-------------+--------------+-------------+------------+

.. |Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor| image:: https://www.tensorflow.org/images/iris_three_species.jpg

.. code:: ipython3

    %%time
    training_data_location='data/training/iris.csv'
    urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',training_data_location)

Let us look at the sample training data

.. code:: ipython3

    !head $training_data_location

Let us add a header and a copy of first line to demonstrate that the
AutoML listing takes care of missing values as well.

.. code:: ipython3

    !sed -i '1s/^/sepal_length,sepal_width,petal_length,petal_width,species\n,,1.4,0.2,Iris-setosa\n/' $training_data_location

.. code:: ipython3

    !head $training_data_location

When training large models with huge amounts of data, you’ll typically
use big data tools, like Amazon Athena, AWS Glue, or Amazon EMR, to
create your data in S3. For the purposes of this example, we’re using
the classic `Iris
dataset <https://en.wikipedia.org/wiki/Iris_flower_data_set>`__, which
the notebook downloads from the source.

We can use use the tools provided by the Amazon SageMaker Python SDK to
upload the data to an S3 bucket.

.. code:: ipython3

    training_input = sagemaker_session.upload_data(training_data_location, bucket, key_prefix=training_input_prefix)
    print ("Training Data Location " + training_input)

Step 4: Train a model
~~~~~~~~~~~~~~~~~~~~~

Next, let us train a model.

.. code:: ipython3

    algo = AlgorithmEstimator(algorithm_arn=algorithm_arn, 
                              role=role, 
                              train_instance_count=1, 
                              train_instance_type=compatible_training_instance_type, 
                              sagemaker_session=sagemaker_session, 
                              output_path='s3://{}/{}/'.format(bucket,training_output_prefix), 
                              base_job_name='automl',
                              hyperparameters={"max_models": "30",
                              "training": "{'classification': 'true', 'target': 'species'}"},
                              tags=[{"Key":"created_by","Value":created_by}]) 
    
    # Note: Apart from classification and target variables, you can also specify following additional parameter to
    # indicate categorical columns.
    #'categorical_columns': '<comma>,<separated>,<list>' 
    
    algo.fit({'training': training_input}) 

Review the leaderboard available in the log to understand how each of
the top 10 models performed. By default, the metrics are based on 5-fold
cross validation.

Step 5: Deploy the model and perform a real-time inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%time
    
    from sagemaker.predictor import csv_serializer
    predictor = algo.deploy(1, compatible_inference_instance_type, serializer=csv_serializer)

Let us view a sample from original training data and create a sample
payload based on one of the entries.

.. code:: ipython3

    !tail $training_data_location

Let us pick a row, modify values slightly, and then perform an
inference.

.. code:: ipython3

    payload="sepal_length,sepal_width,petal_length,petal_width"+"\n"+"6.0,3.1,5.2,1.9"

Now that data has been prepared, let us perform a real-time inference.

.. code:: ipython3

    print(predictor.predict(payload).decode('utf-8'))

**Congratulations!**, you have successfully performed a real-time
inference on the model you trained using H2O.ai’s AutoML algorithm!
Check whether it predicted the correct class.

Step 6: Clean-up
~~~~~~~~~~~~~~~~

Once you have finished performing predictions, you can delete the
endpoint to avoid getting charged for the same.

.. code:: ipython3

    algo.delete_endpoint()

.. code:: ipython3

    #Finally, delete the model you created.
    predictor.delete_model()


Finally, if the AWS Marketplace subscription was created just for the
experiment and you would like to unsubscribe to the product, here are
the steps that can be followed. Before you cancel the subscription,
ensure that you do not have any `deployable
model <https://console.aws.amazon.com/sagemaker/home#/models>`__ created
from the model-package or using the algorithm. Note - You can find this
by looking at container associated with the model.

Steps to un-subscribe to product from AWS Marketplace: 1. Navigate to
**Machine Learning** tab on `Your Software subscriptions
page <https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=lbr_tab_ml>`__
2. Locate the listing that you would need to cancel subscription for,
and then **Cancel Subscription** can be clicked to cancel the
subscription.

This notebook demonstrated how to perform AutoML with Amazon Sagemaker
using H2O.ai’s AutoML listing from AWS Marketplace.
