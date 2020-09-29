AutoGluon-Tabular in AWS Marketplace
====================================

`AutoGluon <https://github.com/awslabs/autogluon>`__ automates machine
learning tasks enabling you to easily achieve strong predictive
performance in your applications. With just a few lines of code, you can
train and deploy high-accuracy deep learning models on tabular, image,
and text data. This notebook shows how to use AutoGluon-Tabular in AWS
Marketplace.

Contents:
~~~~~~~~~

-  `Step 1: Subscribe to AutoML algorithm from AWS
   Marketplace <#Step-1:-Subscribe-to-AutoML-algorithm-from-AWS-Marketplace>`__
-  `Step 2: Set up environment <#Step-2-:-Set-up-environment>`__
-  `Step 3: Prepare and upload
   data <#Step-3:-Prepare-and-upload-data>`__
-  `Step 4: Train a model <#Step-4:-Train-a-model>`__
-  `Step 5: Deploy the model and perform a real-time
   inference <#Step-5:-Deploy-the-model-and-perform-a-real-time-inference>`__
-  `Step 6: Use Batch Transform <#Step-6:-Use-Batch-Transform>`__
-  `Step 7: Clean-up <#Step-7:-Clean-up>`__

Step 1: Subscribe to AutoML algorithm from AWS Marketplace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open `AutoGluon-Tabular listing from AWS
   Marketplace <https://aws.amazon.com/marketplace/pp/prodview-n4zf5pmjt7ism>`__
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

Step 2 : Set up environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Import necessary libraries.
    import os
    import boto3
    import sagemaker
    from time import sleep
    from collections import Counter
    import numpy as np
    import pandas as pd
    from sagemaker import get_execution_role, local, Model, utils, fw_utils, s3
    from sagemaker import AlgorithmEstimator
    from sagemaker.predictor import RealTimePredictor, csv_serializer, StringDeserializer
    from sklearn.metrics import accuracy_score, classification_report
    from IPython.core.display import display, HTML
    from IPython.core.interactiveshell import InteractiveShell
    
    # Print settings
    InteractiveShell.ast_node_interactivity = "all"
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 10)
    
    # Account/s3 setup
    session = sagemaker.Session()
    bucket = session.default_bucket()
    prefix = 'sagemaker/autogluon-tabular'
    region = session.boto_region_name
    role = get_execution_role()


.. code:: ipython3

    compatible_training_instance_type='ml.m5.4xlarge' 
    compatible_inference_instance_type='ml.m5.4xlarge' 

.. code:: ipython3

    #Specify algorithm ARN for AutoGluon-Tabular from AWS Marketplace.  However, for this notebook, the algorithm ARN 
    #has been specified in src/algorithm_arns.py file and you do not need to specify the same explicitly.
    
    from src.algorithm_arns import AlgorithmArnProvider
    
    algorithm_arn = AlgorithmArnProvider.get_algorithm_arn(region)

Step 3: Get the data
~~~~~~~~~~~~~~~~~~~~

| In this example we’ll use the direct-marketing dataset to build a
  binary classification model that predicts whether customers will
  accept or decline a marketing offer.
| First we’ll download the data and split it into train and test sets.
  AutoGluon does not require a separate validation set (it uses bagged
  k-fold cross-validation).

.. code:: ipython3

    # Download and unzip the data
    !aws s3 cp --region {region} s3://sagemaker-sample-data-{region}/autopilot/direct_marketing/bank-additional.zip .
    !unzip -qq -o bank-additional.zip
    !rm bank-additional.zip
    
    local_data_path = './bank-additional/bank-additional-full.csv'
    data = pd.read_csv(local_data_path)
    
    # Split train/test data
    train = data.sample(frac=0.7, random_state=42)
    test = data.drop(train.index)
    
    # Split test X/y
    label = 'y'
    y_test = test[label]
    X_test = test.drop(columns=[label])

Check the data
''''''''''''''

.. code:: ipython3

    train.head(3)
    train.shape
    
    test.head(3)
    test.shape
    
    X_test.head(3)
    X_test.shape

Upload the data to s3

.. code:: ipython3

    train_file = 'train.csv'
    train.to_csv(train_file,index=False)
    train_s3_path = session.upload_data(train_file, key_prefix='{}/data'.format(prefix))
    
    test_file = 'test.csv'
    test.to_csv(test_file,index=False)
    test_s3_path = session.upload_data(test_file, key_prefix='{}/data'.format(prefix))
    
    X_test_file = 'X_test.csv'
    X_test.to_csv(X_test_file,index=False)
    X_test_s3_path = session.upload_data(X_test_file, key_prefix='{}/data'.format(prefix))

Step 4: Train a model
~~~~~~~~~~~~~~~~~~~~~

Next, let us train a model.

**Note:** Depending on how many underlying models are trained,
``train_volume_size`` may need to be increased so that they all fit on
disk.

.. code:: ipython3

    # Define required label and optional additional parameters
    fit_args = {
      'label': 'y',
      # Adding 'best_quality' to presets list will result in better performance (but longer runtime)
      'presets': ['optimize_for_deployment'],
    }
    
    # Pass fit_args to SageMaker estimator hyperparameters
    hyperparameters = {
      'fit_args': fit_args,
      'feature_importance': True
    }

.. code:: ipython3

    algo = AlgorithmEstimator(algorithm_arn=algorithm_arn, 
                              role=role, 
                              train_instance_count=1, 
                              train_instance_type=compatible_training_instance_type, 
                              sagemaker_session=session, 
                              base_job_name='autogluon',
                              hyperparameters=hyperparameters,
                              train_volume_size=100) 
    
    inputs = {'training': train_s3_path}
    
    algo.fit(inputs)

Step 5: Deploy the model and perform a real-time inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deploy a remote endpoint
''''''''''''''''''''''''

.. code:: ipython3

    %%time
    
    from sagemaker.predictor import csv_serializer
    predictor = algo.deploy(1, 
                            compatible_inference_instance_type, 
                            content_type='text/csv', 
                            serializer=csv_serializer, 
                            deserializer=StringDeserializer())

Predict on unlabeled test data
''''''''''''''''''''''''''''''

.. code:: ipython3

    results = predictor.predict(X_test.to_csv(index=False)).splitlines()
    
    # Check output
    print(Counter(results))

Predict on data that includes label column
''''''''''''''''''''''''''''''''''''''''''

Prediction performance metrics will be printed to endpoint logs.

.. code:: ipython3

    results = predictor.predict(test.to_csv(index=False)).splitlines()
    
    # Check output
    print(Counter(results))

Check that classification performance metrics match evaluation printed to endpoint logs as expected
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: ipython3

    y_results = np.array(results)
    
    print("accuracy: {}".format(accuracy_score(y_true=y_test, y_pred=y_results)))
    print(classification_report(y_true=y_test, y_pred=y_results, digits=6))

Step 6: Use Batch Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By including the label column in the test data, you can also evaluate
prediction performance (In this case, passing ``test_s3_path`` instead
of ``X_test_s3_path``).

.. code:: ipython3

    output_path = f's3://{bucket}/{prefix}/output/'
    
    transformer = algo.transformer(instance_count=1, 
                                   instance_type=compatible_inference_instance_type,
                                   strategy='MultiRecord',
                                   max_payload=6,
                                   max_concurrent_transforms=1,                              
                                   output_path=output_path)
    
    transformer.transform(test_s3_path, content_type='text/csv', split_type='Line')
    transformer.wait()

Step 7: Clean-up
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
