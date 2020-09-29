Train, tune, and deploy a custom ML model using For Seller to update: Title_of_your_ML Algorithm  Algorithm from AWS Marketplace
-------------------------------------------------------------------------------------------------------------------------------

 For Seller to update: Add overview of the algorithm here

For Seller to update: Add link to the research paper or a detailed
description document of the algorithm here

This sample notebook shows you how to train a custom ML model using For
Seller to
update:\ `Title_of_your_Algorithm <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__\ 
from AWS Marketplace.

Pre-requisites:
^^^^^^^^^^^^^^^

1. **Note**: This notebook contains elements which render correctly in
   Jupyter interface. Open this notebook from an Amazon SageMaker
   Notebook Instance or Amazon SageMaker Studio.
2. Ensure that IAM role used has **AmazonSageMakerFullAccess**
3. Some hands-on experience using `Amazon
   SageMaker <https://aws.amazon.com/sagemaker/>`__.
4. To use this algorithm successfully, ensure that:

   1. Either your IAM role has these three permissions and you have
      authority to make AWS Marketplace subscriptions in the AWS account
      used:

      1. **aws-marketplace:ViewSubscriptions**
      2. **aws-marketplace:Unsubscribe**
      3. **aws-marketplace:Subscribe**

   2. or your AWS account has a subscription to For Seller to
      update:\ `Title_of_your_algorithm <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__\ .

Contents:
^^^^^^^^^

1. `Subscribe to the algorithm <#1.-Subscribe-to-the-algorithm>`__
2. `Prepare dataset <#2.-Prepare-dataset>`__

   1. `Dataset format expected by the
      algorithm <#A.-Dataset-format-expected-by-the-algorithm>`__
   2. `Configure and visualize train and test
      dataset <#B.-Configure-and-visualize-train-and-test-dataset>`__
   3. `Upload datasets to Amazon
      S3 <#C.-Upload-datasets-to-Amazon-S3>`__

3. `Train a machine learning
   model <#3:-Train-a-machine-learning-model>`__

   1. `Set up environment <#3.1-Set-up-environment>`__
   2. `Train a model <#3.2-Train-a-model>`__

4. `Deploy model and verify
   results <#4:-Deploy-model-and-verify-results>`__

   1. `Deploy trained model <#A.-Deploy-trained-model>`__
   2. `Create input payload <#B.-Create-input-payload>`__
   3. `Perform real-time inference <#C.-Perform-real-time-inference>`__
   4. `Visualize output <#D.-Visualize-output>`__
   5. `Calculate relevant metrics <#E.-Calculate-relevant-metrics>`__
   6. `Delete the endpoint <#F.-Delete-the-endpoint>`__

5. `Tune your model! (optional) <#5:-Tune-your-model!-(optional)>`__

   1. `Tuning Guidelines <#A.-Tuning-Guidelines>`__
   2. `Define Tuning configuration <#B.-Define-Tuning-configuration>`__
   3. `Run a model tuning job <#C.-Run-a-model-tuning-job>`__

6. `Perform Batch inference <#6.-Perform-Batch-inference>`__
7. `Clean-up <#7.-Clean-up>`__

   1. `Delete the model <#A.-Delete-the-model>`__
   2. `Unsubscribe to the listing
      (optional) <#B.-Unsubscribe-to-the-listing-(optional)>`__

Usage instructions
^^^^^^^^^^^^^^^^^^

You can run this notebook one cell at a time (By using Shift+Enter for
running a cell).

1. Subscribe to the algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To subscribe to the algorithm: 1. Open the algorithm listing page For
Seller to
update:\ `Title_of_your_product <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__.
1. On the AWS Marketplace listing, click on **Continue to subscribe**
button. 1. On the **Subscribe to this software** page, review and click
on **“Accept Offer”** if you agree with EULA, pricing, and support
terms. 1. Once you click on **Continue to configuration button** and
then choose a **region**, you will see a **Product Arn**. This is the
algorithm ARN that you need to specify while training a custom ML model.
Copy the ARN corresponding to your region and specify the same in the
following cell.

.. code:: ipython3

    algo_arn='<Customer to specify algorithm ARN corresponding to their AWS region>'

2. Prepare dataset
~~~~~~~~~~~~~~~~~~

 For Seller to update: Add all necessary imports in following cell. If
you need specific packages to be installed, try to provide them in this
section, in a separate cell.

.. code:: ipython3

    import base64
    import json 
    import uuid
    from sagemaker import ModelPackage
    import sagemaker as sage
    from sagemaker import get_execution_role
    from sagemaker import ModelPackage
    from urllib.parse import urlparse
    import boto3
    from IPython.display import Image
    from PIL import Image as ImageEdit
    import urllib.request
    import numpy as np

A. Dataset format expected by the algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Seller to update: In following cell, provide a description of the
dataset accepted by the training job. This will help customer understand
what the dataset fed to the algorithm needs to look like.



You can also find more information about dataset format in **Usage
Information** section of For Seller to
update:\ `Title_of_your_product <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__.

B. Configure and visualize train and test dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Seller to update: upload the sample training dataset into data/train
directory and update the ``training_dataset`` parameter value in
following cell. You are strongly recommended to either upload the
dataset into data/train directory or download it from a reliable source
at runtime. **If you intend to download it at run-time, add relevant
code in following cell.** Do not hardcode your bucket name.

.. code:: ipython3

    training_dataset='data/train/<FileName.ext>'

For Seller to update/read: We recommend that you support a test channel
and accept a test dataset to calculate your algorithm metrics on. Emit
both - training as well as test metrics.

For Seller to update: upload a test dataset into data/test directory.
Alternately, you may want to download the test dataset on-the-fly. **If
you intend to download it at run-time, add relevant code in following
cell.** Update the test_dataset parameter value in following cell.

.. code:: ipython3

    test_dataset='data/test/<FileName.ext>'

For Seller to update: Add code that displays a few rows from the
training dataset. Also explain how the training dataset provided as part
of the notebook was created.


C. Upload datasets to Amazon S3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Seller to read: Do not change bucket parameter value. Do not
hardcode your S3 bucket name.

.. code:: ipython3

    sagemaker_session = sage.Session()
    bucket=sagemaker_session.default_bucket()
    bucket

For Seller to update: Update prefix with a unique S3 prefix for your
algorithm.

.. code:: ipython3

    training_data=sagemaker_session.upload_data(test_dataset, bucket=bucket, key_prefix='<For Seller to update:S3 Prefix>')
    test_data=sagemaker_session.upload_data(test_dataset, bucket=bucket, key_prefix='<For Seller to update:S3 Prefix>')

3: Train a machine learning model
---------------------------------

Now that dataset is available in an accessible Amazon S3 bucket, we are
ready to train a machine learning model.

3.1 Set up environment
~~~~~~~~~~~~~~~~~~~~~~

For Seller to update: Initialize required variables in following cell.

.. code:: ipython3

    role = get_execution_role()


For Seller to update: update algorithm sepcific unique prefix in
following cell.

.. code:: ipython3

    output_location = 's3://{}/<For seller to Update:Update a unique prefix>/{}'.format(bucket, 'output')

3.2 Train a model
~~~~~~~~~~~~~~~~~

For Seller to update: Update following cell with appropriate
hyperparameter values to be passed to the training job

You can also find more information about dataset format in
**Hyperparameters** section of For Seller to
update:\ `Title_of_your_product <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__.

.. code:: ipython3

    #Define hyperparameters
    hyperparameters={}

For Seller to update: Update appropriate values in estimator definition
and ensure that fit call works as expected.

For information on creating an ``Estimator`` object, see
`documentation <https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html>`__

.. code:: ipython3

    #Create an estimator object for running a training job
    estimator = sage.algorithm.AlgorithmEstimator(
        algorithm_arn=algo_arn,
        base_job_name="<For Seller to update: Specify base job name>",
        role=role,
        train_instance_count=1,
        train_instance_type='<For Seller to update: Specify an instance-type recommended for training>',
        input_mode="File",
        output_path=output_location,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters
    )
    #Run the training job.
    estimator.fit({"training": training_dataset,"test":test_dataset})

See this
`blog-post <https://aws.amazon.com/blogs/machine-learning/easily-monitor-and-visualize-metrics-while-training-models-on-amazon-sagemaker/>`__
for more information how to visualize metrics during the process. You
can also open the training job from `Amazon SageMaker
console <https://console.aws.amazon.com/sagemaker/home?#/jobs/>`__ and
monitor the metrics/logs in **Monitor** section.

4: Deploy model and verify results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can deploy the model for performing real-time inference.

For seller to update: Update appropriate values in following cell.

.. code:: ipython3

    model_name='For Seller to update:<specify-model_or_endpoint-name>'
    
    content_type='For Seller to update:<specify_content_type_accepted_by_trained_model>'
    
    real_time_inference_instance_type='For Seller to update:<Update recommended_real-time_inference instance_type>'
    batch_transform_inference_instance_type='For Seller to update:<Update recommended_batch_transform_job_inference instance_type>'

A. Deploy trained model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor = estimator.deploy(1, real_time_inference_instance_type, serializer='<For seller to update>')

Once endpoint is created, you can perform real-time inference.

B. Create input payload
^^^^^^^^^^^^^^^^^^^^^^^

For Seller to update: Add code snippet that reads the input from
‘data/inference/input/real-time/’ directory and converts it into format
expected by the endpoint in the following cell




For Seller to update: Ensure that the ``file_name`` variable points to
the payload you created. Ensure that the ``output_file_name`` variable
points to a file-name in which output of real-time inference needs to be
stored


C. Perform real-time inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Seller to update: review/update ``file_name``, ``output_file name``,
and custom attributes in the following AWS CLI example to perform a
real-time inference using the payload file you created from 2.B

.. code:: ipython3

    !aws sagemaker-runtime invoke-endpoint \
        --endpoint-name $predictor.endpoint \
        --body fileb://$file_name \
        --content-type $content_type \
        --region $sagemaker_session.boto_region_name \
        $output_file_name

D. Visualize output
^^^^^^^^^^^^^^^^^^^

For Seller to update: Write code in the following cell to display the
output generated by real-time inference. This output must match with
output available in data/inference/output/real-time folder.


E. Calculate relevant metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For seller to update: write code to calculate metrics such as accuracy
or any other metrics relevant to the business problem, using the test
dataset. **This is highly recommended if your algorithm does not support
and calculate metrics on test channel**. For information on how to
configure metrics for your algorithm, see `Step 4 of this blog
post <https://aws.amazon.com/blogs/machine-learning/easily-monitor-and-visualize-metrics-while-training-models-on-amazon-sagemaker/>`__.


If `Amazon SageMaker Model
Monitor <https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html>`__
supports the type of problem you are trying to solve using this
algorithm, use the following examples to add Model Monitor support to
your product: For sample code to enable and monitor the model, see
following notebooks: 1. `Enable Amazon SageMaker Model
Monitor <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_model_monitor/enable_model_monitor/SageMaker-Enable-Model-Monitor.ipynb>`__
2. `Amazon SageMaker Model Monitor - visualizing monitoring
results <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_model_monitor/visualization/SageMaker-Model-Monitor-Visualize.ipynb>`__

F. Delete the endpoint
^^^^^^^^^^^^^^^^^^^^^^

Now that you have successfully performed a real-time inference, you do
not need the endpoint any more. you can terminate the same to avoid
being charged.

.. code:: ipython3

    predictor=sage.RealTimePredictor(model_name, sagemaker_session,content_type)
    predictor.delete_endpoint(delete_endpoint_config=True)

Since this is an experiment, you do not need to run a hyperparameter
tuning job. However, if you would like to see how to tune a model
trained using a third-party algorithm with Amazon SageMaker’s
hyperparameter tuning functionality, you can run the optional tuning
step.

5: Tune your model! (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Seller to update/read: It is important to provide hyperparameter
tuning functionality as part of your algorithm. Users of algorithms
range from new developers, to data scientists and ML practitioners. As
an algorithm maker, you need to make your algorithm usable in
production. To be able to do so, you need to give tools such as
capability to tune a custom ML model using Amazon SageMaker Automatic
Model Tuning(HPO) SDK. Enabling your algorithm for automatic model
tuning functionality is really easy. You need to mark appropriate
hyperparameters as Tunable=True and emit multiple metrics that customers
can choose to tune an ML model on.

We recommend that you provide notes on how your customer can scale usage
of your algorithm for really large datasets.

**You are strongly recommended to provide this section with tuning
guidelines and code for running an automatic tuning job**.

For information about Automatic model tuning, see `Perform Automatic
Model
Tuning <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`__

A. Tuning Guidelines
^^^^^^^^^^^^^^^^^^^^

For Seller to update: Provide guidelines on how customer can tune their
ML model effectively using your algorithm in following cell. Provide
details such as which parameter can be tuned for best results.



B. Define Tuning configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For seller to update: Provide a recommended hyperparameter range
configuration in the following cell. This configuration would be used
for running an
`HPO <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html>`__
job. For More information, see `Define
Hyperparameters <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html>`__\ 

.. code:: ipython3

    hyperparameter_ranges = {}

For seller to update: As part of your algorithm, provide multiple
objective metrics so that customer can choose a metric for tuning a
custom ML model. Update the following variable with a most
suitable/popular metric that your algorithm emits. For more information,
see `Define
Metrics <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-metrics.html>`__

.. code:: ipython3

    objective_metric_name = '<For seller to update : Provide an appropriate objective metric emitted by the algorithm>'

For seller to update: Specify whether to maximize or minimize the
objective metric, in following cell.

.. code:: ipython3

    tuning_direction='<For seller to update: Provide tuning direction for objective metric specified>'

C. Run a model tuning job
^^^^^^^^^^^^^^^^^^^^^^^^^

For seller to update: Review/update the tuner configuration including
but not limited to ``base_tuning_job_name``, ``max_jobs``, and
``max_parallel_jobs``.

.. code:: ipython3

    
    tuner = HyperparameterTuner(estimator=estimator, base_tuning_job_name='<For Seller to update: Specify base job name>',
                                    objective_metric_name=objective_metric_name,
                                objective_type=tuning_direction,
                                    hyperparameter_ranges=hyperparameter_ranges,
                                    max_jobs=50, max_parallel_jobs=7)


For seller to update: Uncomment following lines, specify appropriate
channels, and run the tuner to test it out.

.. code:: ipython3

    #Uncomment following two lines to run Hyperparameter optimization job. 
    #tuner.fit({'training':  data})
    #tuner.wait()

For seller to update: Once you have tested the code written in the
preceding cell, comment three lines in the preceding cell so that
customers who choose to simply run entire notebook do not end up
triggering a tuning job.

Once you have completed a tuning job, (or even while the job is still
running) you can `clone and use this
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb>`__
to analyze the results to understand how each hyperparameter effects the
quality of the model.

6. Perform Batch inference
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will perform batch inference using multiple input
payloads together.

.. code:: ipython3

    #upload the batch-transform job input files to S3
    transform_input_folder = "data/inference/input/batch"
    transform_input = sagemaker_session.upload_data(transform_input_folder, key_prefix=model_name) 
    print("Transform input uploaded to " + transform_input)

.. code:: ipython3

    #Run the batch-transform job
    transformer = model.transformer(1, batch_transform_inference_instance_type)
    transformer.transform(transform_input, content_type=content_type)
    transformer.wait()

.. code:: ipython3

    #output is available on following path
    transformer.output_path

For Seller to update: Add code that displays output generated by the
batch transform job available in S3. This output must match the output
available in data/inference/output/batch folder.

7. Clean-up
~~~~~~~~~~~

A. Delete the model
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    predictor.delete_model()

B. Unsubscribe to the listing (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to unsubscribe to the algorithm, follow these steps.
Before you cancel the subscription, ensure that you do not have any
`deployable
model <https://console.aws.amazon.com/sagemaker/home#/models>`__ created
from the model package or using the algorithm. Note - You can find this
information by looking at the container name associated with the model.

**Steps to unsubscribe to product from AWS Marketplace**: 1. Navigate to
**Machine Learning** tab on `Your Software subscriptions
page <https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=mlmp_gitdemo_indust>`__
2. Locate the listing that you want to cancel the subscription for, and
then choose **Cancel Subscription** to cancel the subscription.
