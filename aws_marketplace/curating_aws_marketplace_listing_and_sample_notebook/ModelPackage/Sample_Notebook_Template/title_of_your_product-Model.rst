Deploy For Seller to update: Title_of_your_ML Model  Model Package from AWS Marketplace
--------------------------------------------------------------------------------------

 For Seller to update: Add overview of the ML Model here 
---------------------------------------------------------

This sample notebook shows you how to deploy For Seller to
update:\ `Title_of_your_ML
Model <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__\ 
using Amazon SageMaker.

Pre-requisites:
^^^^^^^^^^^^^^^

1. **Note**: This notebook contains elements which render correctly in
   Jupyter interface. Open this notebook from an Amazon SageMaker
   Notebook Instance or Amazon SageMaker Studio.
2. Ensure that IAM role used has **AmazonSageMakerFullAccess**
3. To deploy this ML model successfully, ensure that:

   1. Either your IAM role has these three permissions and you have
      authority to make AWS Marketplace subscriptions in the AWS account
      used:

      1. **aws-marketplace:ViewSubscriptions**
      2. **aws-marketplace:Unsubscribe**
      3. **aws-marketplace:Subscribe**

   2. or your AWS account has a subscription to For Seller to
      update:\ `Title_of_your_ML
      Model <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__\ .
      If so, skip step: `Subscribe to the model
      package <#1.-Subscribe-to-the-model-package>`__

Contents:
^^^^^^^^^

1. `Subscribe to the model
   package <#1.-Subscribe-to-the-model-package>`__
2. `Create an endpoint and perform real-time
   inference <#2.-Create-an-endpoint-and-perform-real-time-inference>`__

   1. `Create an endpoint <#A.-Create-an-endpoint>`__
   2. `Create input payload <#B.-Create-input-payload>`__
   3. `Perform real-time inference <#C.-Perform-real-time-inference>`__
   4. `Visualize output <#D.-Visualize-output>`__
   5. `Delete the endpoint <#E.-Delete-the-endpoint>`__

3. `Perform batch inference <#3.-Perform-batch-inference>`__
4. `Clean-up <#4.-Clean-up>`__

   1. `Delete the model <#A.-Delete-the-model>`__
   2. `Unsubscribe to the listing
      (optional) <#B.-Unsubscribe-to-the-listing-(optional)>`__

Usage instructions
^^^^^^^^^^^^^^^^^^

You can run this notebook one cell at a time (By using Shift+Enter for
running a cell).

1. Subscribe to the model package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To subscribe to the model package: 1. Open the model package listing
page For Seller to
update:\ `Title_of_your_product <Provide%20link%20to%20your%20marketplace%20listing%20of%20your%20product>`__.
1. On the AWS Marketplace listing, click on the **Continue to
subscribe** button. 1. On the **Subscribe to this software** page,
review and click on **“Accept Offer”** if you and your organization
agrees with EULA, pricing, and support terms. 1. Once you click on
**Continue to configuration button** and then choose a **region**, you
will see a **Product Arn** displayed. This is the model package ARN that
you need to specify while creating a deployable model using Boto3. Copy
the ARN corresponding to your region and specify the same in the
following cell.

.. code:: ipython3

    model_package_arn='<Customer to specify Model package ARN corresponding to their AWS region>'

 For Seller to update: Add all necessary imports in following cell, If
you need specific packages to be installed, # try to provide them in
this section, in a separate cell.

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

.. code:: ipython3

    role = get_execution_role()
    
    sagemaker_session = sage.Session()
    
    bucket=sagemaker_session.default_bucket()
    bucket

2. Create an endpoint and perform real-time inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to understand how real-time inference with Amazon SageMaker
works, see
`Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html>`__.

For Seller to update: update values for four variables in following
cell. Specify a model/endpoint name using only alphanumeric characters.

.. code:: ipython3

    model_name='For Seller to update:<specify-model_or_endpoint-name>'
    
    content_type='For Seller to update:<specify_content_type_accepted_by_model>'
    
    real_time_inference_instance_type='For Seller to update:<Update recommended_real-time_inference instance_type>'
    batch_transform_inference_instance_type='For Seller to update:<Update recommended_batch_transform_job_inference instance_type>'

A. Create an endpoint
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    
    def predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session,content_type)
    
    #create a deployable model from the model package.
    model = ModelPackage(role=role,
                        model_package_arn=model_package_arn,
                        sagemaker_session=sagemaker_session,
                        predictor_cls=predict_wrapper)
    
    #Deploy the model
    predictor = model.deploy(1, real_time_inference_instance_type, endpoint_name=model_name)

Once endpoint has been created, you would be able to perform real-time
inference.

B. Create input payload
^^^^^^^^^^^^^^^^^^^^^^^

For Seller to update: Add code snippet here that reads the input from
‘data/input/real-time/’ directory and converts it into format expected
by the endpoint.




For Seller to update: Ensure that file_name variable points to the
payload you created. Ensure that output_file_name variable points to a
file-name in which output of real-time inference needs to be stored.


C. Perform real-time inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Seller to update: review/update file_name, output_file name, custom
attributes in following AWS CLI to perform a real-time inference using
the payload file you created from 2.B

.. code:: ipython3

    !aws sagemaker-runtime invoke-endpoint \
        --endpoint-name $model_name \
        --body fileb://$file_name \
        --content-type $content_type \
        --region $sagemaker_session.boto_region_name \
        $output_file_name

D. Visualize output
^^^^^^^^^^^^^^^^^^^

For Seller to update: Write code in following cell to display the output
generated by real-time inference. This output must match with output
available in data/output/real-time folder.


For Seller to update: Get innovative! This is also your opportunity to
show-off different capabilities of the model. E.g. if your model does
object detection, multi-class classification, or regression, repeat
steps 2.B,2.C,2.D to show different inputs using files and outputs for
different classes/objects/edge conditions.


E. Delete the endpoint
^^^^^^^^^^^^^^^^^^^^^^

Now that you have successfully performed a real-time inference, you do
not need the endpoint any more. You can terminate the endpoint to avoid
being charged.

.. code:: ipython3

    predictor=sage.RealTimePredictor(model_name, sagemaker_session,content_type)
    predictor.delete_endpoint(delete_endpoint_config=True)

3. Perform batch inference
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will perform batch inference using multiple input
payloads together. If you are not familiar with batch transform, and
want to learn more, see these links: 1. `How it
works <https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-batch-transform.html>`__
2. `How to run a batch transform
job <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html>`__

.. code:: ipython3

    #upload the batch-transform job input files to S3
    transform_input_folder = "data/input/batch"
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
available in data/output/batch folder.

4. Clean-up
~~~~~~~~~~~

A. Delete the model
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.delete_model()

B. Unsubscribe to the listing (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to unsubscribe to the model package, follow these
steps. Before you cancel the subscription, ensure that you do not have
any `deployable
model <https://console.aws.amazon.com/sagemaker/home#/models>`__ created
from the model package or using the algorithm. Note - You can find this
information by looking at the container name associated with the model.

**Steps to unsubscribe to product from AWS Marketplace**: 1. Navigate to
**Machine Learning** tab on `Your Software subscriptions
page <https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=mlmp_gitdemo_indust>`__
2. Locate the listing that you want to cancel the subscription for, and
then choose **Cancel Subscription** to cancel the subscription.
