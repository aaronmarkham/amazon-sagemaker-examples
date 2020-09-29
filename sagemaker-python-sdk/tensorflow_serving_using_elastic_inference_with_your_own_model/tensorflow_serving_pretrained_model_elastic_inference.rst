Using Amazon Elastic Inference with a pre-trained TensorFlow Serving model on SageMaker
=======================================================================================

This notebook demonstrates how to enable and use Amazon Elastic
Inference with our predefined SageMaker TensorFlow Serving containers.

Amazon Elastic Inference (EI) is a resource you can attach to your
Amazon EC2 instances to accelerate your deep learning (DL) inference
workloads. EI allows you to add inference acceleration to an Amazon
SageMaker hosted endpoint or Jupyter notebook for a fraction of the cost
of using a full GPU instance. For more information please visit:
https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html

This notebook’s main objective is to show how to create an endpoint,
backed by an Elastic Inference, to serve our pre-trained TensorFlow
Serving model for predictions. With a more efficient cost per
performance, Amazon Elastic Inference can prove to be useful for those
looking to use GPUs for higher inference performance at a lower cost.

1. `The model <#The-model>`__
2. `Setup role for SageMaker <#Setup-role-for-SageMaker>`__
3. `Load the TensorFlow Serving Model on Amazon SageMaker using Python
   SDK <#Load-the-TensorFlow-Serving-Model-on-Amazon-SageMaker-using-Python-SDK>`__
4. `Deploy the trained Model to an Endpoint with
   EI <#Deploy-the-trained-Model-to-an-Endpoint-with-EI>`__

   1. `Using EI with a SageMaker notebook
      instance <#Using-EI-with-a-SageMaker-notebook-instance>`__
   2. `Invoke the Endpoint to get
      inferences <#Invoke-the-Endpoint-to-get-inferences>`__
   3. `Delete the Endpoint <#Delete-the-Endpoint>`__

If you are familiar with SageMaker and already have a trained model,
skip ahead to the `Deploy the trained Model to an Endpoint with an
attached EI
accelerator <#Deploy-the-trained-Model-to-an-Endpoint-with-an-attached-EI-accelerator>`__

For this example, we will use the SageMaker Python SDK, which helps
deploy your models to train and host in SageMaker. In this particular
example, we will be interested in only the hosting portion of the SDK.

1. Set up our pre-trained model for consumption in SageMaker
2. Host the model in an endpoint with EI
3. Make a sample inference request to the model
4. Delete our endpoint after we’re done using it

The model
---------

The pre-trained model we will be using for this example is a NCHW
ResNet-50 model from the `official Tensorflow model Github
repository <https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model>`__.
For more information in regards to deep residual networks, please check
`here <https://github.com/tensorflow/models/tree/master/official/resnet>`__.
It isn’t a requirement to train our model on SageMaker to use SageMaker
for serving our model.

SageMaker expects our models to be compressed in a tar.gz format in S3.
Thankfully, our model already comes in that format. The predefined
TensorFlow Serving containers use REST API for handling inferences, for
more informationm, please see `Deploying to TensorFlow Serving
Endpoints <https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_tensorflow_serving.rst#making-predictions-against-a-sagemaker-endpoint>`__.

To host our model for inferences in SageMaker, we need to first upload
the SavedModel to S3. This can be done through the AWS console or AWS
command line.

For this example, the SavedModel object will already be hosted in a
public S3 bucket owned by SageMaker.

.. code:: ipython2

    %%time
    import boto3
    
    # use the region-specific saved model object
    region = boto3.Session().region_name
    saved_model = 's3://sagemaker-sample-data-{}/tensorflow/model/resnet/resnet_50_v2_fp32_NCHW.tar.gz'.format(region)

Setup role for SageMaker
------------------------

Let’s start by creating a SageMaker session and specifying the IAM role
arn used to give hosting access to your model. See the
`documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__
for how to create these. Note, if more than one role is required for
notebook instances, training, and/or hosting, please replace the
``sagemaker.get_execution_role()`` with a the appropriate full IAM role
arn string(s).

.. code:: ipython2

    import sagemaker
    
    role = sagemaker.get_execution_role()

Load the TensorFlow Serving Model on Amazon SageMaker using Python SDK
----------------------------------------------------------------------

We can use the SageMaker Python SDK to load our pre-trained TensorFlow
Serving model for hosting in SageMaker for predictions.

There are a few parameters that our TensorFlow Serving Model is
expecting. 1. ``model_data`` - The S3 location of a model tar.gz file to
load in SageMaker 2. ``role`` - An IAM role name or ARN for SageMaker to
access AWS resources on your behalf. 3. ``framework_version`` -
TensorFlow Serving version you want to use for handling your inference
request .

.. code:: ipython2

    from sagemaker.tensorflow.serving import Model
    
    tensorflow_model = Model(model_data=saved_model,
                             role=role,
                             framework_version='1.14')

Deploy the trained Model to an Endpoint with an attached EI accelerator
=======================================================================

The ``deploy()`` method creates an endpoint which serves prediction
requests in real-time.

The only change required for utilizing EI with our SageMaker TensorFlow
Serving containers only requires providing an ``accelerator_type``
parameter, which determines which type of EI accelerator to attach to
your endpoint. The supported types of accelerators can be found here:
https://aws.amazon.com/sagemaker/pricing/instance-types/

.. code:: ipython2

    %%time
    predictor = tensorflow_model.deploy(initial_instance_count=1,
                                        instance_type='ml.m4.xlarge',
                                        accelerator_type='ml.eia1.medium')

Using EI with a SageMaker notebook instance
-------------------------------------------

There is also the ability to utilize an EI accelerator attached to your
local SageMaker notebook instance. For more information, please
reference:
https://docs.aws.amazon.com/sagemaker/latest/dg/ei-notebook-instance.html

Invoke the Endpoint to get inferences
=====================================

Invoking prediction:

.. code:: ipython2

    %%time
    import numpy as np
    random_input = np.random.rand(1, 1, 3, 3)
    
    prediction = predictor.predict({'inputs': random_input.tolist()})
    
    print(prediction)

Delete the Endpoint
===================

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it.

.. code:: ipython2

    print(predictor.endpoint)

.. code:: ipython2

    import sagemaker
    
    predictor.delete_endpoint()
