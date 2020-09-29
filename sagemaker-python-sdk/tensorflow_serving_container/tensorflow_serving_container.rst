Using the SageMaker TensorFlow Serving Container
================================================

The `SageMaker TensorFlow Serving
Container <https://github.com/aws/sagemaker-tensorflow-serving-container>`__
makes it easy to deploy trained TensorFlow models to a SageMaker
Endpoint without the need for any custom model loading or inference
code.

In this example, we will show how deploy one or more pre-trained models
from `TensorFlow Hub <https://www.tensorflow.org/hub/>`__ to a SageMaker
Endpoint using the `SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk>`__, and then use the
model(s) to perform inference requests.

Setup
-----

First, we need to ensure we have an up-to-date version of the SageMaker
Python SDK, and install a few additional python packages.

.. code:: ipython3

    !pip install -U --quiet "sagemaker>=1.14.2,<2"
    !pip install -U --quiet opencv-python tensorflow-hub

Next, we’ll get the IAM execution role from our notebook environment, so
that SageMaker can access resources in your AWS account later in the
example.

.. code:: ipython3

    from sagemaker import get_execution_role
    
    sagemaker_role = get_execution_role()

Download and prepare a model from TensorFlow Hub
------------------------------------------------

The TensorFlow Serving Container works with any model stored in
TensorFlow’s `SavedModel
format <https://www.tensorflow.org/guide/saved_model>`__. This could be
the output of your own training job or a model trained elsewhere. For
this example, we will use a pre-trained version of the MobileNet V2
image classification model from `TensorFlow Hub <https://tfhub.dev/>`__.

The TensorFlow Hub models are pre-trained, but do not include a serving
``signature_def``, so we’ll need to load the model into a TensorFlow
session, define the input and output layers, and export it as a
SavedModel. There is a helper function in this notebook’s
``sample_utils.py`` module that will do that for us.

.. code:: ipython3

    import sample_utils
    
    model_name = 'mobilenet_v2_140_224'
    export_path = 'mobilenet'
    model_path = sample_utils.tfhub_to_savedmodel(model_name, export_path)
    
    print('SavedModel exported to {}'.format(model_path))

After exporting the model, we can inspect it using TensorFlow’s
``saved_model_cli`` command. In the command output, you should see

::

   MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

   signature_def['serving_default']:
   ...

The command output should also show details of the model inputs and
outputs.

.. code:: ipython3

    !saved_model_cli show --all --dir {model_path}

Optional: add a second model
----------------------------

The TensorFlow Serving container can host multiple models, if they are
packaged in the same model archive file. Let’s prepare a second version
of the MobileNet model so we can demonstrate this. The
``mobilenet_v2_035_224`` model is a shallower version of MobileNetV2
that trades accuracy for smaller model size and faster computation, but
has the same inputs and outputs.

.. code:: ipython3

    second_model_name = 'mobilenet_v2_035_224'
    second_model_path = sample_utils.tfhub_to_savedmodel(second_model_name, export_path)
    
    print('SavedModel exported to {}'.format(second_model_path))

Next we need to create a model archive file containing the exported
model.

Create a model archive file
---------------------------

SageMaker models need to be packaged in ``.tar.gz`` files. When your
endpoint is provisioned, the files in the archive will be extracted and
put in ``/opt/ml/model/`` on the endpoint.

.. code:: ipython3

    !tar -C "$PWD" -czf mobilenet.tar.gz mobilenet/

Upload the model archive file to S3
-----------------------------------

We now have a suitable model archive ready in our notebook. We need to
upload it to S3 before we can create a SageMaker Model that. We’ll use
the SageMaker Python SDK to handle the upload.

.. code:: ipython3

    from sagemaker.session import Session
    
    model_data = Session().upload_data(path='mobilenet.tar.gz', key_prefix='model')
    print('model uploaded to: {}'.format(model_data))

Create a SageMaker Model and Endpoint
-------------------------------------

Now that the model archive is in S3, we can create a Model and deploy it
to an Endpoint with a few lines of python code:

.. code:: ipython3

    from sagemaker.tensorflow.serving import Model
    
    # Use an env argument to set the name of the default model.
    # This is optional, but recommended when you deploy multiple models
    # so that requests that don't include a model name are sent to a 
    # predictable model.
    env = {'SAGEMAKER_TFS_DEFAULT_MODEL_NAME': 'mobilenet_v2_140_224'}
    
    model = Model(model_data=model_data, role=sagemaker_role, framework_version='1.15.2', env=env)
    predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')

Make predictions using the endpoint
-----------------------------------

The endpoint is now up and running, and ready to handle inference
requests. The ``deploy`` call above returned a ``predictor`` object. The
``predict`` method of this object handles sending requests to the
endpoint. It also automatically handles JSON serialization of our input
arguments, and JSON deserialization of the prediction results.

We’ll use these sample images:

.. code:: ipython3

    # read the image files into a tensor (numpy array)
    kitten_image = sample_utils.image_file_to_tensor('kitten.jpg')
    
    # get a prediction from the endpoint
    # the image input is automatically converted to a JSON request.
    # the JSON response from the endpoint is returned as a python dict
    result = predictor.predict(kitten_image)
    
    # show the raw result
    print(result)

Add class labels and show formatted results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sample_utils`` module includes functions that can add Imagenet
class labels to our results and print formatted output. Let’s use them
to get a better sense of how well our model worked on the input image.

.. code:: ipython3

    # add class labels to the predicted result
    sample_utils.add_imagenet_labels(result)
    
    # show the probabilities and labels for the top predictions
    sample_utils.print_probabilities_and_labels(result)

Optional: make predictions using the second model
-------------------------------------------------

If you added the second model (``mobilenet_v2_035_224``) in the previous
optional step, then you can also send prediction requests to that model.
To do that, we’ll need to create a new ``predictor`` object.

Note: if you are using local mode (by changing the instance type to
``local`` or ``local_gpu``), you’ll need to create the new predictor
this way instead:

::

   predictor2 = Predictor(predictor.endpoint, model_name='mobilenet_v2_035_224', 
                          sagemaker_session=predictor.sagemaker_session)

.. code:: ipython3

    from sagemaker.tensorflow.serving import Predictor
    
    # use values from the default predictor to set up the new one
    predictor2 = Predictor(predictor.endpoint, model_name='mobilenet_v2_035_224')
    
    # make a new prediction
    bee_image = sample_utils.image_file_to_tensor('bee.jpg')
    result = predictor2.predict(bee_image)
    
    # show the formatted result
    sample_utils.add_imagenet_labels(result)
    sample_utils.print_probabilities_and_labels(result)

Additional Information
----------------------

The TensorFlow Serving Container supports additional features not
covered in this notebook, including support for:

-  TensorFlow Serving REST API requests, including classify and regress
   requests
-  CSV input
-  Other JSON formats

For information on how to use these features, refer to the documentation
in the `SageMaker Python
SDK <https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/deploying_tensorflow_serving.rst>`__.

Cleaning up
-----------

To avoid incurring charges to your AWS account for the resources used in
this tutorial, you need to delete the SageMaker Endpoint.

.. code:: ipython3

    predictor.delete_endpoint()
