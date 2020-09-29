TensorFlow BYOM: Train locally and deploy on SageMaker.
=======================================================

1. `Introduction <#Introduction>`__
2. `Prerequisites and Preprocessing <#Prequisites-and-Preprocessing>`__

   1. `Permissions and environment
      variables <#Permissions-and-environment-variables>`__
   2. `Model definitions <#Model-definitions>`__
   3. `Data Setup <#Data-setup>`__

3. `Training the network locally <#Training>`__
4. `Set up hosting for the model <#Set-up-hosting-for-the-model>`__

   1. `Export from TensorFlow <#Export-the-model-from-tensorflow>`__
   2. `Import model into SageMaker <#Import-model-into-SageMaker>`__
   3. `Create endpoint <#Create-endpoint>`__

5. `Validate the endpoint for use <#Validate-the-endpoint-for-use>`__

**Note**: Compare this with the `tensorflow bring your own model
example <../tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb>`__

Introduction
------------

This notebook can be compared to `Iris classification example
notebook <../tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb>`__
in terms of its functionality. We will do the same classification task,
but we will train the same network locally in the box from where this
notebook is being run. We then setup a real-time hosted endpoint in
SageMaker.

Consider the following model definition for IRIS classification. This
mode uses the ``tensorflow.estimator.DNNClassifier`` which is a
pre-defined estimator module for its model definition. The model
definition is the same as the one used in the `Iris classification
example
notebook <../tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb>`__

Prequisites and Preprocessing
-----------------------------

Permissions and environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we set up the linkage and authentication to AWS services. In this
notebook we only need the roles used to give learning and hosting access
to your data. The Sagemaker SDK will use S3 defualt buckets when needed.
If the ``get_execution_role`` does not return a role with the
appropriate permissions, you’ll need to specify an IAM role arn that
does.

.. code:: ipython2

    import boto3, re
    from sagemaker import get_execution_role
    
    role = get_execution_role()

Model Definitions
~~~~~~~~~~~~~~~~~

We use the
```tensorflow.estimator.DNNClassifier`` <https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier>`__
estimator to set up our network. We also need to write some methods for
serving inputs during hosting and training. These methods are all found
below.

.. code:: ipython2

    !cat iris_dnn_classifier.py

Create an estimator object with this model definition.

.. code:: ipython2

    from iris_dnn_classifier import estimator_fn
    classifier = estimator_fn(run_config = None, params = None)

Data setup
~~~~~~~~~~

Next, we need to pull the data from tensorflow repository and make them
ready for training. The following will code block should do that.

.. code:: ipython2

    import os 
    from six.moves.urllib.request import urlopen
    
    # Data sets
    IRIS_TRAINING = "iris_training.csv"
    IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
    
    IRIS_TEST = "iris_test.csv"
    IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
    
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
          f.write(raw)
    
    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
          f.write(raw)

Create the data input streamer object.

.. code:: ipython2

    from iris_dnn_classifier import train_input_fn
    train_func = train_input_fn('.', params = None)

Training
~~~~~~~~

It is time to train the network. Since we are training the network
locally, we can make use of TensorFlow’s ``tensorflow.Estimator.train``
method. The model is trained locally in the box.

.. code:: ipython2

    classifier.train(input_fn = train_func, steps = 1000)

Set up hosting for the model
----------------------------

Export the model from tensorflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to set up hosting, we have to import the model from training to
hosting. We will begin by exporting the model from TensorFlow and saving
it down. Analogous to the `MXNet
example <../mxnet_mnist_byom/mxnet_mnist.ipynb>`__, some structure needs
to be followed. The exported model has to be converted into a form that
is readable by ``sagemaker.tensorflow.model.TensorFlowModel``. The
following code describes exporting the model in a form that does the
same:

There is a small difference between a SageMaker model and a TensorFlow
model. The conversion is easy and fairly trivial. Simply move the
tensorflow exported model into a directory ``export\Servo\`` and tar the
entire directory. SageMaker will recognize this as a loadable TensorFlow
model.

.. code:: ipython2

    from iris_dnn_classifier import serving_input_fn
    
    exported_model = classifier.export_savedmodel(export_dir_base = 'export/Servo/', 
                                   serving_input_receiver_fn = serving_input_fn)
    
    print (exported_model)
    import tarfile
    with tarfile.open('model.tar.gz', mode='w:gz') as archive:
        archive.add('export', recursive=True)

Import model into SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a new sagemaker session and upload the model on to the default S3
bucket. We can use the ``sagemaker.Session.upload_data`` method to do
this. We need the location of where we exported the model from
TensorFlow and where in our default bucket we want to store the
model(\ ``/model``). The default S3 bucket can be found using the
``sagemaker.Session.default_bucket`` method.

.. code:: ipython2

    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')

Use the ``sagemaker.tensorflow.model.TensorFlowModel`` to import the
model into SageMaker that can be deployed. We need the location of the
S3 bucket where we have the model, the role for authentication and the
entry_point where the model defintion is stored
(``iris_dnn_classifier.py``). The import call is the following:

.. code:: ipython2

    from sagemaker.tensorflow.model import TensorFlowModel
    sagemaker_model = TensorFlowModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                      role = role,
                                      framework_version = '1.12',
                                      entry_point = 'iris_dnn_classifier.py')

Create endpoint
~~~~~~~~~~~~~~~

Now the model is ready to be deployed at a SageMaker endpoint. We can
use the ``sagemaker.tensorflow.model.TensorFlowModel.deploy`` method to
do this. Unless you have created or prefer other instances, we recommend
using 1 ``'ml.m4.xlarge'`` instance for this example. These are supplied
as arguments.

.. code:: ipython2

    %%time
    predictor = sagemaker_model.deploy(initial_instance_count=1,
                                              instance_type='ml.m4.xlarge')

Validate the endpoint for use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now use this endpoint to classify. Run an example prediction on a
sample to ensure that it works.

.. code:: ipython2

    sample = [6.4,3.2,4.5,1.5]
    predictor.predict(sample)

Delete all temporary directories so that we are not affecting the next
run. Also, optionally delete the end points.

.. code:: ipython2

    os.remove('model.tar.gz')
    import shutil
    shutil.rmtree('export')

If you do not want to continue using the endpoint, you can remove it.
Remember, open endpoints are charged. If this is a simple test or
practice, it is recommended to delete them.

.. code:: ipython2

    sagemaker.Session().delete_endpoint(predictor.endpoint)
