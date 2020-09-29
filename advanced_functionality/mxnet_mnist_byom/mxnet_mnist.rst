Mxnet BYOM: Train locally and deploy on SageMaker.
==================================================

1. `Introduction <#Introduction>`__
2. `Prerequisites and Preprocessing <#Prequisites-and-Preprocessing>`__

   1. `Permissions and environment
      variables <#Permissions-and-environment-variables>`__
   2. `Data Setup <#Data-setup>`__

3. `Training the network locally <#Training>`__
4. `Set up hosting for the model <#Set-up-hosting-for-the-model>`__

   1. `Export from MXNet <#Export-the-model-from-mxnet>`__
   2. `Import model into SageMaker <#Import-model-into-SageMaker>`__
   3. `Create endpoint <#Create-endpoint>`__

5. `Validate the endpoint for use <#Validate-the-endpoint-for-use>`__

**Note**: Compare this with the `tensorflow bring your own model
example <../tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb>`__

Introduction
------------

In this notebook, we will train a neural network locally on the location
from where this notebook is run using MXNet. We will then see how to
create an endpoint from the trained MXNet model and deploy it on
SageMaker. We will then inference from the newly created SageMaker
endpoint.

The neural network that we will use is a simple fully-connected neural
network. The definition of the neural network can be found in the
accompanying `mnist.py <mnist.py>`__ file. The ``build_graph`` method
contains the model defnition (shown below).

.. code:: python

   def build_graph():
       data = mx.sym.var('data')
       data = mx.sym.flatten(data=data)
       fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
       act1 = mx.sym.Activation(data=fc1, act_type="relu")
       fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
       act2 = mx.sym.Activation(data=fc2, act_type="relu")
       fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
       return mx.sym.SoftmaxOutput(data=fc3, name='softmax')

From this definitnion we can see that there are two fully-connected
layers of 128 and 64 neurons each. The activations of the last
fully-connected layer is then fed into a Softmax layer of 10 neurons. We
use 10 neurons here because the datatset on which we are going to
predict is the MNIST dataset of hand-written digit recognition which has
10 classes. More details can be found about the dataset on the
`creator’s webpage <http://yann.lecun.com/exdb/mnist/>`__.

Prequisites and Preprocessing
-----------------------------

Permissions and environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we set up the linkage and authentication to AWS services. In this
notebook we only need the roles used to give learning and hosting access
to your data. The Sagemaker SDK will use S3 defualt buckets when needed.
Supply the role in the variable below.

.. code:: ipython3

    import boto3, re
    from sagemaker import get_execution_role
    
    role = get_execution_role()

Data setup
~~~~~~~~~~

Next, we need to pull the data from the author’s site to our local box.
Since we have ``mxnet`` utilities, we will use the utilities to download
the dataset locally.

.. code:: ipython3

    import mxnet as mx
    data = mx.test_utils.get_mnist()

Training
~~~~~~~~

It is time to train the network. Since we are training the network
locally, we can make use of mxnet training tools. The training method is
also in the accompanying `mnist.py <mnist.py>`__ file. The notebook
assumes that this instance is a ``p2.xlarge``. If running this in a
non-GPU notebook instance, please adjust num_gpus=0 and num_cpu=1 The
method is shown below.

.. code:: python

   def train(data, hyperparameters= {'learning_rate': 0.11}, num_cpus=0, num_gpus =1 , **kwargs):
       train_labels = data['train_label']
       train_images = data['train_data']
       test_labels = data['test_label']
       test_images = data['test_data']
       batch_size = 100
       train_iter = mx.io.NDArrayIter(train_images, train_labels, batch_size, shuffle=True)
       val_iter = mx.io.NDArrayIter(test_images, test_labels, batch_size)
       logging.getLogger().setLevel(logging.DEBUG)
       mlp_model = mx.mod.Module(
           symbol=build_graph(),
           context=get_train_context(num_cpus, num_gpus))
       mlp_model.fit(train_iter,
                     eval_data=val_iter,
                     optimizer='sgd',
                     optimizer_params={'learning_rate': float(hyperparameters.get("learning_rate", 0.1))},
                     eval_metric='acc',
                     batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                     num_epoch=10)
       return mlp_model

The method above collects the ``data`` variable that ``get_mnist``
method gives you (which is a dictionary of data arrays) along with a
dictionary of ``hyperparameters`` which only contains learning rate, and
other parameters. It creates a
```mxnet.mod.Module`` <https://mxnet.incubator.apache.org/api/python/module.html>`__
from the network graph we built in the ``build_graph`` method and trains
the network using the ``mxnet.mod.Module.fit`` method.

.. code:: ipython3

    from mnist import train
    model = train(data = data, num_cpus=0, num_gpus=1)

If you want to run the training on a cpu or if you are on an instance
with cpus only, pass appropriate arguments.

Set up hosting for the model
----------------------------

Export the model from mxnet
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to set up hosting, we have to import the model from training to
hosting. We will begin by exporting the model from MXNet and saving it
down. Analogous to the `TensorFlow
example <../tensorflow_iris_byom/tensorflow_BYOM_iris.ipynb>`__, some
structure needs to be followed. The exported model has to be converted
into a form that is readable by ``sagemaker.mxnet.model.MXNetModel``.
The following code describes exporting the model in a form that does the
same:

.. code:: ipython3

    import os
    import json
    os.mkdir('model')
    
    model.save_checkpoint('model/model', 0000)
    with open ( 'model/model-shapes.json', "w") as shapes:
        json.dump([{"shape": model.data_shapes[0][1], "name": "data"}], shapes)
    
    import tarfile
    def flatten(tarinfo):
        tarinfo.name = os.path.basename(tarinfo.name)
        return tarinfo
    
    tar = tarfile.open("model.tar.gz", "w:gz")
    tar.add("model", filter=flatten)
    tar.close()    

The above piece of code essentially hacks the MXNet model export into a
sagemaker-readable model export. Study the exported model files if you
want to organize your exports in the same fashion as well.
Alternatively, you can load the model on MXNet itself and load the
sagemaker model as you normally would. Refer
`here <https://github.com/aws/sagemaker-python-sdk#model-loading>`__ for
details on how to load MXNet models.

Import model into SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a new sagemaker session and upload the model on to the default S3
bucket. We can use the ``sagemaker.Session.upload_data`` method to do
this. We need the location of where we exported the model from MXNet and
where in our default bucket we want to store the model(\ ``/model``).
The default S3 bucket can be found using the
``sagemaker.Session.default_bucket`` method.

.. code:: ipython3

    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')

Use the ``sagemaker.mxnet.model.MXNetModel`` to import the model into
SageMaker that can be deployed. We need the location of the S3 bucket
where we have the model, the role for authentication and the entry_point
where the model defintion is stored (``mnist.py``). The import call is
the following:

.. code:: ipython3

    from sagemaker.mxnet.model import MXNetModel
    sagemaker_model = MXNetModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                      role = role,
                                      entry_point = 'mnist.py')

Create endpoint
~~~~~~~~~~~~~~~

Now the model is ready to be deployed at a SageMaker endpoint. We can
use the ``sagemaker.mxnet.model.MXNetModel.deploy`` method to do this.
Unless you have created or prefer other instances, we recommend using 1
``'ml.c4.xlarge'`` instance for this training. These are supplied as
arguments.

.. code:: ipython3

    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    predictor = sagemaker_model.deploy(initial_instance_count=1,
                                              instance_type='ml.m4.xlarge')

Validate the endpoint for use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now use this endpoint to classify hand-written digits. To see
inference in action, draw a digit in the image box below. The pixel data
from your drawing will be loaded into a ``data`` variable in this
notebook.

*Note: after drawing the image, you’ll need to move to the next notebook
cell.*

.. code:: ipython3

    from IPython.display import HTML
    HTML(open("input.html").read())

.. code:: ipython3

    response = predictor.predict(data)
    print('Raw prediction result:')
    print(response)
    
    labeled_predictions = list(zip(range(10), response[0]))
    print('Labeled predictions: ')
    print(labeled_predictions)
    
    labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
    print('Most likely answer: {}'.format(labeled_predictions[0]))

(Optional) Delete the Endpoint

.. code:: ipython3

    print(predictor.endpoint)

If you do not want continied use of the endpoint, you can remove it.
Remember, open endpoints are charged. If this is a simple test or
practice, it is recommended to delete them.

.. code:: ipython3

    sagemaker.Session().delete_endpoint(predictor.endpoint)

Clear all stored model data so that we don’t overwrite them the next
time.

.. code:: ipython3

    os.remove('model.tar.gz')
    import shutil
    shutil.rmtree('model')

