MNIST Training and Prediction with SageMaker Chainer
----------------------------------------------------

`MNIST <http://yann.lecun.com/exdb/mnist/>`__, the “Hello World” of
machine learning, is a popular dataset for handwritten digit
classification. It consists of 70,000 28x28 grayscale images labeled in
10 digit classes (0 to 9). This tutorial will show how to train a model
to predict handwritten digits on the MNIST dataset by running a Chainer
script on SageMaker using the sagemaker-python-sdk.

For more information about the Chainer container, see the
sagemaker-chainer-containers repository and the sagemaker-python-sdk
repository:

-  https://github.com/aws/sagemaker-chainer-containers
-  https://github.com/aws/sagemaker-python-sdk

For more on Chainer, please visit the Chainer repository:

-  https://github.com/chainer/chainer

This notebook is adapted from the
`MNIST <https://github.com/chainer/chainer/tree/master/examples/mnist>`__
example in the Chainer repository.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    
    # Get a SageMaker-compatible role used by this Notebook Instance.
    role = get_execution_role()

This notebook shows how to use the SageMaker Python SDK to run your code
in a local container before deploying to SageMaker’s managed training or
hosting environments. Just change your estimator’s train_instance_type
to ``local`` or ``local_gpu``. For more information, see `local
mode <https://github.com/aws/sagemaker-python-sdk#local-mode>`__.

In order to use this feature you’ll need to install docker-compose (and
nvidia-docker if training with a GPU). Running following script will
install docker-compose or nvidia-docker-compose and configure the
notebook environment for you.

Note, you can only run a single local notebook at a time.

.. code:: ipython3

    !/bin/bash ./setup.sh

Download MNIST datasets
-----------------------

We can use Chainer’s built-in ``get_mnist()`` method to download, import
and preprocess the MNIST dataset.

.. code:: ipython3

    import chainer
    
    train, test = chainer.datasets.get_mnist()

Parse, save, and upload the data
--------------------------------

We save our data, then use ``sagemaker_session.upload_data`` to upload
the data to an S3 location used for training. The return value
identifies the S3 path to the uploaded data.

.. code:: ipython3

    import os
    import shutil
    import numpy as np
    
    train_images = np.array([data[0] for data in train])
    train_labels = np.array([data[1] for data in train])
    test_images = np.array([data[0] for data in test])
    test_labels = np.array([data[1] for data in test])
    
    try:
        os.makedirs('/tmp/data/train')
        os.makedirs('/tmp/data/test')
    
        np.savez('/tmp/data/train/train.npz', images=train_images, labels=train_labels)
        np.savez('/tmp/data/test/test.npz', images=test_images, labels=test_labels)
    
        train_input = sagemaker_session.upload_data(
            path=os.path.join('/tmp/data', 'train'),
            key_prefix='notebook/chainer/mnist')
        test_input = sagemaker_session.upload_data(
            path=os.path.join('/tmp/data', 'test'),
            key_prefix='notebook/chainer/mnist')
    finally:
        shutil.rmtree('/tmp/data')

Writing the Chainer script to run on Amazon SageMaker
-----------------------------------------------------

Training
~~~~~~~~

We need to provide a training script that can run on the SageMaker
platform. The training script is very similar to a training script you
might run outside of SageMaker, but you can access useful properties
about the training environment through various environment variables,
such as:

-  ``SM_MODEL_DIR``: A string representing the path to the directory to
   write model artifacts to. These artifacts are uploaded to S3 for
   model hosting.
-  ``SM_NUM_GPUS``: An integer representing the number of GPUs available
   to the host.
-  ``SM_OUTPUT_DIR``: A string representing the filesystem path to write
   output artifacts to. Output artifacts may include checkpoints,
   graphs, and other files to save, not including model artifacts. These
   artifacts are compressed and uploaded to S3 to the same S3 prefix as
   the model artifacts.

Supposing two input channels, ‘train’ and ‘test’, were used in the call
to the Chainer estimator’s ``fit()`` method, the following will be set,
following the format ``SM_CHANNEL_[channel_name]``:

-  ``SM_CHANNEL_TRAIN``: A string representing the path to the directory
   containing data in the ‘train’ channel
-  ``SM_CHANNEL_TEST``: Same as above, but for the ‘test’ channel.

A typical training script loads data from the input channels, configures
training with hyperparameters, trains a model, and saves a model to
``model_dir`` so that it can be hosted later. Hyperparameters are passed
to your script as arguments and can be retrieved with an
``argparse.ArgumentParser`` instance. For example, the script run by
this notebook starts with the following:

.. code:: python

   import argparse
   import os

   if __name__=='__main__':

       parser = argparse.ArgumentParser()

       # hyperparameters sent by the client are passed as command-line arguments to the script.
       parser.add_argument('--epochs', type=int, default=50)
       parser.add_argument('--batch-size', type=int, default=64)

       # Data and model checkpoints directories
       parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
       parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
       parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
       parser.add_argument('--test', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
       
       args, _ = parser.parse_known_args()
       
       num_gpus = int(os.environ['SM_NUM_GPUS'])
       
       # ... load from args.train and args.test, train a model, write model to args.model_dir.

Because the Chainer container imports your training script, you should
always put your training code in a main guard
(``if __name__=='__main__':``) so that the container does not
inadvertently run your training code at the wrong point in execution.

For more information about training environment variables, please visit
https://github.com/aws/sagemaker-containers.

Hosting and Inference
~~~~~~~~~~~~~~~~~~~~~

We use a single script to train and host the Chainer model. You can also
write separate scripts for training and hosting. In contrast with the
training script, the hosting script requires you to implement functions
with particular function signatures (or rely on defaults for those
functions).

These functions load your model, deserialize data sent by a client,
obtain inferences from your hosted model, and serialize predictions back
to a client:

-  **``model_fn(model_dir)`` (always required for hosting)**: This
   function is invoked to load model artifacts from those written into
   ``model_dir`` during training.

This is the ``model_fn`` used in the script below during hosting:

.. code:: python

   def model_fn(model_dir):
       model = L.Classifier(MLP(1000, 10))
       serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
       return model.predictor

-  ``input_fn(input_data, content_type)``: This function is invoked to
   deserialize prediction data when a prediction request is made. The
   return value is passed to predict_fn. ``input_data`` is the
   serialized input data in the body of the prediction request, and
   ``content_type``, the MIME type of the data.

-  ``predict_fn(input_data, model)``: This function accepts the return
   value of ``input_fn`` as the ``input_data`` parameter and the return
   value of ``model_fn`` as the ``model`` parameter and returns
   inferences obtained from the model.

-  ``output_fn(prediction, accept)``: This function is invoked to
   serialize the return value from ``predict_fn``, which is passed in as
   the ``prediction`` parameter, back to the SageMaker client in
   response to prediction requests.

``model_fn`` is always required, but default implementations exist for
the remaining functions. These default implementations can deserialize a
NumPy array, invoking the model’s ``__call__`` method on the input data,
and serialize a NumPy array back to the client.

This notebook relies on the default ``input_fn``, ``predict_fn``, and
``output_fn`` implementations. See the Chainer sentiment analysis
notebook for an example of how one can implement these hosting
functions.

Please examine the script below. Training occurs behind the main guard,
which prevents the function from being run when the script is imported,
and ``model_fn`` loads the model saved into ``model_dir`` during
training.

For more on writing Chainer scripts to run on SageMaker, or for more on
the Chainer container itself, please see the following repositories:

-  For writing Chainer scripts to run on SageMaker:
   https://github.com/aws/sagemaker-python-sdk
-  For more on the Chainer container and default hosting functions:
   https://github.com/aws/sagemaker-chainer-containers

.. code:: ipython3

    !pygmentize 'chainer_mnist_single_machine.py'

Create SageMaker chainer estimator
----------------------------------

To run our Chainer training script on SageMaker, we construct a
``sagemaker.chainer.estimator.Chainer`` estimator, which accepts several
constructor arguments:

-  ``entry_point``: The path to the Python script SageMaker runs for
   training and prediction.

-  ``train_instance_count``: An integer representing how many training
   instances to start.

-  ``train_instance_type``: The type of SageMaker instances for
   training. We pass the string ``local`` or ``local_gpu`` here to
   enable the local mode for training in the local environment.
   ``local`` is for cpu training and ``local_gpu`` is for gpu training.
   If you want to train on a remote instance, specify a SageMaker ML
   instance type here accordingly. See `Amazon SageMaker ML Instance
   Types <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__
   for a list of instance types.

-  ``hyperparameters``: A dictionary passed to the ``train`` function as
   ``hyperparameters``.

.. code:: ipython3

    import subprocess
    
    from sagemaker.chainer.estimator import Chainer
    
    instance_type = 'local'
    
    if subprocess.call('nvidia-smi') == 0:
        ## Set type to GPU if one is present
        instance_type = 'local_gpu'
        
    print("Instance type = " + instance_type)
    
    chainer_estimator = Chainer(entry_point='chainer_mnist_single_machine.py', role=role,
                                train_instance_count=1, train_instance_type=instance_type,
                                hyperparameters={'epochs': 3, 'batch_size': 128})

Train on MNIST data in S3
-------------------------

After we’ve constructed our Chainer object, we can fit it using the
MNIST data we uploaded to S3. SageMaker makes sure our data is available
in the local filesystem, so our user script can simply read the data
from disk.

.. code:: ipython3

    chainer_estimator.fit({'train': train_input, 'test': test_input})

Our user script writes various artifacts, such as plots, to a directory
``output_data_dir``, the contents of which SageMaker uploads to S3. Now
we download and extract these artifacts.

.. code:: ipython3

    try:
        os.makedirs('output/single_machine_mnist')
    except OSError:
        pass
    
    chainer_training_job = chainer_estimator.latest_training_job.name
    
    desc = chainer_estimator.sagemaker_session.sagemaker_client. \
               describe_training_job(TrainingJobName=chainer_training_job)
    output_data = desc['ModelArtifacts']['S3ModelArtifacts'].replace('model', 'output')
    !aws --region {sagemaker_session.boto_session.region_name} s3 cp {output_data} output/single_machine_mnist/output.tar
    !tar -xzvf output/single_machine_mnist/output.tar -C output/single_machine_mnist

These plots show the accuracy and loss over each epoch:

.. code:: ipython3

    from IPython.display import Image
    from IPython.display import display
    
    accuracy_graph = Image(filename="output/single_machine_mnist/accuracy.png",
                           width=800,
                           height=800)
    loss_graph = Image(filename="output/single_machine_mnist/loss.png",
                       width=800,
                       height=800)
    
    display(accuracy_graph, loss_graph)

Deploy model to endpoint
------------------------

After training, we deploy the model to an endpoint. Here we also specify
instance_type to be ``local`` or ``local_gpu`` to deploy the model to
the local environment.

.. code:: ipython3

    predictor = chainer_estimator.deploy(initial_instance_count=1, instance_type=instance_type)

Predict Hand-Written Digit
--------------------------

We can use this predictor returned by ``deploy`` to send inference
requests to our locally-hosted model. Let’s get some random test images
in MNIST first.

.. code:: ipython3

    import random
    
    import matplotlib.pyplot as plt
    
    num_samples = 5
    indices = random.sample(range(test_images.shape[0] - 1), num_samples)
    images, labels = test_images[indices], test_labels[indices]
    
    for i in range(num_samples):
        plt.subplot(1,num_samples,i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(labels[i])
        plt.axis('off')

Now let’s see if we can make correct predictions.

.. code:: ipython3

    prediction = predictor.predict(images)
    predicted_label = prediction.argmax(axis=1)
    print('The predicted labels are: {}'.format(predicted_label))

Now let’s get some test data from you! Drawing into the image box loads
the pixel data into a variable named ‘data’ in this notebook, which we
can then pass to the Chainer predictor.

.. code:: ipython3

    from IPython.display import HTML
    HTML(open("input.html").read())

Now let’s see if your writing can be recognized!

.. code:: ipython3

    image = np.array(data, dtype=np.float32)
    prediction = predictor.predict(image)
    predicted_label = prediction.argmax(axis=1)[0]
    print('What you wrote is: {}'.format(predicted_label))

Clean resources
---------------

After you have finished with this example, remember to delete the
prediction endpoint to release the instance associated with it.

.. code:: ipython3

    chainer_estimator.delete_endpoint()
