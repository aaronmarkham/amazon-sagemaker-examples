Training a sentiment analysis model with Chainer
------------------------------------------------

In this notebook, we will train a model that will allow us to analyze
text for positive or negative sentiment. The model will use a recurrent
neural network with long short-term memory blocks to generate word
embeddings.

The Chainer script runs inside of a Docker container running on
SageMaker. For more information about the Chainer container, see the
sagemaker-chainer-containers repository and the sagemaker-python-sdk
repository:

-  https://github.com/aws/sagemaker-chainer-containers
-  https://github.com/aws/sagemaker-python-sdk

For more on Chainer, please visit the Chainer repository:

-  https://github.com/chainer/chainer

The code in this notebook is adapted from the `text
classification <https://github.com/chainer/chainer/tree/master/examples/text_classification>`__
example in the Chainer repository.

.. code:: ipython3

    # Setup
    from sagemaker import get_execution_role
    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    
    # This role retrieves the SageMaker-compatible role used by this Notebook Instance.
    role = get_execution_role()

Downloading training and test data
----------------------------------

We use helper functions provided by ``chainer`` to download and
preprocess the data. We’ll be using the `Stanford Sentiment Treebank
dataset <https://nlp.stanford.edu/sentiment/>`__, which consists of
sentence fragments from movie reviews along with labels indicating
whether the sentence has a positive sentiment (1) or negative sentiment
(0).

.. code:: ipython3

    import dataset
    
    file_paths = dataset.download_dataset("stsa.binary")
    
    new_file_paths = dataset.get_stsa_dataset(file_paths)
    train, test, vocab = dataset.get_stsa_dataset(file_paths)
    
    with open(file_paths[0], 'r') as f:
        for i in range(20):
            line = f.readline()
            print(line)

Uploading the data
------------------

We save the preprocessed data to the local filesystem, and then use the
``sagemaker.Session.upload_data`` function to upload our datasets to an
S3 location. The return value ``inputs`` identifies the S3 location,
which we will use when we start the Training Job.

.. code:: ipython3

    import os
    import shutil
    
    import numpy as np
    
    train_data = [element[0] for element in train]
    train_labels = [element[1] for element in train]
    
    test_data = [element[0] for element in test]
    test_labels = [element[1] for element in test]
    
    
    try:
        os.makedirs('/tmp/data/train_sentiment')
        os.makedirs('/tmp/data/test_sentiment')
        os.makedirs('/tmp/data/vocab')
        np.savez('/tmp/data/train_sentiment/train.npz',data=train_data, labels=train_labels)
        np.savez('/tmp/data/test_sentiment/test.npz', data=test_data, labels=test_labels)
        np.save('/tmp/data/vocab/vocab.npy', vocab)
        train_input = sagemaker_session.upload_data(
                          path=os.path.join('/tmp', 'data', 'train_sentiment'),
                          key_prefix='notebook/chainer_sentiment/train')
        test_input = sagemaker_session.upload_data(
                         path=os.path.join('/tmp', 'data', 'test_sentiment'),
                         key_prefix='notebook/chainer_sentiment/test')
        vocab_input = sagemaker_session.upload_data(
                          path=os.path.join('/tmp', 'data', 'vocab'),
                          key_prefix='notebook/chainer_sentiment/vocab')
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
       
       parser.add_argument('--epochs', type=int, default=30)
       parser.add_argument('--batch-size', type=int, default=64)
       parser.add_argument('--dropout', type=float, default=0.4)
       parser.add_argument('--num-layers', type=int, default=1)
       parser.add_argument('--num-units', type=int, default=300)
       parser.add_argument('--model-type', type=str, default='rnn')

       # Data, model, and output directories. These are required.
       parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
       parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
       parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
       parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
       parser.add_argument('--vocab', type=str, default=os.environ['SM_CHANNEL_VOCAB'])
       
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

Please examine the script below. Training occurs behind the main guard,
which prevents the function from being run when the script is imported,
and ``model_fn`` loads the model saved into ``model_dir`` during
training.

``input_fn`` deserializes the input data into a NumPy array from the
default data format from the predictor Chainer uses to serialize
inference data in the Python SDK (the `NPY
format <https://docs.scipy.org/doc/numpy-1.14.0/neps/npy-format.html>`__).
``predict_fn`` formats words and converts them into word embeddings,
obtains predictions from the model containing the predicted sentiment
and returns a NumPy array that ``output_fn`` serializes to the NPY
format back to the client.

For more on writing Chainer scripts to run on SageMaker, or for more on
the Chainer container itself, please see the following repositories:

-  For writing Chainer scripts to run on SageMaker:
   https://github.com/aws/sagemaker-python-sdk
-  For more on the Chainer container and default hosting functions:
   https://github.com/aws/sagemaker-chainer-containers

The whole script ``src/sentiment_analysis.py`` is displayed below.

.. code:: ipython3

    !pygmentize 'src/sentiment_analysis.py'

Running the training script on SageMaker
----------------------------------------

To train a model with a Chainer script, we construct a ``Chainer``
estimator using the
`sagemaker-python-sdk <https://github.com/aws/sagemaker-python-sdk>`__.
We can pass in an ``entry_point``, the name of a script that contains a
couple of functions with certain signatures (``train`` and
``model_fn``). This script will be run on SageMaker in a container that
invokes these functions to train and load Chainer models.

The ``Chainer`` class allows us to run our training function as a
training job on SageMaker infrastructure. We need to configure it with
our training script, an IAM role, the number of training instances, and
the training instance type. In this case we will run our training job on
a ``ml.p2.xlarge`` instance.

.. code:: ipython3

    from sagemaker.chainer.estimator import Chainer
    
    chainer_estimator = Chainer(entry_point='sentiment_analysis.py',
                                source_dir="src",
                                role=role,
                                sagemaker_session=sagemaker_session,
                                train_instance_count=1,
                                train_instance_type='ml.p2.xlarge',
                                hyperparameters={'epochs': 10, 'batch-size': 64})
    
    chainer_estimator.fit({'train': train_input,
                           'test': test_input,
                           'vocab': vocab_input})

Our Chainer script writes various artifacts, such as plots, to a
directory ``output_data_dir``, the contents of which which SageMaker
uploads to S3. Now we download and extract these artifacts.

.. code:: ipython3

    from s3_util import retrieve_output_from_s3
    
    chainer_training_job = chainer_estimator.latest_training_job.name
    
    desc = sagemaker_session.sagemaker_client. \
               describe_training_job(TrainingJobName=chainer_training_job)
    output_data = desc['ModelArtifacts']['S3ModelArtifacts'].replace('model.tar.gz', 'output.tar.gz')
    
    retrieve_output_from_s3(output_data, 'output/sentiment')

These plots show the accuracy and loss over epochs.

In our user script, ``sentiment_analysis.py``, at the end of the
``train`` function. Our model overfits, but we save only the best model
for deployment.

.. code:: ipython3

    from IPython.display import Image
    from IPython.display import display
    
    accuracy_graph = Image(filename="output/sentiment/accuracy.png",
                           width=800,
                           height=800)
    loss_graph = Image(filename="output/sentiment/loss.png",
                       width=800,
                       height=800)
    
    display(accuracy_graph, loss_graph)

Deploying the Trained Model
---------------------------

After training, we use the Chainer estimator object to create and deploy
a hosted prediction endpoint. We can use a CPU-based instance for
inference (in this case an ``ml.m4.xlarge``), even though we trained on
GPU instances.

The predictor object returned by ``deploy`` lets us call the new
endpoint and perform inference on our sample images.

At the end of training, ``sentiment_analysis.py`` saves the trained
model, the vocabulary, and a dictionary of model properties that are
used to reconstruct the model. These model artifacts are loaded in
``model_fn`` when the model is hosted.

.. code:: ipython3

    predictor = chainer_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Predicting using SageMaker Endpoint
-----------------------------------

The Chainer predictor converts its input into a NumPy array, which it
serializes and sends to the hosted model. The ``predict_fn`` in
``sentiment_analysis.py`` receives this NumPy array and uses the loaded
model to make predictions on the input data, which it returns as a NumPy
array back to the Chainer predictor.

We predict against the hosted model on a batch of sentences. The output,
as defined by ``predict_fn``, consists of the processed input sentence,
the prediction, and the score for that prediction.

.. code:: ipython3

    sentences = ['It is fun and easy to train Chainer models on Amazon SageMaker!',
                 'It used to be slow, difficult, and laborious to train and deploy a model to production.',
                 'But now it is super fast to deploy to production. And I love it when my model generalizes!',]
    predictions = predictor.predict(sentences)
    for prediction in predictions:
        sentence, prediction, score = prediction
        print('sentence: {}\nprediction: {}\nscore: {}\n'.format(sentence, prediction, score))

We now predict against sentences in the test set:

.. code:: ipython3

    with open(file_paths[1], 'r') as f:
        sentences = f.readlines(2000)
        sentences = [sentence[1:].strip() for sentence in sentences]
        predictions = predictor.predict(sentences)
    
    predictions = predictor.predict(sentences)
    
    for prediction in predictions:
        sentence, prediction, score = prediction
        print('sentence: {}\nprediction: {}\nscore: {}\n'.format(sentence, prediction, score))
        

Cleanup
-------

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it.

.. code:: ipython3

    chainer_estimator.delete_endpoint()
