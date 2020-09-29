Sentiment Analysis with Apache MXNet and Gluon
==============================================

*(This notebook was tested with the “Python 3 (Data Science)” kernel.)*

This tutorial shows how to train and test a Sentiment Analysis (Text
Classification) model on Amazon SageMaker using Apache MXNet and the
Gluon API.

Download training and test data
-------------------------------

In this notebook, we train a Sentiment Analysis model on the `SST-2
(Stanford Sentiment Treebank 2)
dataset <https://nlp.stanford.edu/sentiment/index.html>`__. This dataset
consists of movie reviews with one sentence per review. The task is to
classify the review as either positive or negative.

We download the preprocessed version of this dataset from the links
below. Each line in the dataset has space separated tokens, with the
first token being the label: 1 for positive and 0 for negative.

.. code:: bash

    %%bash
    mkdir data
    
    curl https://raw.githubusercontent.com/saurabh3949/Text-Classification-Datasets/master/stsa.binary.phrases.train > data/train
    curl https://raw.githubusercontent.com/saurabh3949/Text-Classification-Datasets/master/stsa.binary.test > data/test 

Upload the data
---------------

We use the ``sagemaker.s3.S3Uploader`` to upload our datasets to an
Amazon S3 location. The return value ``inputs`` identifies the location
– we use this later when we start the training job.

.. code:: ipython3

    from sagemaker import s3, session
    
    bucket = session.Session().default_bucket()
    inputs = s3.S3Uploader.upload('data', 's3://{}/mxnet-gluon-sentiment-example/data'.format(bucket))

Implement the training function
-------------------------------

We need to provide a training script that can run on the SageMaker
platform. The training scripts are essentially the same as one you would
write for local training, but you can also access useful properties
about the training environment through various environment variables. In
addition, hyperparameters are passed to the script as arguments. For
more about writing an MXNet training script for use with SageMaker, see
`the SageMaker
documentation <https://sagemaker.readthedocs.io/en/stable/using_mxnet.html#prepare-an-mxnet-training-script>`__.

The script here is a simplified implementation of `“Bag of Tricks for
Efficient Text Classification” <https://arxiv.org/abs/1607.01759>`__, as
implemented by Facebook’s
`FastText <https://github.com/facebookresearch/fastText/>`__ for text
classification. The model maps each word to a vector and averages
vectors of all the words in a sentence to form a hidden representation
of the sentence, which is inputted to a softmax classification layer.
For more details, please refer to `the
paper <https://arxiv.org/abs/1607.01759>`__.

At the end of every epoch, our script also checks the validation
accuracy, and checkpoints the best model so far, along with the
optimizer state, in the folder ``/opt/ml/checkpoints``. (If the folder
``/opt/ml/checkpoints`` does not exist, this checkpointing step is
skipped.)

.. code:: ipython3

    !pygmentize 'sentiment.py'

Run a SageMaker training job
----------------------------

The ``MXNet`` class allows us to run our training function on SageMaker
infrastructure. We need to configure it with our training script, an IAM
role, the number of training instances, and the training instance type.
In this case we run our training job on a single ``c4.2xlarge``
instance.

.. code:: ipython3

    from sagemaker import get_execution_role
    from sagemaker.mxnet import MXNet
    
    m = MXNet('sentiment.py',
              role=get_execution_role(),
              train_instance_count=1,
              train_instance_type='ml.c4.xlarge',
              framework_version='1.6.0',
              py_version='py3',
              distributions={'parameter_server': {'enabled': True}},
              hyperparameters={'batch-size': 8,
                               'epochs': 2,
                               'learning-rate': 0.01,
                               'embedding-size': 50, 
                               'log-interval': 1000})

After we’ve constructed our ``MXNet`` estimator, we can fit it using the
data we uploaded to S3. SageMaker makes sure our data is available in
the local filesystem, so our training script can simply read the data
from disk.

.. code:: ipython3

    m.fit(inputs)

As can be seen from the logs, our model gets over 80% accuracy on the
test set using the above hyperparameters.

After training, we use our ``MXNet`` object to build and deploy an
``MXNetPredictor`` object. This creates a SageMaker Endpoint that we can
use to perform inference.

.. code:: ipython3

    predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

With our predictor, we can perform inference on a JSON-encoded string
array.

The predictor runs inference on our input data and returns the predicted
sentiment (1 for positive and 0 for negative).

.. code:: ipython3

    data = ["this movie was extremely good .",
            "the plot was very boring .",
            "this film is so slick , superficial and trend-hoppy .",
            "i just could not watch it till the end .",
            "the movie was so enthralling !"]
    
    response = predictor.predict(data)
    print(response)

Cleanup
-------

After you have finished with this example, remember to delete the
prediction endpoint to release the instance(s) associated with it.

.. code:: ipython3

    predictor.delete_endpoint()
