Train and Host a Keras Model with Pipe Mode and Horovod on Amazon SageMaker
===========================================================================

Amazon SageMaker is a fully-managed service that provides developers and
data scientists with the ability to build, train, and deploy machine
learning (ML) models quickly. Amazon SageMaker removes the heavy lifting
from each step of the machine learning process to make it easier to
develop high-quality models. The SageMaker Python SDK makes it easy to
train and deploy models in Amazon SageMaker with several different
machine learning and deep learning frameworks, including TensorFlow and
Keras.

In this notebook, we train and host a `Keras Sequential
model <https://keras.io/getting-started/sequential-model-guide>`__ on
SageMaker. The model used for this notebook is a simple deep
convolutional neural network (CNN) that was extracted from `the Keras
examples <https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py>`__.

| For training our model, we also demonstrate distributed training with
  `Horovod <https://horovod.readthedocs.io>`__ and Pipe Mode. Amazon
  SageMaker’s Pipe Mode streams your dataset directly to your training
  instances instead of being downloaded first, which translates to
  training jobs that start sooner, finish quicker, and need less disk
  space.
|  Instance Type and Pricing: 

This notebook was trained using the Python 3 (TensorFlow CPU Optimized)
kernel using the ml.p3.2xlarge compute instance type in the us-west-2
region. Training time is approximately 70 minutes with the
aforementioned hardware specifications.

Price per hour depends on your region and instance type. You can
reference prices on the `SageMaker pricing
page <https://aws.amazon.com/sagemaker/pricing/>`__.

Setup
-----

First, we define a few variables that are be needed later in the
example.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    
    role = get_execution_role()

We also install a couple libraries we need for visualizing our model’s
prediction results.

.. code:: ipython3

    !pip install matplotlib seaborn

The CIFAR-10 dataset
--------------------

The `CIFAR-10 dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`__
is one of the most popular machine learning datasets. It consists of
60,000 32x32 images belonging to 10 different classes (6,000 images per
class). Here are the classes in the dataset, as well as 10 random images
from each:

.. figure:: https://maet3608.github.io/nuts-ml/_images/cifar10.png
   :alt: cifar10

   cifar10

Prepare the dataset for training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the CIFAR-10 dataset, we first download it and convert it to
TFRecords. This step takes around 5 minutes.

.. code:: ipython3

    !python generate_cifar10_tfrecords.py --data-dir ./data

Next, we upload the data to Amazon S3:

.. code:: ipython3

    from sagemaker.s3 import S3Uploader
    
    bucket = sagemaker_session.default_bucket()
    dataset_uri = S3Uploader.upload('data', 's3://{}/tf-cifar10-example/data'.format(bucket))
    
    display(dataset_uri)

Train the model
---------------

In this tutorial, we train a deep CNN to learn a classification task
with the CIFAR-10 dataset. We compare three different training jobs: a
baseline training job, training with Pipe Mode, and distributed training
with Horovod.

Configure metrics
~~~~~~~~~~~~~~~~~

In addition to running the training job, Amazon SageMaker can retrieve
training metrics directly from the logs and send them to CloudWatch
metrics. Here, we define metrics we would like to observe:

.. code:: ipython3

    metric_definitions = [
        {'Name': 'train:loss', 'Regex': '.*loss: ([0-9\\.]+) - accuracy: [0-9\\.]+.*'},
        {'Name': 'train:accuracy', 'Regex': '.*loss: [0-9\\.]+ - accuracy: ([0-9\\.]+).*'},
        {'Name': 'validation:accuracy', 'Regex': '.*step - loss: [0-9\\.]+ - accuracy: [0-9\\.]+ - val_loss: [0-9\\.]+ - val_accuracy: ([0-9\\.]+).*'},
        {'Name': 'validation:loss', 'Regex': '.*step - loss: [0-9\\.]+ - accuracy: [0-9\\.]+ - val_loss: ([0-9\\.]+) - val_accuracy: [0-9\\.]+.*'},
        {'Name': 'sec/steps', 'Regex': '.* - \d+s (\d+)[mu]s/step - loss: [0-9\\.]+ - accuracy: [0-9\\.]+ - val_loss: [0-9\\.]+ - val_accuracy: [0-9\\.]+'}
    ]

Run a baseline training job on SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SageMaker Python SDK’s ``sagemaker.tensorflow.TensorFlow`` estimator
class makes it easy for us to interact with SageMaker. Here, we create
one to configure a training job. Some parameters worth noting:

-  ``entry_point``: our training script (adapted from `this Keras
   example <https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py>`__).
-  ``metric_definitions``: the metrics (defined above) that we want sent
   to CloudWatch.
-  ``train_instance_count``: the number of training instances. Here, we
   set it to 1 for our baseline training job.

For more details about the TensorFlow estimator class, see the `API
documentation <https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html>`__.

.. code:: ipython3

    from sagemaker.tensorflow import TensorFlow
    
    hyperparameters = {'epochs': 10, 'batch-size': 256}
    tags = [{'Key': 'Project', 'Value': 'cifar10'}]
    
    estimator = TensorFlow(entry_point='keras_cifar10.py',
                           source_dir='source',
                           metric_definitions=metric_definitions,
                           hyperparameters=hyperparameters,
                           role=role,
                           framework_version='1.15.2',
                           py_version='py3',
                           train_instance_count=1,
                           train_instance_type='ml.p3.2xlarge',
                           base_job_name='cifar10-tf',
                           tags=tags)

Once we have our estimator, we call ``fit()`` to start the SageMaker
training job and pass the inputs that we uploaded to Amazon S3 earlier.
We pass the inputs as a dictionary to define different data channels for
training.

.. code:: ipython3

    inputs = {
        'train': '{}/train'.format(dataset_uri),
        'validation': '{}/validation'.format(dataset_uri),
        'eval': '{}/eval'.format(dataset_uri),
    }
    
    estimator.fit(inputs)

View the job training metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now view the metrics from the training job directly in the
SageMaker console.

Log into the `SageMaker
console <https://console.aws.amazon.com/sagemaker/home>`__, choose the
latest training job, and scroll down to the monitor section.
Alternatively, the code below uses the region and training job name to
generate a URL to CloudWatch metrics.

Using CloudWatch metrics, you can change the period and configure the
statistics.

.. code:: ipython3

    from urllib import parse
    
    from IPython.core.display import Markdown
    
    region = sagemaker_session.boto_region_name
    cw_url = parse.urlunparse((
        'https',
        '{}.console.aws.amazon.com'.format(region),
        '/cloudwatch/home',
        '',
        'region={}'.format(region),
        'metricsV2:namespace=/aws/sagemaker/TrainingJobs;dimensions=TrainingJobName;search={}'.format(estimator.latest_training_job.name),
    ))
    
    display(Markdown('CloudWatch metrics: [link]({}). After you choose a metric, '
                     'change the period to 1 Minute (Graphed Metrics -> Period).'.format(cw_url)))

Train on SageMaker with Pipe Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we train our model using Pipe Mode. With Pipe Mode, SageMaker uses
`Linux named pipes <https://www.linuxjournal.com/article/2156>`__ to
stream the training data directly from S3 instead of downloading the
data first.

In our script, we enable Pipe Mode using the following code:

.. code:: python

   from sagemaker_tensorflow import PipeModeDataset

   dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')

When we create our estimator, the only difference from before is that we
also specify ``input_mode='Pipe'``:

.. code:: ipython3

    pipe_mode_estimator = TensorFlow(entry_point='keras_cifar10.py',
                                     source_dir='source',
                                     metric_definitions=metric_definitions,
                                     hyperparameters=hyperparameters,
                                     role=role,
                                     framework_version='1.15.2',
                                     py_version='py3',
                                     train_instance_count=1,
                                     train_instance_type='ml.p3.2xlarge',
                                     input_mode='Pipe',
                                     base_job_name='cifar10-tf-pipe',
                                     tags=tags)

.. code:: ipython3

    pipe_mode_estimator.fit(inputs)

Distributed training with Horovod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Horovod <https://horovod.readthedocs.io>`__ is a distributed training
framework based on MPI. To use Horovod, we make the following changes to
our training script:

1. Enable Horovod:

.. code:: python

   import horovod.keras as hvd

   hvd.init()
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   config.gpu_options.visible_device_list = str(hvd.local_rank())
   K.set_session(tf.Session(config=config))

2. Add these callbacks:

.. code:: python

   hvd.callbacks.BroadcastGlobalVariablesCallback(0)
   hvd.callbacks.MetricAverageCallback()

3. Configure the optimizer:

.. code:: python

   opt = Adam(lr=learning_rate * size, decay=weight_decay)
   opt = hvd.DistributedOptimizer(opt)

To configure the training job, we specify the following for the
distribution:

.. code:: ipython3

    distribution = {
        'mpi': {
            'enabled': True,
            'processes_per_host': 1,  # Number of Horovod processes per host
        }
    }

This is then passed to our estimator:

.. code:: ipython3

    dist_estimator = TensorFlow(entry_point='keras_cifar10.py',
                                source_dir='source',
                                metric_definitions=metric_definitions,
                                hyperparameters=hyperparameters,
                                distributions=distribution,
                                role=role,
                                framework_version='1.15.2',
                                py_version='py3',
                                train_instance_count=2,
                                train_instance_type='ml.p3.2xlarge',
                                base_job_name='cifar10-tf-dist',
                                tags=tags)

.. code:: ipython3

    dist_estimator.fit(inputs)

Deploy the trained model
------------------------

After we train our model, we can deploy it to a SageMaker Endpoint,
which serves prediction requests in real-time. To do so, we simply call
``deploy()`` on our estimator, passing in the desired number of
instances and instance type for the endpoint.

Because we’re using TensorFlow Serving for deployment, our training
script saves the model in TensorFlow’s SavedModel format.

We don’t need accelerated computing power for inference, so let’s switch
over to a ml.m4.xlarge instance type.

For more information about deploying Keras and TensorFlow models in
SageMaker, see `this blog
post <https://aws.amazon.com/blogs/machine-learning/deploy-trained-keras-or-tensorflow-models-using-amazon-sagemaker>`__.

.. code:: ipython3

    predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

Invoke the endpoint
~~~~~~~~~~~~~~~~~~~

To verify the that the endpoint is in service, we generate some random
data in the correct shape and get a prediction.

.. code:: ipython3

    import numpy as np
    
    data = np.random.randn(1, 32, 32, 3)
    print('Predicted class: {}'.format(np.argmax(predictor.predict(data)['predictions'])))

Now let’s use the test dataset for predictions.

.. code:: ipython3

    from keras.datasets import cifar10
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

With the data loaded, we can use it for predictions:

.. code:: ipython3

    from keras.preprocessing.image import ImageDataGenerator
    
    def predict(data):
        predictions = predictor.predict(data)['predictions']
        return predictions
    
    
    predicted = []
    actual = []
    batches = 0
    batch_size = 128
    
    datagen = ImageDataGenerator()
    for data in datagen.flow(x_test, y_test, batch_size=batch_size):
        for i, prediction in enumerate(predict(data[0])):
            predicted.append(np.argmax(prediction))
            actual.append(data[1][i][0])
    
        batches += 1
        if batches >= len(x_test) / batch_size:
            break

With the predictions, we calculate our model accuracy and create a
confusion matrix.

.. code:: ipython3

    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(y_pred=predicted, y_true=actual)
    display('Average accuracy: {}%'.format(round(accuracy*100, 2)))

.. code:: ipython3

    %matplotlib inline
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sn
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_pred=predicted, y_true=actual)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sn.set(rc={'figure.figsize':(11.7,8.27)})
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(cm, annot=True, annot_kws={"size": 10})  # font size

Aided by the colors of the heatmap, we can use this confusion matrix to
understand how well the model performed for each label.

Cleanup
-------

To avoid incurring extra charges to your AWS account, let’s delete the
endpoint we created:

.. code:: ipython3

    predictor.delete_endpoint()
