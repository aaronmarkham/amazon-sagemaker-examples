Hyperparameter Tuning using SageMaker Tensorflow Container
==========================================================

Kernel ``Python 3 (TensorFlow CPU (or GPU) Optimized)`` works well with
this notebook.

This tutorial focuses on how to create a convolutional neural network
model to train the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`__
using **SageMaker TensorFlow container**. It leverages hyperparameter
tuning to kick off multiple training jobs with different hyperparameter
combinations, to find the one with best model training result.

Set up the environment
----------------------

We will set up a few things before starting the workflow.

1. specify the s3 bucket and prefix where training data set and model
   artifacts will be stored
2. get the execution role which will be passed to sagemaker for
   accessing your resources such as s3 bucket

.. code:: ipython3

    import sagemaker
    
    bucket = sagemaker.Session().default_bucket() # we are using a default bucket here but you can change it to any bucket in your account
    prefix = 'sagemaker/DEMO-hpo-tensorflow-high' # you can customize the prefix (subfolder) here
    
    role = sagemaker.get_execution_role() # we are using the notebook instance role for training in this example

Now we’ll import the Python libraries we’ll need.

.. code:: ipython3

    import boto3
    from time import gmtime, strftime
    from sagemaker.tensorflow import TensorFlow
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

Download the MNIST dataset
--------------------------

.. code:: ipython3

    import utils
    from tensorflow.contrib.learn.python.learn.datasets import mnist
    import tensorflow as tf
    
    data_sets = mnist.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)
    
    utils.convert_to(data_sets.train, 'train', 'data')
    utils.convert_to(data_sets.validation, 'validation', 'data')
    utils.convert_to(data_sets.test, 'test', 'data')

Upload the data
---------------

We use the ``sagemaker.Session.upload_data`` function to upload our
datasets to an S3 location. The return value identifies the location –
we will use this later when we start the training job.

.. code:: ipython3

    inputs = sagemaker.Session().upload_data(path='data', bucket=bucket, key_prefix=prefix+'/data/mnist')
    print (inputs)

Construct a script for distributed training
-------------------------------------------

Here is the full code for the network model:

.. code:: ipython3

    !cat 'mnist.py'

The script here is and adaptation of the `TensorFlow MNIST
example <https://github.com/tensorflow/models/tree/master/official/mnist>`__.
It provides a ``model_fn(features, labels, mode)``, which is used for
training, evaluation and inference.

A regular ``model_fn``
~~~~~~~~~~~~~~~~~~~~~~

A regular **``model_fn``** follows the pattern: 1. `defines a neural
network <https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L96>`__
- `applies the ``features`` in the neural
network <https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L178>`__
- `if the ``mode`` is ``PREDICT``, returns the output from the neural
network <https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L186>`__
- `calculates the loss function comparing the output with the
``labels`` <https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L188>`__
- `creates an optimizer and minimizes the loss function to improve the
neural
network <https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L193>`__
- `returns the output, optimizer and loss
function <https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py#L205>`__

Writing a ``model_fn`` for distributed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When distributed training happens, the same neural network will be sent
to the multiple training instances. Each instance will predict a batch
of the dataset, calculate loss and minimize the optimizer. One entire
loop of this process is called **training step**.

Syncronizing training steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A `global
step <https://www.tensorflow.org/api_docs/python/tf/train/global_step>`__
is a global variable shared between the instances. It necessary for
distributed training, so the optimizer will keep track of the number of
**training steps** between runs:

.. code:: python

   train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

That is the only required change for distributed training!

Set up hyperparameter tuning job
--------------------------------

*Note, with the default setting below, the hyperparameter tuning job can
take about 30 minutes to complete.*

Now we will set up the hyperparameter tuning job using SageMaker Python
SDK, following below steps: \* Create an estimator to set up the
TensorFlow training job \* Define the ranges of hyperparameters we plan
to tune, in this example, we are tuning “learning_rate” \* Define the
objective metric for the tuning job to optimize \* Create a
hyperparameter tuner with above setting, as well as tuning resource
configurations

Similar to training a single TensorFlow job in SageMaker, we define our
TensorFlow estimator passing in the TensorFlow script, IAM role, and
(per job) hardware configuration.

.. code:: ipython3

    estimator = TensorFlow(entry_point='mnist.py',
                      role=role,
                      framework_version='1.12.0',
                      training_steps=1000, 
                      evaluation_steps=100,
                      train_instance_count=1,
                      train_instance_type='ml.m4.xlarge',
                      base_job_name='DEMO-hpo-tensorflow')

Once we’ve defined our estimator we can specify the hyperparameters we’d
like to tune and their possible values. We have three different types of
hyperparameters. - Categorical parameters need to take one value from a
discrete set. We define this by passing the list of possible values to
``CategoricalParameter(list)`` - Continuous parameters can take any real
number value between the minimum and maximum value, defined by
``ContinuousParameter(min, max)`` - Integer parameters can take any
integer value between the minimum and maximum value, defined by
``IntegerParameter(min, max)``

*Note, if possible, it’s almost always best to specify a value as the
least restrictive type. For example, tuning learning rate as a
continuous value between 0.01 and 0.2 is likely to yield a better result
than tuning as a categorical parameter with values 0.01, 0.1, 0.15, or
0.2.*

.. code:: ipython3

    hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.01, 0.2)}

Next we’ll specify the objective metric that we’d like to tune and its
definition, which includes the regular expression (Regex) needed to
extract that metric from the CloudWatch logs of the training job. In
this particular case, our script emits loss value and we will use it as
the objective metric, we also set the objective_type to be ‘minimize’,
so that hyperparameter tuning seeks to minize the objective metric when
searching for the best hyperparameter setting. By default,
objective_type is set to ‘maximize’.

.. code:: ipython3

    objective_metric_name = 'loss'
    objective_type = 'Minimize'
    metric_definitions = [{'Name': 'loss',
                           'Regex': 'loss = ([0-9\\.]+)'}]

Now, we’ll create a ``HyperparameterTuner`` object, to which we pass: -
The TensorFlow estimator we created above - Our hyperparameter ranges -
Objective metric name and definition - Tuning resource configurations
such as Number of training jobs to run in total and how many training
jobs can be run in parallel.

.. code:: ipython3

    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                max_jobs=9,
                                max_parallel_jobs=3,
                                objective_type=objective_type)

Launch hyperparameter tuning job
--------------------------------

And finally, we can start our hyperprameter tuning job by calling
``.fit()`` and passing in the S3 path to our train and test dataset.

After the hyperprameter tuning job is created, you should be able to
describe the tuning job to see its progress in the next step, and you
can go to SageMaker console->Jobs to check out the progress of the
progress of the hyperparameter tuning job.

.. code:: ipython3

    tuner.fit(inputs)

Let’s just run a quick check of the hyperparameter tuning jobs status to
make sure it started successfully.

.. code:: ipython3

    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

Analyze tuning job results - after tuning job is completed
----------------------------------------------------------

Please refer to “HPO_Analyze_TuningJob_Results.ipynb” to see example
code to analyze the tuning job results.

Deploy the best model
---------------------

Now that we have got the best model, we can deploy it to an endpoint.
Please refer to other SageMaker sample notebooks or SageMaker
documentation to see how to deploy a model.
