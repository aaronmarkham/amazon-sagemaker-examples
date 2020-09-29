Hyperparameter Tuning with Amazon SageMaker and MXNet
=====================================================

**Creating a Hyperparameter Tuning Job for an MXNet Network**

--------------

--------------

Contents
--------

1. `Background <#Background>`__
2. `Setup <#Setup>`__
3. `Data <#Data>`__
4. `Code <#Code>`__
5. `Tune <#Train>`__
6. `Wrap-up <#Wrap-up>`__

--------------

Background
----------

This example notebook focuses on how to create a convolutional neural
network model to train the `MNIST
dataset <http://yann.lecun.com/exdb/mnist/>`__ using MXNet distributed
training. It leverages SageMaker’s hyperparameter tuning to kick off
multiple training jobs with different hyperparameter combinations, to
find the set with best model performance. This is an important step in
the machine learning process as hyperparameter settings can have a large
impact on model accuracy. In this example, we’ll use the `SageMaker
Python SDK <https://github.com/aws/sagemaker-python-sdk>`__ to create a
hyperparameter tuning job for an MXNet estimator.

--------------

Setup
-----

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the notebook instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the
   `documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/using-identity-based-policies.html>`__
   for more details on creating these. Note, if a role not associated
   with the current notebook instance, or more than one role is required
   for training and/or hosting, please replace
   ``sagemaker.get_execution_role()`` with a the appropriate full IAM
   role arn string(s).

.. code:: ipython3

    import sagemaker
    
    role = sagemaker.get_execution_role()

Now we’ll import the Python libraries we’ll need.

.. code:: ipython3

    import sagemaker
    import boto3
    from sagemaker.mxnet import MXNet
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

--------------

Data
----

The MNIST dataset is widely used for handwritten digit classification,
and consists of 70,000 labeled 28x28 pixel grayscale images of
hand-written digits. The dataset is split into 60,000 training images
and 10,000 test images. There are 10 classes (one for each of the 10
digits). See `here <http://yann.lecun.com/exdb/mnist/>`__ for more
details on MNIST.

For this example notebook we’ll use a version of the dataset that’s
already been published in the desired format to a shared S3 bucket.
Let’s specify that location now.

.. code:: ipython3

    region = boto3.Session().region_name
    train_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)
    test_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)

--------------

Code
----

To use SageMaker’s pre-built MXNet containers, we need to pass in an
MXNet script for the container to run. For our example, we’ll define
several functions, including: - ``load_data()`` and ``find_file()``
which help bring in our MNIST dataset as NumPy arrays -
``build_graph()`` which defines our neural network structure -
``train()`` which is the main function that is run during each training
job and calls the other functions in order to read in the dataset,
create a neural network, and train it.

There are also several functions for hosting which we won’t define, like
``input_fn()``, ``output_fn()``, and ``predict_fn()``. These will take
on their default values as described
`here <https://github.com/aws/sagemaker-python-sdk#model-serving>`__,
and are not important for the purpose of showcasing SageMaker’s
hyperparameter tuning.

.. code:: ipython3

    !cat mnist.py

Once we’ve specified and tested our training script to ensure it works,
we can start our tuning job. Testing can be done in either local mode or
using SageMaker training. Please see the `MXNet MNIST example
notebooks <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb>`__
for more detail.

--------------

Tune
----

Similar to training a single MXNet job in SageMaker, we define our MXNet
estimator passing in the MXNet script, IAM role, (per job) hardware
configuration, and any hyperparameters we’re not tuning.

.. code:: ipython3

    estimator = MXNet(entry_point='mnist.py',
                      role=role,
                      train_instance_count=1,
                      train_instance_type='ml.m4.xlarge',
                      sagemaker_session=sagemaker.Session(),
                      py_version='py3',
                      framework_version='1.4.1',
                      base_job_name='DEMO-hpo-mxnet',
                      hyperparameters={'batch_size': 100})

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
least restrictive type. For example, tuning ``thresh`` as a continuous
value between 0.01 and 0.2 is likely to yield a better result than
tuning as a categorical parameter with possible values of 0.01, 0.1,
0.15, or 0.2.*

.. code:: ipython3

    hyperparameter_ranges = {'optimizer': CategoricalParameter(['sgd', 'Adam']),
                             'learning_rate': ContinuousParameter(0.01, 0.2),
                             'num_epoch': IntegerParameter(10, 50)}

Next we’ll specify the objective metric that we’d like to tune and its
definition. This includes the regular expression (Regex) needed to
extract that metric from the CloudWatch logs of our training job.

.. code:: ipython3

    objective_metric_name = 'Validation-accuracy'
    metric_definitions = [{'Name': 'Validation-accuracy',
                           'Regex': 'Validation-accuracy=([0-9\\.]+)'}]

Now, we’ll create a ``HyperparameterTuner`` object, which we pass: - The
MXNet estimator we created above - Our hyperparameter ranges - Objective
metric name and definition - Number of training jobs to run in total and
how many training jobs should be run simultaneously. More parallel jobs
will finish tuning sooner, but may sacrifice accuracy. We recommend you
set the parallel jobs value to less than 10% of the total number of
training jobs (we’ll set it higher just for this example to keep it
short). - Whether we should maximize or minimize our objective metric
(we haven’t specified here since it defaults to ‘Maximize’, which is
what we want for validation accuracy)

.. code:: ipython3

    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                max_jobs=9,
                                max_parallel_jobs=3)

And finally, we can start our tuning job by calling ``.fit()`` and
passing in the S3 paths to our train and test datasets.

.. code:: ipython3

    tuner.fit({'train': train_data_location, 'test': test_data_location})

Let’s just run a quick check of the hyperparameter tuning jobs status to
make sure it started successfully and is ``InProgress``.

.. code:: ipython3

    boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']

--------------

Wrap-up
-------

Now that we’ve started our hyperparameter tuning job, it will run in the
background and we can close this notebook. Once finished, we can use the
`HPO Analysis
notebook <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb>`__
to determine which set of hyperparameters worked best.

For more detail on Amazon SageMaker’s Hyperparameter Tuning, please
refer to the AWS documentation.
