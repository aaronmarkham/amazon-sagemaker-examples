TensorFlow Eager Execution with Amazon SageMaker Script Mode and Automatic Model Tuning
=======================================================================================

Starting with TensorFlow version 1.11, you can use SageMaker’s prebuilt
TensorFlow containers with TensorFlow training scripts similar to those
you would use outside SageMaker. This feature is named Script Mode.

In this notebook, we will use Script Mode in conjunction with
TensorFlow’s Eager Execution mode, which is the default execution mode
of TensorFlow 2 onwards. Eager execution is an imperative interface
where operations are executed immediately, rather than building a static
computational graph. Advantages of Eager Execution include a more
intuitive interface with natural Python control flow and less
boilerplate, easier debugging, and support for dynamic models and almost
all of the available TensorFlow operations. It also features close
integration with tf.keras to make rapid prototyping even easier.

To demonstrate Script Mode, this notebook focuses on presenting a
relatively complete workflow. The workflow includes local training and
hosted training in SageMaker, as well as local inference and SageMaker
hosted inference with a real time endpoint. Additionally, Automatic
Model Tuning in SageMaker will be used to tune the model’s
hyperparameters. This workflow will be applied to a straightforward
regression task, predicting house prices based on the well-known Boston
Housing dataset. More specifically, this public dataset contains 13
features regarding housing stock of towns in the Boston area, including
features such as average number of rooms, accessibility to radial
highways, adjacency to the Charles River, etc.

To begin, we’ll import some necessary packages and set up directories
for training and test data.

.. code:: ipython3

    import os
    import tensorflow as tf
    
    
    tf.enable_eager_execution()
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    train_dir = os.path.join(os.getcwd(), 'data/train')
    os.makedirs(train_dir, exist_ok=True)
    
    test_dir = os.path.join(os.getcwd(), 'data/test')
    os.makedirs(test_dir, exist_ok=True)

Prepare dataset
===============

Next, we’ll import the dataset. The dataset itself is small and
relatively issue-free. For example, there are no missing values, a
common problem for many other datasets. Accordingly, preprocessing just
involves normalizing the data.

.. code:: ipython3

    from tensorflow.python.keras.datasets import boston_housing
    
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    
    # normalization of dataset
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    
    x_train = (x_train - mean) / (std + 1e-8)
    x_test = (x_test - mean) / (std + 1e-8)
    
    print('x train', x_train.shape, x_train.mean(), x_train.std())
    print('y train', y_train.shape, y_train.mean(), y_train.std())
    print('x test', x_test.shape, x_test.mean(), x_test.std())
    print('y test', y_test.shape, y_test.mean(), y_test.std())

The data is saved as Numpy files prior to both local mode training and
hosted training in SageMaker.

.. code:: ipython3

    import numpy as np
    
    np.save(os.path.join(train_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(train_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(test_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(test_dir, 'y_test.npy'), y_test)

Local mode training
-------------------

Amazon SageMaker’s Local Mode training feature is a convenient way to
make sure your code is working as expected before moving on to full
scale, hosted training. To train in Local Mode, it is necessary to have
docker-compose or nvidia-docker-compose (for GPU) installed in the
notebook instance. Running following script will install docker-compose
or nvidia-docker-compose and configure the notebook environment for you.

.. code:: ipython3

    !/bin/bash ./setup.sh

Next, we’ll set up a TensorFlow Estimator for Local Mode training. One
of the key parameters for an Estimator is the ``train_instance_type``,
which is the kind of hardware on which training will run. In the case of
Local Mode, we simply set this parameter to ``local`` to invoke Local
Mode training on the CPU, or to ``local_gpu`` if the instance has a GPU.
Other parameters of note are the algorithm’s hyperparameters, which are
passed in as a dictionary, and a Boolean parameter indicating that we
are using Script Mode.

Recall that we are using Local Mode here mainly to make sure our code is
working. Accordingly, instead of performing a full cycle of training
with many epochs (passes over the full dataset), we’ll train only for a
small number of epochs to confirm the code is working properly and avoid
wasting training time unnecessarily.

.. code:: ipython3

    import sagemaker
    from sagemaker.tensorflow import TensorFlow
    
    model_dir = '/opt/ml/model'
    train_instance_type = 'local'
    hyperparameters = {'epochs': 5, 'batch_size': 128, 'learning_rate': 0.01}
    local_estimator = TensorFlow(entry_point='train.py',
                           source_dir='train_model',
                           model_dir=model_dir,
                           train_instance_type=train_instance_type,
                           train_instance_count=1,
                           hyperparameters=hyperparameters,
                           role=sagemaker.get_execution_role(),
                           base_job_name='tf-eager-scriptmode-bostonhousing',
                           framework_version='1.12.0',
                           py_version='py3',
                           script_mode=True)

.. code:: ipython3

    inputs = {'train': f'file://{train_dir}',
              'test': f'file://{test_dir}'}
    
    local_estimator.fit(inputs)

Now that we’ve confirmed that our code is working, we have a model
checkpoint saved in S3 that we can retrieve and load. We can then make
predictions and compare them with the test set as a further sanity
check.

.. code:: ipython3

    !aws s3 cp {local_estimator.model_data} ./local_model/model.tar.gz

.. code:: ipython3

    !tar -xvzf ./local_model/model.tar.gz -C ./local_model

After the model checkpoint has been retrieved, we can load (“restore”)
it. Keep in mind that this is a checkpoint rather than a model in
TensorFlow’s SavedModel format, which is necessary for TensorFlow
Serving. In the section below on hosted endpoints, we’ll use a
SavedModel to serve predictions with TensorFlow Serving.

.. code:: ipython3

    from tensorflow.contrib.eager.python import tfe
    from train_model import model_def
    
    tf.keras.backend.clear_session()
    device = '/cpu:0' 
    
    with tf.device(device):    
        local_model = model_def.get_model()
        saver = tfe.Saver(local_model.variables)
        saver.restore('local_model/weights.ckpt')

With the model checkpoint restored, we can now generate predictions and
compare them to the actual housing prices in the test set. The values
are in units of $1000s. In case you’re wondering why the actual values
seem relatively low compared to today’s big city housing prices: the
paper referencing the dataset was originally published in 1978.

.. code:: ipython3

    with tf.device(device):   
        local_predictions = local_model.predict(x_test)
        
    print('predictions: \t{}'.format(local_predictions[:10].flatten().round(decimals=1)))
    print('target values: \t{}'.format(y_test[:10].round(decimals=1)))

SageMaker hosted training
-------------------------

Now that we’ve confirmed our code is working locally, we can move on to
use SageMaker’s hosted training functionality. Hosted training is
preferred to for doing actual training, especially large-scale,
distributed training. Before starting hosted training, the data must be
uploaded to S3. We’ll do that now, and confirm the upload was
successful.

.. code:: ipython3

    s3_prefix = 'tf-eager-scriptmode-bostonhousing'
    
    traindata_s3_prefix = '{}/data/train'.format(s3_prefix)
    testdata_s3_prefix = '{}/data/test'.format(s3_prefix)

.. code:: ipython3

    train_s3 = sagemaker.Session().upload_data(path='./data/train/', key_prefix=traindata_s3_prefix)
    test_s3 = sagemaker.Session().upload_data(path='./data/test/', key_prefix=testdata_s3_prefix)
    
    inputs = {'train':train_s3, 'test': test_s3}
    
    print(inputs)

We’re now ready to set up an Estimator object for hosted training. It is
similar to the Local Mode Estimator, except the ``train_instance_type``
has been set to a ML instance type instead of ``local`` for Local Mode.
Also, since we know our code is working now, we train for a larger
number of epochs.

With these two changes, we simply call ``fit`` to start the actual
hosted training.

.. code:: ipython3

    train_instance_type = 'ml.c5.xlarge'
    hyperparameters = {'epochs': 30, 'batch_size': 128, 'learning_rate': 0.01}
    
    estimator = TensorFlow(entry_point='train.py',
                           source_dir='train_model',
                           model_dir=model_dir,
                           train_instance_type=train_instance_type,
                           train_instance_count=1,
                           hyperparameters=hyperparameters,
                           role=sagemaker.get_execution_role(),
                           base_job_name='tf-eager-scriptmode-bostonhousing',
                           framework_version='1.12.0',
                           py_version='py3',
                           script_mode=True)

.. code:: ipython3

    estimator.fit(inputs)

As with the Local Mode training, hosted training produces a model
checkpoint saved in S3 that we can retrieve and load. We can then make
predictions and compare them with the test set. This also demonstrates
the modularity of SageMaker: having trained the model in SageMaker, you
can now take the model out of SageMaker and run it anywhere else.
Alternatively, you can deploy the model using SageMaker’s hosted
endpoints functionality.

.. code:: ipython3

    !aws s3 cp {estimator.model_data} ./model/model.tar.gz

.. code:: ipython3

    !tar -xvzf ./model/model.tar.gz -C ./model

.. code:: ipython3

    tf.keras.backend.clear_session()
    device = '/cpu:0' 
    
    with tf.device(device):    
        model = model_def.get_model()
        saver = tfe.Saver(model.variables)
        saver.restore('model/weights.ckpt')

.. code:: ipython3

    with tf.device(device):   
        predictions = model.predict(x_test)
        
    print('predictions: \t{}'.format(predictions[:10].flatten().round(decimals=1)))
    print('target values: \t{}'.format(y_test[:10].round(decimals=1)))

SageMaker hosted endpoint
-------------------------

After multiple sanity checks, we’re confident that our model is
performing as expected. If we wish to deploy the model to production, a
convenient option is to use a SageMaker hosted endpoint. The endpoint
will retrieve the TensorFlow SavedModel created during training and
deploy it within a TensorFlow Serving container. This all can be
accomplished with one line of code, an invocation of the Estimator’s
deploy method.

.. code:: ipython3

    predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m5.xlarge')

As one last sanity check, we can compare the predictions generated by
the endpoint with those generated locally by the model checkpoint we
retrieved from hosted training in SageMaker.

.. code:: ipython3

    results = predictor.predict(x_test[:10])['predictions'] 
    flat_list = [float('%.1f'%(item)) for sublist in results for item in sublist]
    print('predictions: \t{}'.format(np.array(flat_list)))
    print('target values: \t{}'.format(y_test[:10].round(decimals=1)))

Before proceeding with the rest of this notebook, you can delete the
prediction endpoint to release the instance(s) associated with it.

.. code:: ipython3

    sagemaker.Session().delete_endpoint(predictor.endpoint)

Automatic Model Tuning
----------------------

Selecting the right hyperparameter values to train your model can be
difficult. The right answer is dependent on your data; some algorithms
have many different hyperparameters that can be tweaked; some are very
sensitive to the hyperparameter values selected; and most have a
non-linear relationship between model fit and hyperparameter values.
SageMaker Automatic Model Tuning helps automate the hyperparameter
tuning process: it runs multiple training jobs with different
hyperparameter combinations to find the set with the best model
performance.

We begin by specifying the hyperparameters we wish to tune, and the
range of values over which to tune each one. We also must specify an
objective metric to be optimized: in this use case, we’d like to
minimize the validation loss.

.. code:: ipython3

    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
    from time import gmtime, strftime 
    
    hyperparameter_ranges = {
            'learning_rate': ContinuousParameter(0.001, 0.2, scaling_type="Logarithmic"),
            'epochs': IntegerParameter(10, 50),
            'batch_size': IntegerParameter(64, 256),
        }
    
    metric_definitions = [{'Name': 'loss',
                           'Regex': ' loss: ([0-9\\.]+)'},
                         {'Name': 'val_loss',
                           'Regex': ' val_loss: ([0-9\\.]+)'}]
    
    objective_metric_name = 'val_loss'
    objective_type = 'Minimize'

Next we specify a HyperparameterTuner object that takes the above
definitions as parameters. Each tuning job must be given a budget - a
maximum number of training jobs - and the tuning job will complete once
that many training jobs have been executed.

We also can specify how much parallelism to employ, in this case five
jobs, meaning that the tuning job will complete after three series of
five jobs in parallel have completed. For the default Bayesian
Optimization tuning strategy used here, the search is informed by the
results of previous groups of training jobs, so we don’t run all of the
jobs in parallel, but rather divide the jobs into groups of parallel
jobs. In other words, more parallel jobs will finish tuning sooner, but
may sacrifice accuracy.

Now we can launch a hyperparameter tuning job by calling the ``fit``
method of the HyperparameterTuner object. We will wait until the tuning
finished, which may take around 10 minutes.

.. code:: ipython3

    tuner = HyperparameterTuner(estimator,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                max_jobs=15,
                                max_parallel_jobs=5,
                                objective_type=objective_type)
    
    tuning_job_name = "tf-bostonhousing-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    tuner.fit(inputs, job_name=tuning_job_name)
    tuner.wait()

After the tuning job is finished, we can use the
``HyperparameterTuningJobAnalytics`` method to list the top 5 tuning
jobs with the best performance. Although the results typically vary from
tuning job to tuning job, the best validation loss from the tuning job
(under the FinalObjectiveValue column) likely will be lower than the
validation loss from the hosted training job above. For an example of a
more in-depth analysis of a tuning job, see
HPO_Analyze_TuningJob_Results.ipynb notebook.

.. code:: ipython3

    tuner_metrics = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
    tuner_metrics.dataframe().sort_values(['FinalObjectiveValue'], ascending=True).head(5)

The total training time and training jobs status can be checked with the
following script. Because automatic early stopping is by default off,
all the training jobs should be completed normally.

.. code:: ipython3

    total_time = tuner_metrics.dataframe()['TrainingElapsedTimeSeconds'].sum() / 3600
    print("The total training time is {:.2f} hours".format(total_time))
    tuner_metrics.dataframe()['TrainingJobStatus'].value_counts()

Assuming the best model from the tuning job is better than the model
produced by the hosted training job above, we could now easily deploy
that model. By calling the ``deploy`` method of the HyperparameterTuner
object we instantiated above, we can directly deploy the best model from
the tuning job to a SageMaker hosted endpoint:

``tuning_predictor = tuner.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')``

Since we already looked at how to use a SageMaker hosted endpoint above,
we won’t repeat that here. We’ve covered a lot of content in this
notebook: local and hosted training with Script Mode, local and hosted
inference in SageMaker, and Automatic Model Tuning. These are likely to
be central elements for most deep learning workflows in SageMaker.
