MNIST Handwritten Digits Classification Experiment
--------------------------------------------------

This demo shows how you can use SageMaker Experiment Management Python
SDK to organize, track, compare, and evaluate your machine learning (ML)
model training experiments.

You can track artifacts for experiments, including data sets,
algorithms, hyper-parameters, and metrics. Experiments executed on
SageMaker such as SageMaker Autopilot jobs and training jobs will be
automatically tracked. You can also track artifacts for additional steps
within an ML workflow that come before/after model training e.g. data
pre-processing or post-training model evaluation.

The APIs also let you search and browse your current and past
experiments, compare experiments, and identify best performing models.

Now we will demonstrate these capabilities through an MNIST handwritten
digits classification example. The experiment will be organized as
follow:

1. Download and prepare the MNIST dataset.
2. Train a Convolutional Neural Network (CNN) Model. Tune the hyper
   parameter that configures the number of hidden channels in the model.
   Track the parameter configurations and resulting model accuracy using
   SageMaker Experiments Python SDK.
3. Finally use the search and analytics capabilities of Python SDK to
   search, compare and evaluate the performance of all model versions
   generated from model tuning in Step 2.
4. We will also see an example of tracing the complete linage of a model
   version i.e. the collection of all the data pre-processing and
   training configurations and inputs that went into creating that model
   version.

Make sure you selected ``Python 3 (Data Science)`` kernel.

Install Python SDKs
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sys

.. code:: ipython3

    !{sys.executable} -m pip install sagemaker-experiments

Install PyTroch
~~~~~~~~~~~~~~~

.. code:: ipython3

    !{sys.executable} -m pip install torch
    !{sys.executable} -m pip install torchvision

Setup
~~~~~

.. code:: ipython3

    import time
    
    import boto3
    import numpy as np
    import pandas as pd
    %config InlineBackend.figure_format = 'retina'
    from matplotlib import pyplot as plt
    from torchvision import datasets, transforms
    
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    from sagemaker.analytics import ExperimentAnalytics
    
    from smexperiments.experiment import Experiment
    from smexperiments.trial import Trial
    from smexperiments.trial_component import TrialComponent
    from smexperiments.tracker import Tracker

.. code:: ipython3

    sess = boto3.Session()
    sm = sess.client('sagemaker')
    role = get_execution_role()

Create a S3 bucket to hold data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # create a s3 bucket to hold data, note that your account might already created a bucket with the same name
    account_id = sess.client('sts').get_caller_identity()["Account"]
    bucket = 'sagemaker-experiments-{}-{}'.format(sess.region_name, account_id)
    prefix = 'mnist'
    
    try:
        if sess.region_name == "us-east-1":
            sess.client('s3').create_bucket(Bucket=bucket)
        else:
            sess.client('s3').create_bucket(Bucket=bucket, 
                                            CreateBucketConfiguration={'LocationConstraint': sess.region_name})
    except Exception as e:
        print(e)

Dataset
~~~~~~~

We download the MNIST hand written digits dataset, and then apply
transformation on each of the image.

.. code:: ipython3

    # download the dataset
    # this will not only download data to ./mnist folder, but also load and transform (normalize) them
    train_set = datasets.MNIST('mnist', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]), 
        download=True)
                               
    test_set = datasets.MNIST('mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]),
        download=False)

.. code:: ipython3

    plt.imshow(train_set.data[2].numpy())

After transforming the images in the dataset, we upload it to s3.

.. code:: ipython3

    inputs = sagemaker.Session().upload_data(path='mnist', bucket=bucket, key_prefix=prefix)
    print('input spec: {}'.format(inputs))

Now lets track the parameters from the data pre-processing step.

.. code:: ipython3

    with Tracker.create(display_name="Preprocessing", sagemaker_boto_client=sm) as tracker:
        tracker.log_parameters({
            "normalization_mean": 0.1307,
            "normalization_std": 0.3081,
        })
        # we can log the s3 uri to the dataset we just uploaded
        tracker.log_input(name="mnist-dataset", media_type="s3/uri", value=inputs)

Step 1 - Set up the Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an experiment to track all the model training iterations.
Experiments are a great way to organize your data science work. You can
create experiments to organize all your model development work for : [1]
a business use case you are addressing (e.g. create experiment named
“customer churn prediction”), or [2] a data science team that owns the
experiment (e.g. create experiment named “marketing analytics
experiment”), or [3] a specific data science and ML project. Think of it
as a “folder” for organizing your “files”.

Create an Experiment
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    mnist_experiment = Experiment.create(
        experiment_name=f"mnist-hand-written-digits-classification-{int(time.time())}", 
        description="Classification of mnist hand-written digits", 
        sagemaker_boto_client=sm)
    print(mnist_experiment)

Step 2 - Track Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

Now create a Trial for each training run to track the it’s inputs, parameters, and metrics.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While training the CNN model on SageMaker, we will experiment with
several values for the number of hidden channel in the model. We will
create a Trial to track each training job run. We will also create a
TrialComponent from the tracker we created before, and add to the Trial.
This will enrich the Trial with the parameters we captured from the data
pre-processing stage.

Note the execution of the following code takes a while.

.. code:: ipython3

    from sagemaker.pytorch import PyTorch, PyTorchModel

.. code:: ipython3

    hidden_channel_trial_name_map = {}

If you want to run the following training jobs asynchronously, you may
need to increase your resource limit. Otherwise, you can run them
sequentially.

.. code:: ipython3

    preprocessing_trial_component = tracker.trial_component

.. code:: ipython3

    for i, num_hidden_channel in enumerate([2, 5, 10, 20, 32]):
        # create trial
        trial_name = f"cnn-training-job-{num_hidden_channel}-hidden-channels-{int(time.time())}"
        cnn_trial = Trial.create(
            trial_name=trial_name, 
            experiment_name=mnist_experiment.experiment_name,
            sagemaker_boto_client=sm,
        )
        hidden_channel_trial_name_map[num_hidden_channel] = trial_name
        
        # associate the proprocessing trial component with the current trial
        cnn_trial.add_trial_component(preprocessing_trial_component)
        
        # all input configurations, parameters, and metrics specified in estimator 
        # definition are automatically tracked
        estimator = PyTorch(
            entry_point='./mnist.py',
            role=role,
            sagemaker_session=sagemaker.Session(sagemaker_client=sm),
            framework_version='1.1.0',
            train_instance_count=1,
            train_instance_type='ml.c4.xlarge',
            hyperparameters={
                'epochs': 2,
                'backend': 'gloo',
                'hidden_channels': num_hidden_channel,
                'dropout': 0.2,
                'kernel_size': 5,
                'optimizer': 'sgd'
            },
            metric_definitions=[
                {'Name':'train:loss', 'Regex':'Train Loss: (.*?);'},
                {'Name':'test:loss', 'Regex':'Test Average loss: (.*?),'},
                {'Name':'test:accuracy', 'Regex':'Test Accuracy: (.*?)%;'}
            ],
            enable_sagemaker_metrics=True
        )
        
        cnn_training_job_name = "cnn-training-job-{}".format(int(time.time()))
        
        # Now associate the estimator with the Experiment and Trial
        estimator.fit(
            inputs={'training': inputs}, 
            job_name=cnn_training_job_name,
            experiment_config={
                "TrialName": cnn_trial.trial_name,
                "TrialComponentDisplayName": "Training",
            },
            wait=True,
        )
        
        # give it a while before dispatching the next training job
        time.sleep(2)

Compare the model training runs for an experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we will use the analytics capabilities of Python SDK to query and
compare the training runs for identifying the best model produced by our
experiment. You can retrieve trial components by using a search
expression.

Some Simple Analyses
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    search_expression = {
        "Filters":[
            {
                "Name": "DisplayName",
                "Operator": "Equals",
                "Value": "Training",
            }
        ],
    }

.. code:: ipython3

    trial_component_analytics = ExperimentAnalytics(
        sagemaker_session=Session(sess, sm), 
        experiment_name=mnist_experiment.experiment_name,
        search_expression=search_expression,
        sort_by="metrics.test:accuracy.max",
        sort_order="Descending",
        metric_names=['test:accuracy'],
        parameter_names=['hidden_channels', 'epochs', 'dropout', 'optimizer']
    )

.. code:: ipython3

    trial_component_analytics.dataframe()

To isolate and measure the impact of change in hidden channels on model
accuracy, we vary the number of hidden channel and fix the value for
other hyperparameters.

Next let’s look at an example of tracing the lineage of a model by
accessing the data tracked by SageMaker Experiments for
``cnn-training-job-2-hidden-channels`` trial

.. code:: ipython3

    lineage_table = ExperimentAnalytics(
        sagemaker_session=Session(sess, sm), 
        search_expression={
            "Filters":[{
                "Name": "Parents.TrialName",
                "Operator": "Equals",
                "Value": hidden_channel_trial_name_map[2]
            }]
        },
        sort_by="CreationTime",
        sort_order="Ascending",
    )

.. code:: ipython3

    lineage_table.dataframe()

Deploy endpoint for the best training-job / trial component
-----------------------------------------------------------

Now we’ll take the best (as sorted) and create an endpoint for it.

.. code:: ipython3

    #Pulling best based on sort in the analytics/dataframe so first is best....
    best_trial_component_name = trial_component_analytics.dataframe().iloc[0]['TrialComponentName']
    best_trial_component = TrialComponent.load(best_trial_component_name)
    
    model_data = best_trial_component.output_artifacts['SageMaker.ModelArtifact'].value
    env = {'hidden_channels': str(int(best_trial_component.parameters['hidden_channels'])), 
           'dropout': str(best_trial_component.parameters['dropout']), 
           'kernel_size': str(int(best_trial_component.parameters['kernel_size']))}
    model = PyTorchModel(
                model_data, 
                role, 
                './mnist.py', 
                env=env, 
                sagemaker_session=sagemaker.Session(sagemaker_client=sm),
                framework_version='1.1.0',
                name=best_trial_component.trial_component_name)
    
    predictor = model.deploy(
        instance_type='ml.m5.xlarge',
        initial_instance_count=1)

Cleanup
-------

Once we’re doing don’t forget to clean up the endpoint to prevent
unnecessary billing.

   Trial components can exist independent of trials and experiments. You
   might want keep them if you plan on further exploration. If so,
   comment out tc.delete()

.. code:: ipython3

    predictor.delete_endpoint()
    
    def cleanup(experiment):
        for trial_summary in experiment.list_trials():
            trial = Trial.load(sagemaker_boto_client=sm, trial_name=trial_summary.trial_name)
            for trial_component_summary in trial.list_trial_components():
                tc = TrialComponent.load(
                    sagemaker_boto_client=sm,
                    trial_component_name=trial_component_summary.trial_component_name)
                trial.remove_trial_component(tc)
                try:
                    # comment out to keep trial components
                    tc.delete()
                except:
                    # tc is associated with another trial
                    continue
                # to prevent throttling
                time.sleep(.5)
            trial.delete()
        experiment.delete()
    
    cleanup(mnist_experiment)

