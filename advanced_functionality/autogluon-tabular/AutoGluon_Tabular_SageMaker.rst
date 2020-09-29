AutoGluon Tabular with SageMaker
================================

`AutoGluon <https://github.com/awslabs/autogluon>`__ automates machine
learning tasks enabling you to easily achieve strong predictive
performance in your applications. With just a few lines of code, you can
train and deploy high-accuracy deep learning models on tabular, image,
and text data. This notebook shows how to use AutoGluon-Tabular with
Amazon SageMaker by creating custom containers.

Prerequisites
-------------

If using a SageMaker hosted notebook, select kernel ``conda_mxnet_p36``.

.. code:: ipython3

    # Make sure docker compose is set up properly for local mode
    !./setup.sh

.. code:: ipython3

    # Imports
    import os
    import boto3
    import sagemaker
    from time import sleep
    from collections import Counter
    import numpy as np
    import pandas as pd
    from sagemaker import get_execution_role, local, Model, utils, fw_utils, s3
    from sagemaker.estimator import Estimator
    from sagemaker.predictor import RealTimePredictor, csv_serializer, StringDeserializer
    from sklearn.metrics import accuracy_score, classification_report
    from IPython.core.display import display, HTML
    from IPython.core.interactiveshell import InteractiveShell
    
    # Print settings
    InteractiveShell.ast_node_interactivity = "all"
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 10)
    
    # Account/s3 setup
    session = sagemaker.Session()
    local_session = local.LocalSession()
    bucket = session.default_bucket()
    prefix = 'sagemaker/autogluon-tabular'
    region = session.boto_region_name
    role = get_execution_role()
    client = session.boto_session.client(
        "sts", region_name=region, endpoint_url=utils.sts_regional_endpoint(region)
        )
    account = client.get_caller_identity()['Account']
    ecr_uri_prefix = utils.get_ecr_image_uri_prefix(account, region)
    registry_id = fw_utils._registry_id(region, 'mxnet', 'py3', account, '1.6.0')
    registry_uri = utils.get_ecr_image_uri_prefix(registry_id, region)

Build docker images
~~~~~~~~~~~~~~~~~~~

First, build autogluon package to copy into docker image.

.. code:: ipython3

    if not os.path.exists('package'):
        !pip install PrettyTable -t package
        !pip install --upgrade boto3 -t package
        !pip install bokeh -t package
        !pip install --upgrade matplotlib -t package
        !pip install autogluon -t package

Now build the training/inference image and push to ECR

.. code:: ipython3

    training_algorithm_name = 'autogluon-sagemaker-training'
    inference_algorithm_name = 'autogluon-sagemaker-inference'

.. code:: ipython3

    !./container-training/build_push_training.sh {account} {region} {training_algorithm_name} {ecr_uri_prefix} {registry_id} {registry_uri}
    !./container-inference/build_push_inference.sh {account} {region} {inference_algorithm_name} {ecr_uri_prefix} {registry_id} {registry_uri}

Get the data
~~~~~~~~~~~~

| In this example we’ll use the direct-marketing dataset to build a
  binary classification model that predicts whether customers will
  accept or decline a marketing offer.
| First we’ll download the data and split it into train and test sets.
  AutoGluon does not require a separate validation set (it uses bagged
  k-fold cross-validation).

.. code:: ipython3

    # Download and unzip the data
    !aws s3 cp --region {region} s3://sagemaker-sample-data-{region}/autopilot/direct_marketing/bank-additional.zip .
    !unzip -qq -o bank-additional.zip
    !rm bank-additional.zip
    
    local_data_path = './bank-additional/bank-additional-full.csv'
    data = pd.read_csv(local_data_path)
    
    # Split train/test data
    train = data.sample(frac=0.7, random_state=42)
    test = data.drop(train.index)
    
    # Split test X/y
    label = 'y'
    y_test = test[label]
    X_test = test.drop(columns=[label])

Check the data
''''''''''''''

.. code:: ipython3

    train.head(3)
    train.shape
    
    test.head(3)
    test.shape
    
    X_test.head(3)
    X_test.shape

Upload the data to s3

.. code:: ipython3

    train_file = 'train.csv'
    train.to_csv(train_file,index=False)
    train_s3_path = session.upload_data(train_file, key_prefix='{}/data'.format(prefix))
    
    test_file = 'test.csv'
    test.to_csv(test_file,index=False)
    test_s3_path = session.upload_data(test_file, key_prefix='{}/data'.format(prefix))
    
    X_test_file = 'X_test.csv'
    X_test.to_csv(X_test_file,index=False)
    X_test_s3_path = session.upload_data(X_test_file, key_prefix='{}/data'.format(prefix))

Hyperparameter Selection
------------------------

The minimum required settings for training is just a target label,
``fit_args['label']``.

Additional optional hyperparameters can be passed to the
``autogluon.task.TabularPrediction.fit`` function via ``fit_args``.

Below shows a more in depth example of AutoGluon-Tabular hyperparameters
from the example `Predicting Columns in a Table - In
Depth <https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-indepth.html#model-ensembling-with-stacking-bagging>`__.
Please see `fit
parameters <https://autogluon.mxnet.io/api/autogluon.task.html?highlight=eval_metric#autogluon.task.TabularPrediction.fit>`__
for further information. Note that in order for hyperparameter ranges to
work in SageMaker, values passed to the ``fit_args['hyperparameters']``
must be represented as strings.

.. code:: python

   nn_options = {
       'num_epochs': "10",
       'learning_rate': "ag.space.Real(1e-4, 1e-2, default=5e-4, log=True)",
       'activation': "ag.space.Categorical('relu', 'softrelu', 'tanh')",
       'layers': "ag.space.Categorical([100],[1000],[200,100],[300,200,100])",
       'dropout_prob': "ag.space.Real(0.0, 0.5, default=0.1)"
   }

   gbm_options = {
       'num_boost_round': "100",
       'num_leaves': "ag.space.Int(lower=26, upper=66, default=36)"
   }

   model_hps = {'NN': nn_options, 'GBM': gbm_options} 

   fit_args = {
     'label': 'y',
     'presets': ['best_quality', 'optimize_for_deployment'],
     'time_limits': 60*10,
     'hyperparameters': model_hps,
     'hyperparameter_tune': True,
     'search_strategy': 'skopt'
   }

   hyperparameters = {
     'fit_args': fit_args,
     'feature_importance': True
   }

**Note:** Your hyperparameter choices may affect the size of the model
package, which could result in additional time taken to upload your
model and complete training. Including ``'optimize_for_deployment'`` in
the list of ``fit_args['presets']`` is recommended to greatly reduce
upload times.

.. code:: ipython3

    # Define required label and optional additional parameters
    fit_args = {
      'label': 'y',
      # Adding 'best_quality' to presets list will result in better performance (but longer runtime)
      'presets': ['optimize_for_deployment'],
    }
    
    # Pass fit_args to SageMaker estimator hyperparameters
    hyperparameters = {
      'fit_args': fit_args,
      'feature_importance': True
    }

Train
-----

| For local training set ``train_instance_type`` to ``local`` .
| For non-local training the recommended instance type is
  ``ml.m5.2xlarge``.

**Note:** Depending on how many underlying models are trained,
``train_volume_size`` may need to be increased so that they all fit on
disk.

.. code:: ipython3

    %%time
    
    instance_type = 'ml.m5.2xlarge'
    #instance_type = 'local'
    
    ecr_image = f'{ecr_uri_prefix}/{training_algorithm_name}:latest'
    
    estimator = Estimator(image_name=ecr_image,
                          role=role,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          hyperparameters=hyperparameters,
                          train_volume_size=100)
    
    # Set inputs. Test data is optional, but requires a label column.
    inputs = {'training': train_s3_path, 'testing': test_s3_path}
    
    estimator.fit(inputs)

Create Model
~~~~~~~~~~~~

.. code:: ipython3

    # Create predictor object
    class AutoGluonTabularPredictor(RealTimePredictor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, content_type='text/csv', 
                             serializer=csv_serializer, 
                             deserializer=StringDeserializer(), **kwargs)

.. code:: ipython3

    ecr_image = f'{ecr_uri_prefix}/{inference_algorithm_name}:latest'
    
    if instance_type == 'local':
        model = estimator.create_model(image=ecr_image, role=role)
    else:
        model_uri = os.path.join(estimator.output_path, estimator._current_job_name, "output", "model.tar.gz")
        model = Model(model_uri, ecr_image, role=role, sagemaker_session=session, predictor_cls=AutoGluonTabularPredictor)

Batch Transform
~~~~~~~~~~~~~~~

For local mode, either ``s3://<bucket>/<prefix>/output/`` or
``file:///<absolute_local_path>`` can be used as outputs.

By including the label column in the test data, you can also evaluate
prediction performance (In this case, passing ``test_s3_path`` instead
of ``X_test_s3_path``).

.. code:: ipython3

    output_path = f's3://{bucket}/{prefix}/output/'
    # output_path = f'file://{os.getcwd()}'
    
    transformer = model.transformer(instance_count=1, 
                                    instance_type=instance_type,
                                    strategy='MultiRecord',
                                    max_payload=6,
                                    max_concurrent_transforms=1,                              
                                    output_path=output_path)
    
    transformer.transform(test_s3_path, content_type='text/csv', split_type='Line')
    transformer.wait()

Endpoint
~~~~~~~~

Deploy remote or local endpoint
'''''''''''''''''''''''''''''''

.. code:: ipython3

    instance_type = 'ml.m5.2xlarge'
    #instance_type = 'local'
    
    predictor = model.deploy(initial_instance_count=1, 
                             instance_type=instance_type)

Attach to endpoint (or reattach if kernel was restarted)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: ipython3

    # Select standard or local session based on instance_type
    if instance_type == 'local': 
        sess = local_session
    else: 
        sess = session
    
    # Attach to endpoint
    predictor = AutoGluonTabularPredictor(predictor.endpoint, sagemaker_session=sess)

Predict on unlabeled test data
''''''''''''''''''''''''''''''

.. code:: ipython3

    results = predictor.predict(X_test.to_csv(index=False)).splitlines()
    
    # Check output
    print(Counter(results))

Predict on data that includes label column
''''''''''''''''''''''''''''''''''''''''''

Prediction performance metrics will be printed to endpoint logs.

.. code:: ipython3

    results = predictor.predict(test.to_csv(index=False)).splitlines()
    
    # Check output
    print(Counter(results))

Check that classification performance metrics match evaluation printed to endpoint logs as expected
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: ipython3

    y_results = np.array(results)
    
    print("accuracy: {}".format(accuracy_score(y_true=y_test, y_pred=y_results)))
    print(classification_report(y_true=y_test, y_pred=y_results, digits=6))

Clean up endpoint
'''''''''''''''''

.. code:: ipython3

    predictor.delete_endpoint()
