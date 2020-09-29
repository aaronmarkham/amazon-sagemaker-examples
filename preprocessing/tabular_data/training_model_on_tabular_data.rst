Training a Model on Tabular Data using Amazon SageMaker
-------------------------------------------------------

The purpose of this notebook is to demonstrate how to train a machine
learning model via Amazon SageMaker using tabular data. In this notebook
you can train either an XGBoost or Linear Learner (regression) model on
tabular data in Amazon SageMaker.

Prerequisite
^^^^^^^^^^^^

This notebook is a sequel to the
`preprocessing_tabular_data.ipynb <preprocessing_tabular_data.ipynb>`__
notebook. Before running this notebook, run
`preprocessing_tabular_data.ipynb <preprocessing_tabular_data.ipynb>`__
to preprocess the data used in this notebook.

Notes
^^^^^

In this notebook, we use the sklearn framework for data partitionining
and storemagic to share dataframes in
`preprocessing_tabular_data.ipynb <preprocessing_tabular_data.ipynb>`__.
While we load data into memory here we do note that is it possible to
skip this and load your partitioned data directly to an S3 bucket.

Tabular Data Sets
^^^^^^^^^^^^^^^^^

-  `boston house
   data <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`__
-  `california house
   data <https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html>`__
-  `diabetes
   data <https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html>`__

Library Dependencies:
^^^^^^^^^^^^^^^^^^^^^

-  sagemaker >= 2.0.0
-  numpy
-  pandas
-  plotly
-  sklearn
-  matplotlib
-  seaborn

Setting up the notebook
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import os
    import sys
    import plotly.express as px
    import plotly.offline as pyo
    import plotly.graph_objs as go
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    
    from sklearn.datasets import *
    import ast
    import sklearn.model_selection
    
    ## SageMaker dependencies
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.inputs import TrainingInput
    from sagemaker.image_uris import retrieve
    
    ## This instantiates a SageMaker session that we will be operating in. 
    session = sagemaker.Session()
    
    ## This object represents the IAM role that we are assigned.
    role = sagemaker.get_execution_role()
    print(role)

Step 1: Load Relevant Variables from preprocessing_tabular_data.ipynb (Required for this notebook)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we load in our training, test, and validation data sets. We
preprocessed this data in the
`preprocessing_tabular_data.ipynb <preprocessing_tabular_data.ipynb>`__
and persisted it using storemagic.

.. code:: ipython3

    # Load relevant dataframes and variables from preprocessing_tabular_data.ipynb required for this notebook
    %store -r X_train
    %store -r X_test
    %store -r X_val
    %store -r Y_train
    %store -r Y_test
    %store -r Y_val
    %store -r choosen_data_set
    %store -r sagemaker_version

Step 2: Uploading the data to S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we upload our training and validation data to an S3 bucket. This is
a critical step because we will be specifying this S3 bucket’s location
during the training step.

.. code:: ipython3

    data_dir = '../data/' + choosen_data_set
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    prefix = choosen_data_set+'-deploy-hl'
    pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
    pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
    
    val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
    train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)

Here we have a pointer to our training and validation data sets stored
in an S3 bucket.

.. code:: ipython3

    s3_input_train = TrainingInput(s3_data=train_location, content_type='text/csv')
    s3_input_validation = TrainingInput(s3_data=val_location, content_type='text/csv')

Step 3: Select and Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select between the XGBoost or Linear Learner algorithm by assigning
model_selected to either ‘xgboost’ or ‘linear-learner’.

.. code:: ipython3

    # Select between xgboost or linear-learner (regression)
    models = ['xgboost', 'linear-learner']
    model_selected = "xgboost"
    assert model_selected in models
    print("Selected model:", model_selected)

Here we retrieve our container and instantiate our model object using
the Estimator class.

.. code:: ipython3

    container = retrieve(framework=model_selected, region=session.boto_region_name, version='latest')
    
    model = sagemaker.estimator.Estimator(container,
                                        role, 
                                        instance_count=1, 
                                        instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                        sagemaker_session=session)

Step 4: Set hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thus far, we have instantiated our model with our container and uploaded
our preprocessed data to our S3 bucket. Next, we set our hyperparameters
for our choosen model. We note that both
`XGBoost <https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/xgboost_hyperparameters.html>`__
and `linear
learner <https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/ll_hyperparameters.html>`__
have different hyperparameters that can be set.

.. code:: ipython3

    if model_selected == "xgboost":
        model.set_hyperparameters(max_depth=5,
                                eta=0.2,
                                gamma=4,
                                min_child_weight=6,
                                subsample=0.8,
                                objective='reg:linear',
                                early_stopping_rounds=10,
                                num_round=1)
        
    if model_selected == 'linear-learner':
        model.set_hyperparameters(feature_dim=X_train.shape[1],
                               predictor_type='regressor',
                               mini_batch_size=100)

Our estimator object is instantiated with hyperparameter settings, now
it is time to train! To do this we specify our S3 bucket’s location that
is storing our training data and validation data and pass it via a
dictionary to the fit method.

.. code:: ipython3

    model.fit({'train': s3_input_train, 'validation': s3_input_validation}, wait=False)

Step 6: Save Trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

The model has been trained. Below we show how to view all trained models
in our S3 bucket and how to select and download a model of your choice
locally.

Below we show a list of all trained models in our S3 bucket.

.. code:: ipython3

    list_of_trained_models = sagemaker.s3.S3Downloader.list(s3_uri='s3://{}/{}/output'.format(session.default_bucket(), prefix))
    print("\n".join(list_of_trained_models))

To download a particular model assign the ``s3_uri`` parameter below to
be one of the models shown above. Below we select the last trained model
to download.

.. code:: ipython3

    sagemaker.s3.S3Downloader.download(s3_uri=list_of_trained_models[-1], local_path='./')

Below we safe guard your kernel environment by installing your original
sagemaker version.

.. code:: ipython3

    if not sagemaker_version is None:
        !{sys.executable} -m pip install -qU sagemaker_version
