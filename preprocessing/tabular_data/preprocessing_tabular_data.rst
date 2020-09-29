Preprocessing Tabular Data
--------------------------

The purpose of this notebook is to demonstrate how to preprocess tabular
data for training a machine learning model via Amazon SageMaker. In this
notebook we focus on preprocessing our tabular data and in a sequel
notebook,
`training_model_on_tabular_data.ipynb <training_model_on_tabular_data.ipynb>`__
we use our preprocessed tabular data to train a machine learning model.
We showcase how to preprocess 3 different tabular data sets.

Notes
^^^^^

In this notebook, we use the sklearn framework for data partitionining
and storemagic to share dataframes in
`training_model_on_tabular_data.ipynb <training_model_on_tabular_data.ipynb>`__.
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
    import subprocess
    import pkg_resources
    
    def get_sagemaker_version():
        "Return the version of 'sagemaker' in your kernel or -1 if 'sagemaker' is not installed"
        for i in pkg_resources.working_set:
            if i.key == "sagemaker":
                return "%s==%s" % (i.key, i.version)
        return -1
    
    # Store original 'sagemaker' version
    sagemaker_version = get_sagemaker_version()
    
    # Install any missing dependencies
    !{sys.executable} -m pip install -qU 'plotly' 'sagemaker>=2.0.0'
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import *
    import sklearn.model_selection
    
    # SageMaker dependencies
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.image_uris import retrieve
    
    # This instantiates a SageMaker session that we will be operating in. 
    session = sagemaker.Session()
    
    # This object represents the IAM role that we are assigned.
    role = sagemaker.get_execution_role()
    print(role)


Step 1: Select and Download Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here you can select the tabular data set of your choice to preprocess.

.. code:: ipython3

    data_sets = {'diabetes': 'load_diabetes()', 'california': 'fetch_california_housing()', 'boston' : 'load_boston()'}

To do select a particular dataset, assign **choosen_data_set** below to
be one of ‘diabetes’, ‘california’, or ‘boston’ where each name
corresponds to the it’s respective dataset.

-  ‘boston’ : `boston house
   data <https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html>`__
-  ‘california’ : `california house
   data <https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html>`__
-  ‘diabetes’ : `diabetes
   data <https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html>`__

.. code:: ipython3

    # Change choosen_data_set variable to one of the data sets above. 
    choosen_data_set = 'california'
    assert choosen_data_set in data_sets.keys()
    print("I selected the '{}' dataset!".format(choosen_data_set))

Step 2: Describe Feature Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here you can select the tabular data set of your choice to preprocess.

.. code:: ipython3

    data_set = eval(data_sets[choosen_data_set])
    
    X = pd.DataFrame(data_set.data, columns=data_set.feature_names)
    Y = pd.DataFrame(data_set.target)
    
    print("Features:", list(X.columns))
    print("Dataset shape:", X.shape)
    print("Dataset Type:", type(X))
    print("Label set shape:", Y.shape)
    print("Label set Type:", type(X))

We describe both our training data inputs X and outputs Y by computing the count, mean, std, min, percentiles.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    display(X.describe())

.. code:: ipython3

    display(Y.describe())

Step 3: Plot on Feature Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we show a heatmap and clustergrid across all our features. These
visualizations help us analyze correlated features and are particularly
important if we want to remove redundant features. The heatmap computes
a similarity score across each feature and colors like features using
this score. The clustergrid is similar, however it presents feature
correlations hierarchically.

**Note**: For the purposes of this notebook we do not remove any
features but by gathering the findings from these plots one may choose
to and can do so at this point.

.. code:: ipython3

    plt.figure(figsize=(14,12))
    cor = X.corr()
    sns.heatmap(cor, annot=True, cmap=sns.diverging_palette(20, 220, n=200))
    plt.show()

.. code:: ipython3

    cluster_map = sns.clustermap(cor, cmap =sns.diverging_palette(20, 220, n=200), linewidths = 0.1); 
    plt.setp(cluster_map.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
    cluster_map

Step 4: Partition Dataset into Train, Test, Validation Splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here using the sklearn framework we partition our selected dataset into
Train, Test and Validation splits. We choose a partition size of 1/3 and
then further split the training set into 2/3 training and 1/3 validation
set.

.. code:: ipython3

    # We partition the dataset into 2/3 training and 1/3 test set.
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33)
    
    # We further split the training set into a validation set i.e., 2/3 training set, and 1/3 validation set
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.33)

Step 5: Store Variables using storemagic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use storemagic to persist all relevant variables so they can be
reused in our sequel notebook,
`training_model_on_tabular_data.ipynb <training_model_on_tabular_data.ipynb>`__.

Alternatively, it is possible to upload your partitioned data to an S3
bucket and point to it during the model training phase. We note that
this is beyond the scope of this notebook hence why we omit it.

.. code:: ipython3

    # Using storemagic we persist the variables below so we can access them in the training_model_on_tabular_data.ipynb
    %store X_train
    %store X_test
    %store X_val
    %store Y_train
    %store Y_test
    %store Y_val
    %store choosen_data_set
    %store sagemaker_version
