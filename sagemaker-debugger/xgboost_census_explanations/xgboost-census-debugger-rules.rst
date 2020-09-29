.. raw:: html

   <h1>

Table of Contents

.. raw:: html

   </h1>

.. raw:: html

   <div class="toc">

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

Explainability with Amazon SageMaker Debugger

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

Introduction

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

Saving tensors

.. raw:: html

   </li>

.. raw:: html

   <li>

Analysis

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </li>

.. raw:: html

   <li>

Section 1 - Setup

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

1.1 Import necessary libraries

.. raw:: html

   </li>

.. raw:: html

   <li>

1.2 AWS region and IAM Role

.. raw:: html

   </li>

.. raw:: html

   <li>

1.3 S3 bucket and prefix to hold training data, debugger information and
model artifact

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </li>

.. raw:: html

   <li>

Section 2 - Data preparation

.. raw:: html

   </li>

.. raw:: html

   <li>

Section 3 - Train XGBoost model in Amazon SageMaker with debugger
enabled.

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

3.1 Install the ‘smdebug’ open source library

.. raw:: html

   </li>

.. raw:: html

   <li>

3.2 Build the XGBoost container

.. raw:: html

   </li>

.. raw:: html

   <li>

3.3 Enabling Debugger in Estimator object

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

DebuggerHookConfig

.. raw:: html

   </li>

.. raw:: html

   <li>

Rules

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </li>

.. raw:: html

   <li>

3.4 Result

.. raw:: html

   </li>

.. raw:: html

   <li>

3.5 Check the status of the Rule Evaluation Job

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </li>

.. raw:: html

   <li>

Section 4 - Analyze debugger output

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

Retrieving and Analyzing tensors

.. raw:: html

   </li>

.. raw:: html

   <li>

Plot Performance metrics

.. raw:: html

   </li>

.. raw:: html

   <li>

Feature importance

.. raw:: html

   </li>

.. raw:: html

   <li>

SHAP

.. raw:: html

   </li>

.. raw:: html

   <li>

Global explanations

.. raw:: html

   </li>

.. raw:: html

   <li>

Local explanations

.. raw:: html

   <ul class="toc-item">

.. raw:: html

   <li>

Force plot

.. raw:: html

   </li>

.. raw:: html

   <li>

Stacked force plot

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </li>

.. raw:: html

   <li>

Outliers

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </li>

.. raw:: html

   <li>

Conclusion

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </div>

Explainability with Amazon SageMaker Debugger
=============================================

**Explain a XGBoost model that predicts an individual’s income**

This notebook demonstrates how to use Amazon SageMaker Debugger to
capture the feature importance and SHAP values for a XGBoost model.

*This notebook was created and tested on an ml.t2.medium notebook
instance.*

Introduction 
-------------

Amazon SageMaker Debugger is the capability of Amazon SageMaker that
allows debugging machine learning training. The capability helps you
monitor the training jobs in near real time using rules and alert you
once it has detected inconsistency in training.

Using Amazon SageMaker Debugger is a two step process: Saving tensors
and Analysis. Let’s look at each one of them closely.

Saving tensors
~~~~~~~~~~~~~~

In deep learning algorithms, tensors define the state of the training
job at any particular instant in its lifecycle. Amazon SageMaker
Debugger exposes a library which allows you to capture these tensors and
save them for analysis.

Although XGBoost is not a deep learning algorithm, Amazon SageMaker
Debugger is highly customizable and can help provide interpretability by
saving insightful metrics, such as performance metrics or feature
importances, at different frequencies. Refer to
`documentation <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/xgboost.md>`__
for details on how to save the metrics you want.

Metrics saved can also include feature importance and SHAP values for
all features in the dataset. The feature importance and SHAP values
collected are what we will use to provide local and global
explainability.

Analysis
~~~~~~~~

After the tensors are saved, perform automatic analysis by running
debugging **Rules**. On a very broad level, a rule is Python code used
to detect certain conditions during training. Some of the conditions
that a data scientist training an algorithm may care about are
monitoring for gradients getting too large or too small, detecting
overfitting, and so on. Amazon SageMaker Debugger comes pre-packaged
with certain rules that can be invoked on Amazon SageMaker. Users can
also write their own rules using the Amazon SageMaker Debugger APIs. For
more information about automatic analysis using a rule, see the `rules
documentation <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md>`__.

Section 1 - Setup 
------------------

In this section, we will import the necessary libraries, setup variables
and examine dataset used. that was used to train the XGBoost model to
predict an individual’s income.

Let’s start by specifying:

-  The AWS region used to host your model.
-  The IAM role associated with this SageMaker notebook instance.
-  The S3 bucket used to store the data used to train the model, save
   debugger information during training and the trained model artifact.

1.1 Import necessary libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import boto3
    import sagemaker
    import os
    import pandas as pd
    
    from sagemaker import get_execution_role

1.2 AWS region and IAM Role
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    region = boto3.Session().region_name
    print("AWS Region: {}".format(region))
    
    role = get_execution_role()
    print("RoleArn: {}".format(role))

1.3 S3 bucket and prefix to hold training data, debugger information and model artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    bucket = sagemaker.Session().default_bucket()
    prefix = "DEMO-smdebug-xgboost-adult-income-prediction"

Amazon SageMaker Debugger is available in Amazon SageMaker XGBoost
container version 0.90-2 or later. If you want to use XGBoost with
Amazon SageMaker Debugger, you have to specify ``repo_version='0.90-2'``
in the ``get_image_uri`` function.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(region, "xgboost", repo_version="0.90-2")

Section 2 - Data preparation 
-----------------------------

We’ll be using the `Adult Census
dataset <https://archive.ics.uci.edu/ml/datasets/adult>`__ for this
exercise. This data was extracted from the `1994 Census bureau
database <http://www.census.gov/en.html>`__ by Ronny Kohavi and Barry
Becker (Data Mining and Visualization, Silicon Graphics), with the task
being to predict if an individual person makes over 50K a year.

We’ll be using the `SHAP <https://github.com/slundberg/shap>`__ library
to perform visual analysis. The library contains the dataset pre-loaded
which we will utilize here.

.. code:: ipython3

    !python -m pip install shap

.. code:: ipython3

    import shap
    X, y = shap.datasets.adult()
    X_display, y_display = shap.datasets.adult(display=True)
    feature_names = list(X.columns)

.. code:: ipython3

    # create a train/test split
    from sklearn.model_selection import train_test_split   # For splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)
    X_train_display = X_display.loc[X_train.index]

.. code:: ipython3

    train = pd.concat([pd.Series(y_train, index=X_train.index,
                                 name='Income>50K', dtype=int), X_train], axis=1)
    test = pd.concat([pd.Series(y_test, index=X_test.index,
                                name='Income>50K', dtype=int), X_test], axis=1)
    
    # Use 'csv' format to store the data
    # The first column is expected to be the output column
    train.to_csv('train.csv', index=False, header=False)
    test.to_csv('validation.csv', index=False, header=False)
    
    boto3.Session().resource('s3').Bucket(bucket).Object(
        os.path.join(prefix, 'data/train.csv')).upload_file('train.csv')
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(
        prefix, 'data/validation.csv')).upload_file('validation.csv')

Section 3 - Train XGBoost model in Amazon SageMaker with debugger enabled. 
---------------------------------------------------------------------------

Now train an XGBoost model with Amazon SageMaker Debugger enabled and
monitor the training jobs. This is done using the Amazon SageMaker
Estimator API. While the training job is running, use Amazon SageMaker
Debugger API to access saved tensors in real time and visualize them.
You can rely on Amazon SageMaker Debugger to take care of downloading a
fresh set of tensors every time you query for them.

Amazon SageMaker Debugger is available in Amazon SageMaker XGBoost
container version 0.90-2 or later. If you want to use XGBoost with
Amazon SageMaker Debugger, you have to specify ``repo_version='0.90-2'``
in the ``get_image_uri`` function.

3.1 Install the ‘smdebug’ open source library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !python -m pip install smdebug 

3.2 Build the XGBoost container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(region, "xgboost", repo_version="0.90-2")

.. code:: ipython3

    base_job_name = "demo-smdebug-xgboost-adult-income-prediction-classification"
    bucket_path = 's3://{}'.format(bucket)
    
    hyperparameters = {
        "max_depth": "5",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.7",
        "silent": "0",
        "objective": "binary:logistic",
        "num_round": "51",
    }
    save_interval = 5

3.3 Enabling Debugger in Estimator object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DebuggerHookConfig
^^^^^^^^^^^^^^^^^^

Enabling Amazon SageMaker Debugger in training job can be accomplished
by adding its configuration into Estimator object constructor:

.. code:: python

   from sagemaker.debugger import DebuggerHookConfig, CollectionConfig

   estimator = Estimator(
       ...,
       debugger_hook_config = DebuggerHookConfig(
           s3_output_path="s3://{bucket_name}/{location_in_bucket}",  # Required
           collection_configs=[
               CollectionConfig(
                   name="metrics",
                   parameters={
                       "save_interval": "10"
                   }
               )
           ]
       )
   )

Here, the ``DebuggerHookConfig`` object instructs ``Estimator`` what
data we are interested in. Two parameters are provided in the example:

-  ``s3_output_path``: it points to S3 bucket/path where we intend to
   store our debugging tensors. Amount of data saved depends on multiple
   factors, major ones are: training job / data set / model / frequency
   of saving tensors. This bucket should be in your AWS account, and you
   should have full access control over it. **Important Note**: this s3
   bucket should be originally created in the same region where your
   training job will be running, otherwise you might run into problems
   with cross region access.

-  ``collection_configs``: it enumerates named collections of tensors we
   want to save. Collections are a convinient way to organize relevant
   tensors under same umbrella to make it easy to navigate them during
   analysis. In this particular example, you are instructing Amazon
   SageMaker Debugger that you are interested in a single collection
   named ``metrics``. We also instructed Amazon SageMaker Debugger to
   save metrics every 10 iteration. See
   `Collection <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#collection>`__
   documentation for all parameters that are supported by Collections
   and DebuggerConfig documentation for more details about all
   parameters DebuggerConfig supports.

Rules
^^^^^

Enabling Rules in training job can be accomplished by adding the
``rules`` configuration into Estimator object constructor.

-  ``rules``: This new parameter will accept a list of rules you wish to
   evaluate against the tensors output by this training job. For rules,
   Amazon SageMaker Debugger supports two types:

   -  SageMaker Rules: These are rules specially curated by the data
      science and engineering teams in Amazon SageMaker which you can
      opt to evaluate against your training job.
   -  Custom Rules: You can optionally choose to write your own rule as
      a Python source file and have it evaluated against your training
      job. To provide Amazon SageMaker Debugger to evaluate this rule,
      you would have to provide the S3 location of the rule source and
      the evaluator image.

In this example, you will use a Amazon SageMaker’s LossNotDecreasing
rule, which helps you identify if you are running into a situation where
the training loss is not going down.

.. code:: python

   from sagemaker.debugger import rule_configs, Rule

   estimator = Estimator(
       ...,
       rules=[
           Rule.sagemaker(
               rule_configs.loss_not_decreasing(),
               rule_parameters={
                   "collection_names": "metrics",
                   "num_steps": "10",
               },
           ),
       ],
   )

-  ``rule_parameters``: In this parameter, you provide the runtime
   values of the parameter in your constructor. You can still choose to
   pass in other values which may be necessary for your rule to be
   evaluated. In this example, you will use Amazon SageMaker’s
   LossNotDecreasing rule to monitor the ``metircs`` collection. The
   rule will alert you if the tensors in ``metrics`` has not decreased
   for more than 10 steps.

.. code:: ipython3

    from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig, CollectionConfig
    from sagemaker.estimator import Estimator
    
    xgboost_estimator = Estimator(
        role=role,
        base_job_name=base_job_name,
        train_instance_count=1,
        train_instance_type='ml.m5.4xlarge',
        image_name=container,
        hyperparameters=hyperparameters,
        train_max_run=1800,
    
        debugger_hook_config=DebuggerHookConfig(
            s3_output_path=bucket_path,  # Required
            collection_configs=[
                CollectionConfig(
                    name="metrics",
                    parameters={
                        "save_interval": str(save_interval)
                    }
                ),
                CollectionConfig(
                    name="feature_importance",
                    parameters={
                        "save_interval": str(save_interval)
                    }
                ),
                CollectionConfig(
                    name="full_shap",
                    parameters={
                        "save_interval": str(save_interval)
                    }
                ),
                CollectionConfig(
                    name="average_shap",
                    parameters={
                        "save_interval": str(save_interval)
                    }
                ),
            ],
        ),
    
        rules=[
            Rule.sagemaker(
                rule_configs.loss_not_decreasing(),
                rule_parameters={
                    "collection_names": "metrics",
                    "num_steps": str(save_interval * 2),
                },
            ),
        ],
    )

With the next step, start a training job by using the Estimator object
you created above. This job is started in an asynchronous, non-blocking
way. This means that control is passed back to the notebook and further
commands can be run while the training job is progressing.

.. code:: ipython3

    from sagemaker.session import s3_input
    
    train_input = s3_input("s3://{}/{}/{}".format(bucket,
                                                  prefix, "data/train.csv"), content_type="csv")
    validation_input = s3_input(
        "s3://{}/{}/{}".format(bucket, prefix, "data/validation.csv"), content_type="csv")
    xgboost_estimator.fit(
        {"train": train_input, "validation": validation_input},
        # This is a fire and forget event. By setting wait=False, you submit the job to run in the background.
        # Amazon SageMaker starts one training job and release control to next cells in the notebook.
        # Follow this notebook to see status of the training job.
        wait=False
    )

3.4 Result
~~~~~~~~~~

As a result of the above command, Amazon SageMaker starts **one training
job and one rule job** for you. The first one is the job that produces
the tensors to be analyzed. The second one analyzes the tensors to check
if ``train-error`` and ``validation-error`` are not decreasing at any
point during training.

Check the status of the training job below. After your training job is
started, Amazon SageMaker starts a rule-execution job to run the
LossNotDecreasing rule.

The cell below will block till the training job is complete.

.. code:: ipython3

    import time
    
    for _ in range(36):
        job_name = xgboost_estimator.latest_training_job.name
        client = xgboost_estimator.sagemaker_session.sagemaker_client
        description = client.describe_training_job(TrainingJobName=job_name)
        training_job_status = description["TrainingJobStatus"]
        rule_job_summary = xgboost_estimator.latest_training_job.rule_job_summary()
        rule_evaluation_status = rule_job_summary[0]["RuleEvaluationStatus"]
        print("Training job status: {}, Rule Evaluation Status: {}".format(training_job_status, rule_evaluation_status))
        
        if training_job_status in ["Completed", "Failed"]:
            break
    
        time.sleep(10)

3.5 Check the status of the Rule Evaluation Job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the rule evaluation job that Amazon SageMaker started for you,
run the command below. The results show you the
``RuleConfigurationName``, ``RuleEvaluationJobArn``,
``RuleEvaluationStatus``, ``StatusDetails``, and
``RuleEvaluationJobArn``. If the tensors meets a rule evaluation
condition, the rule execution job throws a client error with
``RuleEvaluationConditionMet``.

The logs of the rule evaluation job are available in the Cloudwatch
Logstream ``/aws/sagemaker/ProcessingJobs`` with
``RuleEvaluationJobArn``.

You can see that once the rule execution job starts, it identifies the
loss not decreasing situation in the training job, it raises the
``RuleEvaluationConditionMet`` exception, and it ends the job.

.. code:: ipython3

    xgboost_estimator.latest_training_job.rule_job_summary()

Section 4 - Analyze debugger output 
------------------------------------

Now that you’ve trained the system, analyze the data. Here, you focus on
after-the-fact analysis.

You import a basic analysis library, which defines the concept of trial,
which represents a single training run.

Retrieving and Analyzing tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before getting to analysis, here are some notes on concepts being used
in Amazon SageMaker Debugger that help with analysis. - **Trial** -
Object that is a centerpiece of the SageMaker Debugger API when it comes
to getting access to tensors. It is a top level abstract that represents
a single run of a training job. All tensors emitted by a training job
are associated with its *trial*. - **Step** - Object that represents
next level of abstraction. In SageMaker Debugger, *step* is a
representation of a single batch of a training job. Each trial has
multiple steps. Each tensor is associated with multiple steps and has a
particular value at each of the steps. - **Tensor** - object that
represent actual *tensor* saved during training job. *Note* - it could
be a scalar as well (for example, metrics are saved as scalars).

For more details on aforementioned concepts as well as on SageMaker
Debugger API in general (including examples) see `SageMaker Debugger
Analysis
API <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md>`__
documentation.

In the following code cell, use a **Trial** to access tensors. You can
do that by inspecting currently running training job and extract
necessary parameters from its debug configuration to instruct SageMaker
Debugger where the data you are looking for is located. Keep in mind the
following: - Tensors are being stored in your own S3 bucket to which you
can navigate and manually inspect its content if desired. - You might
notice a slight delay before trial object is created. This is normal as
SageMaker Debugger monitors the corresponding bucket with tensors and
waits until tensors appear in it. The delay is introduced by less than
instantaneous upload of tensors from a training container to your S3
bucket.

.. code:: ipython3

    from smdebug.trials import create_trial
    
    s3_output_path = xgboost_estimator.latest_job_debugger_artifacts_path()
    trial = create_trial(s3_output_path)

You can list all the tensors that you know something about. Each one of
these names is the name of a tensor. The name is a combination of the
feature name, which in these cases, is auto-assigned by XGBoost, and
whether it’s an evaluation metric, feature importance, or SHAP value.

.. code:: ipython3

    trial.tensor_names()

For each tensor, for each step we can get the values at that step.

.. code:: ipython3

    trial.tensor("average_shap/f1").values()

Plot Performance metrics
~~~~~~~~~~~~~~~~~~~~~~~~

You can also create a simple function that visualizes the training and
validation errors as the training progresses. The error should get
smaller over time, as the system converges to a good solution.

.. code:: ipython3

    from itertools import islice
    import matplotlib.pyplot as plt
    import re
    
    MAX_PLOTS = 35
    
    
    def get_data(trial, tname):
        """
        For the given tensor name, walks though all the iterations
        for which you have data and fetches the values.
        Returns the set of steps and the values.
        """
        tensor = trial.tensor(tname)
        steps = tensor.steps()
        vals = [tensor.value(s) for s in steps]
        return steps, vals
    
    
    def plot_collection(trial, collection_name, regex='.*', figsize=(8, 6)):
        """
        Takes a `trial` and a collection name, and 
        plots all tensors that match the given regex.
        """
        fig, ax = plt.subplots(figsize=figsize)
        tensors = sorted(trial.collection(collection_name).tensor_names)
        matched_tensors = [t for t in tensors if re.match(regex, t)]
        for tensor_name in islice(matched_tensors, MAX_PLOTS):
            steps, data = get_data(trial, tensor_name)
            ax.plot(steps, data, label=tensor_name)
    
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Iteration')

.. code:: ipython3

    plot_collection(trial, "metrics")

Feature importance
~~~~~~~~~~~~~~~~~~

You can also visualize the feature priorities as determined by
`xgboost.get_score() <https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score>`__.
If you instructed Estimator to log the ``feature_importance``
collection, all importance types supported by ``xgboost.get_score()``
will be available in the collection.

.. code:: ipython3

    def plot_feature_importance(trial, importance_type="weight"):
        SUPPORTED_IMPORTANCE_TYPES = [
            "weight", "gain", "cover", "total_gain", "total_cover"]
        if importance_type not in SUPPORTED_IMPORTANCE_TYPES:
            raise ValueError(
                f"{importance_type} is not one of the supported importance types.")
        plot_collection(
            trial,
            "feature_importance",
            regex=f"feature_importance/{importance_type}/.*")

.. code:: ipython3

    plot_feature_importance(trial, importance_type="cover")

SHAP
~~~~

`SHAP <https://github.com/slundberg/shap>`__ (SHapley Additive
exPlanations) is another approach to explain the output of machine
learning models. SHAP values represent a feature’s contribution to a
change in the model output. You instructed Estimator to log the average
SHAP values in this example so the SHAP values (as calculated by
`xgboost.predict(pred_contribs=True) <https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.predict>`__)
will be available the ``average_shap`` collection.

.. code:: ipython3

    plot_collection(trial, "average_shap")

Global explanations
~~~~~~~~~~~~~~~~~~~

Global explanatory methods allow understanding the model and its feature
contributions in aggregate over multiple datapoints. Here we show an
aggregate bar plot that plots the mean absolute SHAP value for each
feature.

Specifically, the below plot indicates that the value of relationship
(Wife=5, Husband=4, Own-child=3, Other-relative=2, Unmarried=1,
Not-in-family=0) plays the most important role in predicting the income
probability being higher than 50K.

.. code:: ipython3

    shap_values = trial.tensor("full_shap/f0").value(trial.last_complete_step)
    shap_no_base = shap_values[:, :-1]
    shap_base_value = shap_values[0, -1]
    shap.summary_plot(shap_no_base, plot_type='bar', feature_names=feature_names)

The detailed summary plot below can provide more context over the above
bar chart. It tells which features are most important and, in addition,
their range of effects over the dataset. The color allows us to match
how changes in the value of a feature effect the change in prediction.

The ‘red’ indicates higher value of the feature and ‘blue’ indicates
lower (normalized over the features). This allows conclusions such as
’increase in age leads to higher log odds for prediction, eventually
leading to ``True`` predictions more often.

.. code:: ipython3

    shap.summary_plot(shap_no_base, X_train)

Local explanations
~~~~~~~~~~~~~~~~~~

Local explainability aims to explain model behavior for a fixed input
point. This can be used for either auditing models before deployment or
to provide explanations for specific inference predictions.

.. code:: ipython3

    shap.initjs()

Force plot
^^^^^^^^^^

A force plot explanation shows how features are contributing to push the
model output from the base value (the average model output over the
dataset) to the model output. Features pushing the prediction higher are
shown in **red**, those pushing the prediction lower are in **blue**.

Plot below indicates that for this particular data point the prediction
probability (0.48) is higher than the average (~0.2) primarily because
this person is in a relationship (``Relationship = Wife``), and to
smaller degree because of the higher-than-average age. Similarly the
model reduces the probability due specific ``Sex`` and ``Race`` values
indicating existence of bias in model behavior (possibly due to bias in
the data).

.. code:: ipython3

    shap.force_plot(shap_base_value, shap_no_base[100, :],
                    X_train_display.iloc[100, :], link="logit", matplotlib=True)

Stacked force plot
^^^^^^^^^^^^^^^^^^

SHAP allows stacking multiple force-plots after rotating 90 degress to
understand the explanations for multiple datapoints. If Javascript is
enabled, then in the notebook this plot is interactive, allowing
understanding the change in output for each feature independently. This
stacking of force plots provides a balance between local and global
explainability.

.. code:: ipython3

    import numpy as np
    N_ROWS = shap_no_base.shape[0]
    N_SAMPLES = min(100, N_ROWS)
    sampled_indices = np.random.randint(N_ROWS, size=N_SAMPLES)

.. code:: ipython3

    shap.force_plot(shap_base_value,
                    shap_no_base[sampled_indices, :],
                    X_train_display.iloc[sampled_indices, :],
                    link='logit')

Outliers
~~~~~~~~

Outliers are extreme values that deviate from other observations on
data. It’s useful to understand the influence of various features for
outlier predictions to determine if it’s a novelty, an experimental
error, or a shortcoming in the model.

Here we show force plot for prediction outliers that are on either side
of the baseline value.

.. code:: ipython3

    # top outliers
    from scipy import stats
    N_OUTLIERS = 3  # number of outliers on each side of the tail
    
    shap_sum = np.sum(shap_no_base, axis=1)
    z_scores = stats.zscore(shap_sum)
    outlier_indices = (np.argpartition(z_scores, -N_OUTLIERS)
                       [-N_OUTLIERS:]).tolist()
    outlier_indices += (np.argpartition(z_scores, N_OUTLIERS)
                        [:N_OUTLIERS]).tolist()

.. code:: ipython3

    for fig_index, outlier_index in enumerate(outlier_indices, start=1):
        shap.force_plot(shap_base_value,
                        shap_no_base[outlier_index, :],
                        X_train_display.iloc[outlier_index, :],
                        matplotlib=True,
                        link='logit')

Conclusion
----------

This notebook discussed the importance of explainability for improved ML
adoption and. We introduced the Amazon SageMaker Debugger capability
with built-in tensor collections to enable model explainability. The
notebook walked you through training an ML model for a financial
services use case of individual income prediction. We further analyzed
the global and local explanations of the model by visualizing the
captured tensors.
