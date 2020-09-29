SageMaker Model Monitor - visualizing monitoring results
========================================================

The prebuilt container from SageMaker computes a variety of statistics
and evaluates constraints out of the box. This notebook demonstrates how
you can visualize them. You can grab the ProcessingJob arn from the
executions behind a MonitoringSchedule and use this notebook to
visualize the results.

Let’s import some python libraries that will be helpful for
visualization

.. code:: ipython3

    from IPython.display import HTML, display
    import json
    import os
    import boto3
    
    import sagemaker
    from sagemaker import session
    from sagemaker.model_monitor import MonitoringExecution
    from sagemaker.s3 import S3Downloader

Get Utilities for Rendering
---------------------------

The functions for plotting and rendering distribution statistics or
constraint violations are implemented in a ``utils`` file so let’s grab
that.

.. code:: ipython3

    !wget https://raw.githubusercontent.com/awslabs/amazon-sagemaker-examples/master/sagemaker_model_monitor/visualization/utils.py
    
    import utils as mu

Get Execution and Baseline details from Processing Job Arn
----------------------------------------------------------

Enter the ProcessingJob arn for an execution of a MonitoringSchedule
below to get the result files associated with that execution

.. code:: ipython3

    processing_job_arn = "FILL-IN-PROCESSING-JOB-ARN" 

.. code:: ipython3

    execution = MonitoringExecution.from_processing_arn(sagemaker_session=session.Session(), processing_job_arn=processing_job_arn)
    exec_inputs = {inp['InputName']: inp for inp in execution.describe()['ProcessingInputs']}
    exec_results = execution.output.destination

.. code:: ipython3

    baseline_statistics_filepath = exec_inputs['baseline']['S3Input']['S3Uri'] if 'baseline' in exec_inputs else None
    execution_statistics_filepath = os.path.join(exec_results, 'statistics.json')
    violations_filepath = os.path.join(exec_results, 'constraint_violations.json')
    
    baseline_statistics = json.loads(S3Downloader.read_file(baseline_statistics_filepath)) if baseline_statistics_filepath is not None else None
    execution_statistics = json.loads(S3Downloader.read_file(execution_statistics_filepath))
    violations = json.loads(S3Downloader.read_file(violations_filepath))['violations']

Overview
--------

The code below shows the violations and constraint checks across all
features in a simple table.

.. code:: ipython3

    mu.show_violation_df(baseline_statistics=baseline_statistics, latest_statistics=execution_statistics, violations=violations)

Distributions
-------------

This section visualizes the distribution and renders the distribution
statistics for all features

.. code:: ipython3

    features = mu.get_features(execution_statistics)
    feature_baselines = mu.get_features(baseline_statistics)

.. code:: ipython3

    mu.show_distributions(features)

Execution Stats vs Baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    mu.show_distributions(features, feature_baselines)
