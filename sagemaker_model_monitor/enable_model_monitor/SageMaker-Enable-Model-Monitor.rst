Enable Amazon SageMaker Model Monitor
=====================================

Amazon SageMaker provides the ability to monitor machine learning models
in production and detect deviations in data quality in comparison to a
baseline dataset (e.g. training data set). This notebook walks you
through enabling data capture and setting up continous monitoring for an
existing Endpoint.

This Notebook helps with the following: \* Update your existing
SageMaker Endpoint to enable Model Monitoring \* Analyze the training
dataset to generate a baseline constraint \* Setup a MonitoringSchedule
for monitoring deviations from the specified baseline

--------------

Step 1: Enable real-time inference data capture
===============================================

To enable data capture for monitoring the model data quality, you
specify the new capture option called ``DataCaptureConfig``. You can
capture the request payload, the response payload or both with this
configuration. The capture config applies to all variants. Please
provide the Endpoint name in the following cell:

.. code:: ipython3

    # Please fill in the following for enabling data capture
    endpoint_name = 'FILL-IN-HERE-YOUR-ENDPOINT-NAME'
    s3_capture_upload_path = 'FILL-IN-HERE-YOUR-S3-BUCKET-PREFIX-HERE' #example: s3://bucket-name/path/to/endpoint-data-capture/
    
    ##### 
    ## IMPORTANT
    ##
    ## Please make sure to add the "s3:PutObject" permission to the "role' you provided in the SageMaker Model 
    ## behind this Endpoint. Otherwise, Endpoint data capture will not work.
    ## 
    ##### 

.. code:: ipython3

    from sagemaker.model_monitor import DataCaptureConfig
    from sagemaker import RealTimePredictor
    from sagemaker import session
    import boto3
    sm_session = session.Session(boto3.Session())
    
    # Change parameters as you would like - adjust sampling percentage, 
    #  chose to capture request or response or both.
    #  Learn more from our documentation
    data_capture_config = DataCaptureConfig(
                            enable_capture = True,
                            sampling_percentage=50,
                            destination_s3_uri=s3_capture_upload_path,
                            kms_key_id=None,
                            capture_options=["REQUEST", "RESPONSE"],
                            csv_content_types=["text/csv"],
                            json_content_types=["application/json"])
    
    # Now it is time to apply the new configuration and wait for it to be applied
    predictor = RealTimePredictor(endpoint=endpoint_name)
    predictor.update_data_capture_config(data_capture_config=data_capture_config)
    sm_session.wait_for_endpoint(endpoint=endpoint_name)

Before you proceed:
-------------------

Currently SageMaker supports monitoring Endpoints out of the box only
for **tabular (csv, flat-json)** datasets. If your Endpoint uses some
other datasets, these following steps will NOT work for you.

Step 2: Model Monitor - Baselining
==================================

In addition to collecting the data, SageMaker allows you to monitor and
evaluate the data observed by the Endpoints. For this : 1. We need to
create a baseline with which we compare the realtime traffic against. 1.
Once a baseline is ready, we can setup a schedule to continously
evaluate/compare against the baseline.

Constraint suggestion with baseline/training dataset
----------------------------------------------------

The training dataset with which you trained the model is usually a good
baseline dataset. Note that the training dataset’s data schema and the
inference dataset schema should exactly match (i.e. number and order of
the features).

Using our training dataset, we’ll ask SageMaker to suggest a set of
baseline constraints and generate descriptive statistics to explore the
data.

.. code:: ipython3

    baseline_data_uri = 'FILL-ME-IN' ##'s3://bucketname/path/to/baseline/data' - Where your training data is
    baseline_results_uri = 'FILL-ME-IN' ##'s3://bucketname/path/to/baseline/data' - Where the results are to be stored in
    
    print('Baseline data uri: {}'.format(baseline_data_uri))
    print('Baseline results uri: {}'.format(baseline_results_uri))

Create a baselining job with the training dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have the training data ready in S3, let’s kick off a job to
``suggest`` constraints. ``DefaultModelMonitor.suggest_baseline(..)``
kicks off a ``ProcessingJob`` using a SageMaker provided Model Monitor
container to generate the constraints. Please edit the configurations to
fit your needs.

.. code:: ipython3

    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    
    my_default_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )
    
    my_default_monitor.suggest_baseline(
        baseline_dataset=baseline_data_uri+'/training-dataset-with-header.csv',
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results_uri,
        wait=True
    )

Explore the generated constraints and statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import pandas as pd
    
    baseline_job = my_default_monitor.latest_baselining_job
    schema_df = pd.io.json.json_normalize(baseline_job.baseline_statistics().body_dict["features"])
    schema_df.head(10)

.. code:: ipython3

    constraints_df = pd.io.json.json_normalize(baseline_job.suggested_constraints().body_dict["features"])
    constraints_df.head(10)

Before proceeding to enable monitoring, you could chose to edit the
constraint file as required to fine tune the constraints.

Step 3: Enable continous monitoring
===================================

We have collected the data above, here we proceed to analyze and monitor
the data with MonitoringSchedules.

Create a schedule
~~~~~~~~~~~~~~~~~

We are ready to create a model monitoring schedule for the Endpoint
created earlier with the baseline resources (constraints and
statistics).

.. code:: ipython3

    from sagemaker.model_monitor import CronExpressionGenerator
    from time import gmtime, strftime
    
    mon_schedule_name = 'FILL-IN-HERE'
    s3_report_path = 'FILL-IN-HERE'
    my_default_monitor.create_monitoring_schedule(
        monitor_schedule_name=mon_schedule_name,
        endpoint_input=predictor.endpoint,
        output_s3_uri=s3_report_path,
        statistics=my_default_monitor.baseline_statistics(),
        constraints=my_default_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=True,
    
    )

.. code:: ipython3

    desc_schedule_result = my_default_monitor.describe_schedule()
    print('Schedule status: {}'.format(desc_schedule_result['MonitoringScheduleStatus']))

All set
~~~~~~~

Now that your monitoring schedule has been created. Please return to the
Amazon SageMaker Studio to list the executions for this Schedule and
observe the results going forward.
