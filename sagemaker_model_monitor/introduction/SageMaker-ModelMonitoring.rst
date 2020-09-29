Amazon SageMaker Model Monitor
==============================

This notebook shows how to: \* Host a machine learning model in Amazon
SageMaker and capture inference requests, results, and metadata \*
Analyze a training dataset to generate baseline constraints \* Monitor a
live endpoint for violations against constraints

Setup
-----

To get started, make sure you have these prerequisites completed.

-  Specify an AWS Region to host your model.
-  An IAM role ARN exists that is used to give Amazon SageMaker access
   to your data in Amazon Simple Storage Service (Amazon S3). See the
   documentation for how to fine tune the permissions needed.
-  Create an S3 bucket used to store the data used to train your model,
   any additional model data, and the data captured from model
   invocations. For demonstration purposes, you are using the same
   bucket for these. In reality, you might want to separate them with
   different security policies.

.. code:: ipython3

    %%time
    
    # Handful of configuration
    
    import os
    import boto3
    import re
    import json
    from sagemaker import get_execution_role, session
    
    region= boto3.Session().region_name
    
    role = get_execution_role()
    print("RoleArn: {}".format(role))
    
    # You can use a different bucket, but make sure the role you chose for this notebook
    # has the s3:PutObject permissions. This is the bucket into which the data is captured
    bucket =  session.Session(boto3.Session()).default_bucket()
    print("Demo Bucket: {}".format(bucket))
    prefix = 'sagemaker/DEMO-ModelMonitor'
    
    data_capture_prefix = '{}/datacapture'.format(prefix)
    s3_capture_upload_path = 's3://{}/{}'.format(bucket, data_capture_prefix)
    reports_prefix = '{}/reports'.format(prefix)
    s3_report_path = 's3://{}/{}'.format(bucket,reports_prefix)
    code_prefix = '{}/code'.format(prefix)
    s3_code_preprocessor_uri = 's3://{}/{}/{}'.format(bucket,code_prefix, 'preprocessor.py')
    s3_code_postprocessor_uri = 's3://{}/{}/{}'.format(bucket,code_prefix, 'postprocessor.py')
    
    print("Capture path: {}".format(s3_capture_upload_path))
    print("Report path: {}".format(s3_report_path))
    print("Preproc Code path: {}".format(s3_code_preprocessor_uri))
    print("Postproc Code path: {}".format(s3_code_postprocessor_uri))

You can quickly verify that the execution role for this notebook has the
necessary permissions to proceed. Put a simple test object into the S3
bucket you speciﬁed above. If this command fails, update the role to
have ``s3:PutObject`` permission on the bucket and try again.

.. code:: ipython3

    # Upload some test files
    boto3.Session().resource('s3').Bucket(bucket).Object("test_upload/test.txt").upload_file('test_data/upload-test-file.txt')
    print("Success! You are all set to proceed.")

PART A: Capturing real-time inference data from Amazon SageMaker endpoints
==========================================================================

Create an endpoint to showcase the data capture capability in action.

Upload the pre-trained model to Amazon S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code uploads a pre-trained XGBoost model that is ready for you to
deploy. This model was trained using the XGB Churn Prediction Notebook
in SageMaker. You can also use your own pre-trained model in this step.
If you already have a pretrained model in Amazon S3, you can add it
instead by specifying the s3_key.

.. code:: ipython3

    model_file = open("model/xgb-churn-prediction-model.tar.gz", 'rb')
    s3_key = os.path.join(prefix, 'xgb-churn-prediction-model.tar.gz')
    boto3.Session().resource('s3').Bucket(bucket).Object(s3_key).upload_fileobj(model_file)

Deploy the model to Amazon SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with deploying a pre-trained churn prediction model. Here, you
create the model object with the image and model data.

.. code:: ipython3

    from time import gmtime, strftime
    from sagemaker.model import Model
    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    model_name = "DEMO-xgb-churn-pred-model-monitor-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    model_url = 'https://{}.s3-{}.amazonaws.com/{}/xgb-churn-prediction-model.tar.gz'.format(bucket, region, prefix)
    image_uri = get_image_uri(boto3.Session().region_name, 'xgboost', '0.90-1')
    
    model = Model(image=image_uri, model_data=model_url, role=role)

To enable data capture for monitoring the model data quality, you
specify the new capture option called ``DataCaptureConfig``. You can
capture the request payload, the response payload or both with this
configuration. The capture config applies to all variants. Go ahead with
the deployment.

.. code:: ipython3

    from sagemaker.model_monitor import DataCaptureConfig
    
    endpoint_name = 'DEMO-xgb-churn-pred-model-monitor-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    print("EndpointName={}".format(endpoint_name))
    
    data_capture_config = DataCaptureConfig(
                            enable_capture=True,
                            sampling_percentage=100,
                            destination_s3_uri=s3_capture_upload_path)
    
    predictor = model.deploy(initial_instance_count=1,
                    instance_type='ml.m4.xlarge',
                    endpoint_name=endpoint_name,
                    data_capture_config=data_capture_config)

Invoke the deployed model
-------------------------

You can now send data to this endpoint to get inferences in real time.
Because you enabled the data capture in the previous steps, the request
and response payload, along with some additional metadata, is saved in
the Amazon Simple Storage Service (Amazon S3) location you have
specified in the DataCaptureConfig.

This step invokes the endpoint with included sample data for about 2
minutes. Data is captured based on the sampling percentage specified and
the capture continues until the data capture option is turned off.

.. code:: ipython3

    from sagemaker.predictor import RealTimePredictor
    import time
    
    predictor = RealTimePredictor(endpoint=endpoint_name,content_type='text/csv')
    
    # get a subset of test data for a quick test
    !head -120 test_data/test-dataset-input-cols.csv > test_data/test_sample.csv
    print("Sending test traffic to the endpoint {}. \nPlease wait...".format(endpoint_name))
    
    with open('test_data/test_sample.csv', 'r') as f:
        for row in f:
            payload = row.rstrip('\n')
            response = predictor.predict(data=payload)
            time.sleep(0.5)
            
    print("Done!")        

View captured data
------------------

Now list the data capture files stored in Amazon S3. You should expect
to see different files from different time periods organized based on
the hour in which the invocation occurred. The format of the Amazon S3
path is:

``s3://{destination-bucket-prefix}/{endpoint-name}/{variant-name}/yyyy/mm/dd/hh/filename.jsonl``

.. code:: ipython3

    s3_client = boto3.Session().client('s3')
    current_endpoint_capture_prefix = '{}/{}'.format(data_capture_prefix, endpoint_name)
    result = s3_client.list_objects(Bucket=bucket, Prefix=current_endpoint_capture_prefix)
    capture_files = [capture_file.get("Key") for capture_file in result.get('Contents')]
    print("Found Capture Files:")
    print("\n ".join(capture_files))

Next, view the contents of a single capture file. Here you should see
all the data captured in an Amazon SageMaker specific JSON-line
formatted file. Take a quick peek at the first few lines in the captured
file.

.. code:: ipython3

    def get_obj_body(obj_key):
        return s3_client.get_object(Bucket=bucket, Key=obj_key).get('Body').read().decode("utf-8")
    
    capture_file = get_obj_body(capture_files[-1])
    print(capture_file[:2000])

Finally, the contents of a single line is present below in a formatted
JSON file so that you can observe a little better.

.. code:: ipython3

    import json
    print(json.dumps(json.loads(capture_file.split('\n')[0]), indent=2))

As you can see, each inference request is captured in one line in the
jsonl file. The line contains both the input and output merged together.
In the example, you provided the ContentType as ``text/csv`` which is
reflected in the ``observedContentType`` value. Also, you expose the
encoding that you used to encode the input and output payloads in the
capture format with the ``encoding`` value.

To recap, you observed how you can enable capturing the input or output
payloads to an endpoint with a new parameter. You have also observed
what the captured format looks like in Amazon S3. Next, continue to
explore how Amazon SageMaker helps with monitoring the data collected in
Amazon S3.

PART B: Model Monitor - Baselining and continuous monitoring
============================================================

In addition to collecting the data, Amazon SageMaker provides the
capability for you to monitor and evaluate the data observed by the
endpoints. For this: 1. Create a baseline with which you compare the
realtime traffic. 1. Once a baseline is ready, setup a schedule to
continously evaluate and compare against the baseline.

1. Constraint suggestion with baseline/training dataset
-------------------------------------------------------

The training dataset with which you trained the model is usually a good
baseline dataset. Note that the training dataset data schema and the
inference dataset schema should exactly match (i.e. the number and order
of the features).

From the training dataset you can ask Amazon SageMaker to suggest a set
of baseline ``constraints`` and generate descriptive ``statistics`` to
explore the data. For this example, upload the training dataset that was
used to train the pre-trained model included in this example. If you
already have it in Amazon S3, you can directly point to it.

.. code:: ipython3

    # copy over the training dataset to Amazon S3 (if you already have it in Amazon S3, you could reuse it)
    baseline_prefix = prefix + '/baselining'
    baseline_data_prefix = baseline_prefix + '/data'
    baseline_results_prefix = baseline_prefix + '/results'
    
    baseline_data_uri = 's3://{}/{}'.format(bucket,baseline_data_prefix)
    baseline_results_uri = 's3://{}/{}'.format(bucket, baseline_results_prefix)
    print('Baseline data uri: {}'.format(baseline_data_uri))
    print('Baseline results uri: {}'.format(baseline_results_uri))


.. code:: ipython3

    training_data_file = open("test_data/training-dataset-with-header.csv", 'rb')
    s3_key = os.path.join(baseline_prefix, 'data', 'training-dataset-with-header.csv')
    boto3.Session().resource('s3').Bucket(bucket).Object(s3_key).upload_fileobj(training_data_file)

Create a baselining job with training dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have the training data ready in Amazon S3, start a job to
``suggest`` constraints. ``DefaultModelMonitor.suggest_baseline(..)``
starts a ``ProcessingJob`` using an Amazon SageMaker provided Model
Monitor container to generate the constraints.

.. code:: ipython3

    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    
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

    s3_client = boto3.Session().client('s3')
    result = s3_client.list_objects(Bucket=bucket, Prefix=baseline_results_prefix)
    report_files = [report_file.get("Key") for report_file in result.get('Contents')]
    print("Found Files:")
    print("\n ".join(report_files))

.. code:: ipython3

    import pandas as pd
    
    baseline_job = my_default_monitor.latest_baselining_job
    schema_df = pd.io.json.json_normalize(baseline_job.baseline_statistics().body_dict["features"])
    schema_df.head(10)

.. code:: ipython3

    constraints_df = pd.io.json.json_normalize(baseline_job.suggested_constraints().body_dict["features"])
    constraints_df.head(10)

2. Analyzing collected data for data quality issues
---------------------------------------------------

When you have collected the data above, analyze and monitor the data
with Monitoring Schedules

Create a schedule
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # First, copy over some test scripts to the S3 bucket so that they can be used for pre and post processing
    boto3.Session().resource('s3').Bucket(bucket).Object(code_prefix+"/preprocessor.py").upload_file('preprocessor.py')
    boto3.Session().resource('s3').Bucket(bucket).Object(code_prefix+"/postprocessor.py").upload_file('postprocessor.py')

You can create a model monitoring schedule for the endpoint created
earlier. Use the baseline resources (constraints and statistics) to
compare against the realtime traffic.

.. code:: ipython3

    from sagemaker.model_monitor import CronExpressionGenerator
    from time import gmtime, strftime
    
    mon_schedule_name = 'DEMO-xgb-churn-pred-model-monitor-schedule-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    my_default_monitor.create_monitoring_schedule(
        monitor_schedule_name=mon_schedule_name,
        endpoint_input=predictor.endpoint,
        #record_preprocessor_script=pre_processor_script,
        post_analytics_processor_script=s3_code_postprocessor_uri,
        output_s3_uri=s3_report_path,
        statistics=my_default_monitor.baseline_statistics(),
        constraints=my_default_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    
    )

Start generating some artificial traffic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cell below starts a thread to send some traffic to the endpoint.
Note that you need to stop the kernel to terminate this thread. If there
is no traffic, the monitoring jobs are marked as ``Failed`` since there
is no data to process.

.. code:: ipython3

    from threading import Thread
    from time import sleep
    import time
    
    endpoint_name=predictor.endpoint
    runtime_client = boto3.client('runtime.sagemaker')
    
    # (just repeating code from above for convenience/ able to run this section independently)
    def invoke_endpoint(ep_name, file_name, runtime_client):
        with open(file_name, 'r') as f:
            for row in f:
                payload = row.rstrip('\n')
                response = runtime_client.invoke_endpoint(EndpointName=ep_name,
                                              ContentType='text/csv', 
                                              Body=payload)
                response['Body'].read()
                time.sleep(1)
                
    def invoke_endpoint_forever():
        while True:
            invoke_endpoint(endpoint_name, 'test_data/test-dataset-input-cols.csv', runtime_client)
            
    thread = Thread(target = invoke_endpoint_forever)
    thread.start()
    
    # Note that you need to stop the kernel to stop the invocations

Describe and inspect the schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you describe, observe that the MonitoringScheduleStatus changes to
Scheduled.

.. code:: ipython3

    desc_schedule_result = my_default_monitor.describe_schedule()
    print('Schedule status: {}'.format(desc_schedule_result['MonitoringScheduleStatus']))

List executions
~~~~~~~~~~~~~~~

The schedule starts jobs at the previously specified intervals. Here,
you list the latest five executions. Note that if you are kicking this
off after creating the hourly schedule, you might find the executions
empty. You might have to wait until you cross the hour boundary (in UTC)
to see executions kick off. The code below has the logic for waiting.

Note: Even for an hourly schedule, Amazon SageMaker has a buffer period
of 20 minutes to schedule your execution. You might see your execution
start in anywhere from zero to ~20 minutes from the hour boundary. This
is expected and done for load balancing in the backend.

.. code:: ipython3

    mon_executions = my_default_monitor.list_executions()
    print("We created a hourly schedule above and it will kick off executions ON the hour (plus 0 - 20 min buffer.\nWe will have to wait till we hit the hour...")
    
    while len(mon_executions) == 0:
        print("Waiting for the 1st execution to happen...")
        time.sleep(60)
        mon_executions = my_default_monitor.list_executions()    

Inspect a specific execution (latest execution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous cell, you picked up the latest completed or failed
scheduled execution. Here are the possible terminal states and what each
of them mean: \* Completed - This means the monitoring execution
completed and no issues were found in the violations report. \*
CompletedWithViolations - This means the execution completed, but
constraint violations were detected. \* Failed - The monitoring
execution failed, maybe due to client error (perhaps incorrect role
premissions) or infrastructure issues. Further examination of
FailureReason and ExitMessage is necessary to identify what exactly
happened. \* Stopped - job exceeded max runtime or was manually stopped.

.. code:: ipython3

    latest_execution = mon_executions[-1] # latest execution's index is -1, second to last is -2 and so on..
    time.sleep(60)
    latest_execution.wait(logs=False)
    
    print("Latest execution status: {}".format(latest_execution.describe()['ProcessingJobStatus']))
    print("Latest execution result: {}".format(latest_execution.describe()['ExitMessage']))
    
    latest_job = latest_execution.describe()
    if (latest_job['ProcessingJobStatus'] != 'Completed'):
            print("====STOP==== \n No completed executions to inspect further. Please wait till an execution completes or investigate previously reported failures.")

.. code:: ipython3

    report_uri=latest_execution.output.destination
    print('Report Uri: {}'.format(report_uri))

List the generated reports
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from urllib.parse import urlparse
    s3uri = urlparse(report_uri)
    report_bucket = s3uri.netloc
    report_key = s3uri.path.lstrip('/')
    print('Report bucket: {}'.format(report_bucket))
    print('Report key: {}'.format(report_key))
    
    s3_client = boto3.Session().client('s3')
    result = s3_client.list_objects(Bucket=report_bucket, Prefix=report_key)
    report_files = [report_file.get("Key") for report_file in result.get('Contents')]
    print("Found Report Files:")
    print("\n ".join(report_files))

Violations report
~~~~~~~~~~~~~~~~~

If there are any violations compared to the baseline, they will be
listed here.

.. code:: ipython3

    violations = my_default_monitor.latest_monitoring_constraint_violations()
    pd.set_option('display.max_colwidth', -1)
    constraints_df = pd.io.json.json_normalize(violations.body_dict["violations"])
    constraints_df.head(10)

Other commands
~~~~~~~~~~~~~~

We can also start and stop the monitoring schedules.

.. code:: ipython3

    #my_default_monitor.stop_monitoring_schedule()
    #my_default_monitor.start_monitoring_schedule()

Delete the resources
--------------------

You can keep your endpoint running to continue capturing data. If you do
not plan to collect more data or use this endpoint further, you should
delete the endpoint to avoid incurring additional charges. Note that
deleting your endpoint does not delete the data that was captured during
the model invocations. That data persists in Amazon S3 until you delete
it yourself.

But before that, you need to delete the schedule first.

.. code:: ipython3

    my_default_monitor.delete_monitoring_schedule()
    time.sleep(60) # actually wait for the deletion

.. code:: ipython3

    predictor.delete_endpoint()

.. code:: ipython3

    predictor.delete_model()
