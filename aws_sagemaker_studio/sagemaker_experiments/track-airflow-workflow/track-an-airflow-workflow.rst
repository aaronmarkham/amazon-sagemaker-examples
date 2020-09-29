Track an Airflow Workflow
=========================

This notebook uses `fashion-mnist
dataset <https://www.tensorflow.org/datasets/catalog/fashion_mnist>`__
classification task as an example to show how one can track Airflow
Workflow executions using Sagemaker Experiments.

Overall, the notebook is organized as follow:

1. Download dataset and upload to Amazon S3.
2. Create a simple CNN model to do the classification.
3. Define the workflow as a DAG with two executions, a SageMaker
   TrainingJob for training the CNN model, followed by a SageMaker
   TransformJob to run batch predictions on model.
4. Host and run the workflow locally, and track the workflow run as an
   Experiment.
5. List executions.

Note that if you are running the notebook in SageMaker Studio, please
select ``Python3 (Tensorflow CPU Optimized)`` Kernel; if you are running
in SageMaker Notebook, please select ``conda_tensorflow_py36`` kernel.

Setup
-----

.. code:: ipython3

    import sys
    import os
    
    # append source code directory
    sys.path.insert(0, os.path.abspath('./code'))

.. code:: ipython3

    !{sys.executable} -m pip uninstall -y enum34
    !{sys.executable} -m pip install werkzeug==0.15.4
    !{sys.executable} -m pip install apache-airflow
    !{sys.executable} -m pip install sagemaker-experiments
    !{sys.executable} -m pip install matplotlib


.. code:: ipython3

    import boto3
    import os
    import time
    from datetime import datetime
    
    import sagemaker
    from sagemaker.s3 import S3Uploader
    from sagemaker import get_execution_role
    from sagemaker.session import Session
    from smexperiments.experiment import Experiment
    from smexperiments.trial import Trial
    from smexperiments.trial_component import TrialComponent
    from sagemaker.analytics import ExperimentAnalytics
    
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    
    from model import get_model
    
    %config InlineBackend.figure_format = 'retina'
    import matplotlib.pyplot as plt

.. code:: ipython3

    sess = boto3.Session()
    sm = sess.client('sagemaker')
    sagemaker_sess = sagemaker.Session()
    role = get_execution_role()

Create a S3 bucket to hold data
-------------------------------

.. code:: ipython3

    # create a s3 bucket to hold data, note that your account might already created a bucket with the same name
    account_id = sess.client('sts').get_caller_identity()["Account"]
    bucket = 'sagemaker-experiments-{}-{}'.format(sess.region_name, account_id)
    prefix = 'fashion-mnist'
    
    try:
        if sess.region_name == "us-east-1":
            sess.client('s3').create_bucket(Bucket=bucket)
        else:
            sess.client('s3').create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={'LocationConstraint': sess.region_name}
            )
    except Exception as e:
        print(e)

Preparing dataset
-----------------

.. code:: ipython3

    # download the fashion-mnist dataset
    # the dataset will be downloaded to ~/.keras/datasets/fashion-mnist/
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

.. code:: ipython3

    # image example
    plt.imshow(x_train[9])

We will be creating a SageMaker Training Job and fitting by
``(x_train, y_train)``, and then a SageMaker Transform Job to perform
batch inference over a large-scale (10K) test data. To do the batch
inference, we need first flatten each sampl image (28x28) in ``x_test``
into an float array with 784 features, and then concatenate all
flattened samples into a ``csv`` file.

.. code:: ipython3

    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    np.savetxt('./x_test.csv', x_test_flat, delimiter=",")

Upload the dataset to s3
------------------------

.. code:: ipython3

    # upload training data to s3
    # you may need to modifiy the path to .keras dir
    train_input = S3Uploader.upload(
        local_path=f'{os.path.expanduser("~")}/.keras/datasets/fashion-mnist/', 
        desired_s3_uri=f"s3://{bucket}/{prefix}/data/train",
        session=sagemaker_sess,
    )
    print('train input spec: {}'.format(train_input))

.. code:: ipython3

    # upload test data to s3 for batch inference
    test_input = S3Uploader.upload(
        local_path='./x_test.csv', 
        desired_s3_uri=f"s3://{bucket}/{prefix}/data/test",
        session=sagemaker_sess,
    )
    print('test input spec: {}'.format(test_input))

Create a simple CNN
-------------------

The CNN we use in this example contains two consecutive (Conv2D -
MaxPool - Dropout) modules, followed by a feed-forward layer, and a
softmax layer to normalize the output into a valid probability
distribution.

.. code:: ipython3

    # use default parameters
    model = get_model()
    model.summary()

Create workflow configurations
------------------------------

For the purpose of demonstration, we will be executing our workflow
locally. Lets first create a dir under airflow root to store our DAGs.

.. code:: ipython3

    if not os.path.exists(os.path.expanduser('~/airflow')):
        # to generate airflow dir
        !airflow -h
    
    if not os.path.exists(os.path.expanduser('~/airflow/dags')):
        !mkdir {os.path.expanduser('~/airflow/dags')}

We will create an experiment named
``fashion-mnist-classification-experiment`` to track our workflow
execution first.

.. code:: ipython3

    experiment = Experiment.create(
        experiment_name=f"fashion-mnist-classification-experiment",
        description="An classification experiment on fashion mnist dataset using tensorflow framework."
    )

The following cell defines our DAG, which is a workflow with two steps.
One is running a training job on SageMaker, then followed by running a
transform job to perform batch inference on the fashion-mnist testset we
created before.

We will write the DAG defnition into the ``airflow/dags`` we just
created above.

.. code:: ipython3

    %%writefile ~/airflow/dags/fashion-mnist-dag.py
    import time
    
    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.tensorflow import TensorFlow
    from sagemaker.tensorflow.serving import Model
    from sagemaker.workflow.airflow import training_config, transform_config_from_estimator
    
    import airflow
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator
    
    experiment_name = "fashion-mnist-classification-experiment"
    
    sess = boto3.Session()
    account_id = sess.client('sts').get_caller_identity()["Account"]
    bucket_name = 'sagemaker-experiments-{}-{}'.format(sess.region_name, account_id)
    
    # for training job
    train_input = f"s3://{bucket_name}/fashion-mnist/data/train"
    # for batch transform job
    test_input = f"s3://{bucket_name}/fashion-mnist/data/test/x_test.csv"
    
    role = get_execution_role()
    
    base_job_name = 'fashion-mnist-cnn'
    
    py_version = 'py3'
    tf_framework_version = '1.13'
    
    # callable for SageMaker training in TensorFlow
    def train(data, **context):
        estimator = TensorFlow(
            base_job_name=base_job_name,
            source_dir="code",
            entry_point='train.py',
            role=role,
            framework_version=tf_framework_version,
            py_version=py_version,
            hyperparameters={
                'epochs': 10, 
                'batch-size' : 256
            },
            train_instance_count=1, 
            train_instance_type="ml.m4.xlarge"
        )
        estimator.fit(data, experiment_config={"ExperimentName": experiment_name, "TrialComponentDisplayName": "Training"})
        return estimator.latest_training_job.job_name
    
    
    # callable for SageMaker batch transform
    def transform(data, **context):
        training_job = context['ti'].xcom_pull(task_ids='training')
        estimator = TensorFlow.attach(training_job)
        # create a model
        tensorflow_serving_model = Model(
            model_data=estimator.model_data,
            role=role,
            framework_version=tf_framework_version,
            sagemaker_session=sagemaker.Session(),
        )
        transformer = tensorflow_serving_model.transformer(
            instance_count=1,
            instance_type="ml.m4.xlarge",
            max_concurrent_transforms=5,
            max_payload=1,
        )
        transformer.transform(
            data, 
            job_name=f"{base_job_name}-{int(time.time())}", 
            content_type='text/csv', 
            split_type="Line", 
            experiment_config={"ExperimentName": experiment_name, "TrialComponentDisplayName": "Transform"}
        )
    
        
    default_args = {
        'owner': 'airflow',
        'start_date': airflow.utils.dates.days_ago(2),
        'provide_context': True
    }
    
    dag = DAG('fashion-mnist', default_args=default_args, schedule_interval='@once')
    
    train_op = PythonOperator(
        task_id='training',
        python_callable=train,
        op_args=[train_input],
        provide_context=True,
        dag=dag)
    
    transform_op = PythonOperator(
        task_id='transform',
        python_callable=transform,
        op_args=[test_input],
        provide_context=True,
        dag=dag)
    
    transform_op.set_upstream(train_op)

Now, lets init the airflow db and host it locally

.. code:: ipython3

    !airflow initdb
    !airflow webserver -p 8080 -D

Then, we start a backfill job to execute our workflow. Note, we use
backfill job simply because we dont want to wait until the airflow
scheduler to trigger the workflow to run.

.. code:: ipython3

    !airflow backfill fashion-mnist -s 2020-01-01 --reset_dagruns -y

List workflow executions
------------------------

Each execution in the workflow is modeled by a trial, lets list our
workflow executions

.. code:: ipython3

    executions = experiment.list_trials(
        sort_by="CreationTime", 
        sort_order="Ascending"
    )

.. code:: ipython3

    execs_details = []
    for exe in executions:
        execs_details.append([exe.trial_name, exe.trial_source['SourceArn'], exe.creation_time])
    execs_table = pd.DataFrame(execs_details, columns=['Name', 'Source', 'CreationTime'])

.. code:: ipython3

    execs_table

Let’s take a closer look at the jobs we created and executed by our
workflow

.. code:: ipython3

    table = ExperimentAnalytics(
        sagemaker_session=sagemaker_sess, 
        experiment_name=experiment.experiment_name,
        sort_by="CreationTime",
        sort_order="Ascending"
    )

.. code:: ipython3

    table.dataframe()

cleanup
~~~~~~~

Run the following cell to clean up the sample experiment, if you are
working on your own experiment, please ignore.

.. code:: ipython3

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

.. code:: ipython3

    cleanup(experiment)
