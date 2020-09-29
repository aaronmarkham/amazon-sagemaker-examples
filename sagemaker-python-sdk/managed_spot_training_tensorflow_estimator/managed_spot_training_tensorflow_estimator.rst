Training using SageMaker Estimators on SageMaker Managed Spot Training
======================================================================

The example here is almost the same as `Creating, training, and serving
using SageMaker
Estimators <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb>`__.

This notebook tackles the exact same problem with the same solution, but
it has been modified to be able to run using SageMaker Managed Spot
infrastructure. SageMaker Managed Spot uses `EC2 Spot
Instances <https://aws.amazon.com/ec2/spot/>`__ to run Training at a
lower cost.

Please read the original notebook and try it out to gain an
understanding of the ML use-case and how it is being solved. We will not
delve into that here in this notebook.

First setup variables and define functions
------------------------------------------

Again, we won’t go into detail explaining the code below, it has been
lifted verbatim from `Creating, training, and serving using SageMaker
Estimators <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators/tensorflow_iris_dnn_classifier_using_estimators.ipynb>`__

.. code:: ipython2

    !pip install -qU awscli boto3 sagemaker

.. code:: ipython2

    from sagemaker import get_execution_role
    from sagemaker.session import Session
    
    # S3 bucket for saving code and model artifacts.
    # Feel free to specify a different bucket here if you wish.
    bucket = Session().default_bucket()
    
    # Location to save your custom code in tar.gz format.
    custom_code_upload_location = 's3://{}/customcode/tensorflow_iris'.format(bucket)
    
    # Location where results of model training are saved.
    model_artifacts_location = 's3://{}/artifacts'.format(bucket)
    
    #IAM execution role that gives SageMaker access to resources in your AWS account.
    role = get_execution_role()
    
    def estimator(model_path, hyperparameters):
        feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])]
        return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir=model_path)
    
    def estimator(model_path, hyperparameters):
        feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])]
        return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir=model_path)
    
    def train_input_fn(training_dir, hyperparameters):
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=os.path.join(training_dir, 'iris_training.csv'),
            target_dtype=np.int,
            features_dtype=np.float32)
    
        return tf.estimator.inputs.numpy_input_fn(
            x={INPUT_TENSOR_NAME: np.array(training_set.data)},
            y=np.array(training_set.target),
            num_epochs=None,
            shuffle=True)()
    
    def serving_input_fn(hyperparameters):
        feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()
    
    import boto3
    region = boto3.Session().region_name

Managed Spot Training with a TensorFlow Estimator
=================================================

For Managed Spot Training using a TensorFlow Estimator we need to
configure two things: 1. Enable the ``train_use_spot_instances``
constructor arg - a simple self-explanatory boolean. 2. Set the
``train_max_wait`` constructor arg - this is an int arg representing the
amount of time you are willing to wait for Spot infrastructure to become
available. Some instance types are harder to get at Spot prices and you
may have to wait longer. You are not charged for time spent waiting for
Spot infrastructure to become available, you’re only charged for actual
compute time spent once Spot instances have been successfully procured.

Normally, a third requirement would also be necessary here - modifying
your code to ensure a regular checkpointing cadence - however,
TensorFlow Estimators already do this, so no changes are necessary here.
Checkpointing is highly recommended for Manage Spot Training jobs due to
the fact that Spot instances can be interrupted with short notice and
using checkpoints to resume from the last interruption ensures you don’t
lose any progress made before the interruption.

Feel free to toggle the ``train_use_spot_instances`` variable to see the
effect of running the same job using regular (a.k.a. “On Demand”)
infrastructure.

Note that ``train_max_wait`` can be set if and only if
``train_use_spot_instances`` is enabled and **must** be greater than or
equal to ``train_max_run``.

.. code:: ipython2

    train_use_spot_instances = True
    train_max_run=3600
    train_max_wait = 7200 if train_use_spot_instances else None

.. code:: ipython2

    from sagemaker.tensorflow import TensorFlow
    
    
    iris_estimator = TensorFlow(entry_point='iris_dnn_classifier.py',
                                role=role,
                                framework_version='1.12.0',
                                output_path=model_artifacts_location,
                                code_location=custom_code_upload_location,
                                train_instance_count=1,
                                train_instance_type='ml.c4.xlarge',
                                training_steps=1000,
                                evaluation_steps=100,
                                train_use_spot_instances=train_use_spot_instances,
                                train_max_run=train_max_run,
                                train_max_wait=train_max_wait)
    # use the region-specific sample data bucket
    train_data_location = 's3://sagemaker-sample-data-{}/tensorflow/iris'.format(region)
    iris_estimator.fit(train_data_location)

Savings
=======

Towards the end of the job you should see two lines of output printed:

-  ``Training seconds: X`` : This is the actual compute-time your
   training job spent
-  ``Billable seconds: Y`` : This is the time you will be billed for
   after Spot discounting is applied.

If you enabled the ``train_use_spot_instances`` var then you should see
a notable difference between ``X`` and ``Y`` signifying the cost savings
you will get for having chosen Managed Spot Training. This should be
reflected in an additional line: -
``Managed Spot Training savings: (1-Y/X)*100 %``
