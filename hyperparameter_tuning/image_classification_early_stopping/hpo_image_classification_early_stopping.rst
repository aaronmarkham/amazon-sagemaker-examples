Automatic Model Tuning : Automatic training job early stopping
==============================================================

**Using automatic training job early stopping to speed up the tuning of
an end-to-end multiclass image classification task**

Contents
--------

1. `Background <#Background>`__
2. `Set_up <#Set-up>`__
3. `Data_preparation <#Data-preparation>`__
4. `Set_up_hyperparameter_tuning_job <#Set-up-hyperparameter-tuning-job>`__
5. `Launch_hyperparameter_tuning_job <#Launch-hyperparameter-tuning-job>`__
6. `Launch_hyperparameter_tuning_job_with_automatic_early_stopping <#Launch-hyperparameter-tuning-job-with-automatic-early-stopping>`__
7. `Wrap_up <#Wrap-up>`__

--------------

Background
----------

Selecting the right hyperparameter values for machine learning model can
be difficult. The right answer dependes on the algorithm and the data;
Some algorithms have many tuneable hyperparameters; Some are very
sensitive to the hyperparameter values selected; and yet most have a
non-linear relationship between model fit and hyperparameter values.
Amazon SageMaker Automatic Model Tuning helps by automating the
hyperparameter tuning process.

Experienced data scientist often stop a training when it is not
promising based on the first few validation metrics emitted during the
training. This notebook will demonstrate how to use the automatic
training job early stopping of Amazon SageMaker Automatic Model Tuning
to speed up the tuning process with a simple switch.

--------------

Set up
------

Let us start by specifying:

-  The role that is used to give learning and hosting the access to the
   data. This will automatically be obtained from the role used to start
   the notebook.
-  The S3 bucket that will be used for loading training data and saving
   model data.
-  The Amazon SageMaker image classification docker image which need not
   to be changed.

.. code:: ipython3

    import boto3
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    role = get_execution_role()
    
    bucket = '<<bucket-name>>' # customize to your bucket
    
    training_image = get_image_uri(boto3.Session().region_name, 'image-classification')

Data preparation
----------------

In this example, `caltech-256
dataset <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`__
dataset will be used, which contains 30608 images of 256 objects.

.. code:: ipython3

    import os 
    import urllib.request
    import boto3
    
    def download(url):
        filename = url.split("/")[-1]
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
    
            
    def upload_to_s3(channel, file):
        s3 = boto3.resource('s3')
        data = open(file, "rb")
        key = channel + '/' + file
        s3.Bucket(bucket).put_object(Key=key, Body=data)
    
    s3_train_key = "image-classification-full-training/train"
    s3_validation_key = "image-classification-full-training/validation"
    
    download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
    upload_to_s3(s3_train_key, 'caltech-256-60-train.rec')
    download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
    upload_to_s3(s3_validation_key, 'caltech-256-60-val.rec')

Set up hyperparameter tuning job
--------------------------------

For this example, three hyperparameters will be tuned: learning_rate,
mini_batch_size and optimizer, which has the greatest impact on the
objective metric. See
`here <https://docs.aws.amazon.com/sagemaker/latest/dg/IC-tuning.html>`__
for more detail and the full list of hyperparameters that can be tuned.

Before launching the tuning job, training jobs that the hyperparameter
tuning job will launch need to be configured by defining an estimator
that specifies the following information:

-  The container image for the algorithm (image-classification).
-  The s3 location for training and validation data.
-  The type and number of instances to use for the training jobs.
-  The output specification where the output can be stored after
   training.

The values of any hyperparameters that are not tuned in the tuning job
(StaticHyperparameters): \* **num_layers**: The number of layers (depth)
for the network. We use 18 in this samples but other values such as 50,
152 can be used. \* **image_shape**: The input image
dimensions,‘num_channels, height, width’, for the network. It should be
no larger than the actual image size. The number of channels should be
same as in the actual image. \* **num_classes**: This is the number of
output classes for the new dataset. For caltech, we use 257 because it
has 256 object categories + 1 clutter class. \*
**num_training_samples**: This is the total number of training samples.
It is set to 15240 for caltech dataset with the current split. \*
**epochs**: Number of training epochs. In this example we set it to only
10 to save the cost. If you would like to get higher accuracy the number
of epochs can be increased. \* **top_k**: Report the top-k accuracy
during training. \* **precision_dtype**: Training data type precision
(default: float32). If set to ‘float16’, the training will be done in
mixed_precision mode and will be faster than float32 mode. \*
**augmentation_type**: ‘crop’. Randomly crop the image and flip the
image horizontally.

.. code:: ipython3

    import sagemaker
    
    s3_train_data = 's3://{}/{}/'.format(bucket, s3_train_key)
    s3_validation_data = 's3://{}/{}/'.format(bucket, s3_validation_key)
    
    s3_output_key = "image-classification-full-training/output"
    s3_output = 's3://{}/{}/'.format(bucket, s3_output_key)
    
    s3_input_train = sagemaker.s3_input(s3_data=s3_train_data, content_type='application/x-recordio')
    s3_input_validation = sagemaker.s3_input(s3_data=s3_validation_data, content_type='application/x-recordio')

.. code:: ipython3

    sess = sagemaker.Session()
    imageclassification = sagemaker.estimator.Estimator(training_image, 
                                                        role, 
                                                        train_instance_count=1,
                                                        train_instance_type='ml.p3.2xlarge',
                                                        output_path=s3_output, 
                                                        sagemaker_session=sess)
    
    imageclassification.set_hyperparameters(num_layers=18, 
                                            image_shape='3,224,224',
                                            num_classes=257, 
                                            epochs=10, 
                                            top_k='2',
                                            num_training_samples=15420,  
                                            precision_dtype='float32',
                                            augmentation_type='crop')

Next, the tuning job with the following configurations need to be
specified: \* the hyperparameters that SageMaker Automatic Model Tuning
will tune: learning_rate, mini_batch_size and optimizer \* the maximum
number of training jobs it will run to optimize the objective metric: 20
\* the number of parallel training jobs that will run in the tuning job:
2 \* the objective metric that Automatic Model Tuning will use:
validation:accuracy

.. code:: ipython3

    from time import gmtime, strftime 
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
    
    tuning_job_name = "imageclassif-job-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    
    hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.00001, 1.0),
                             'mini_batch_size': IntegerParameter(16, 64),
                             'optimizer': CategoricalParameter(['sgd', 'adam', 'rmsprop', 'nag'])}
    
    objective_metric_name = 'validation:accuracy'
    
    tuner = HyperparameterTuner(imageclassification, 
                                objective_metric_name, 
                                hyperparameter_ranges,
                                objective_type='Maximize', 
                                max_jobs=20, 
                                max_parallel_jobs=2)

Launch hyperparameter tuning job
--------------------------------

Now we can launch a hyperparameter tuning job by calling fit in tuner.
We will wait until the tuning finished, which may take around 2 hours.

.. code:: ipython3

    tuner.fit({'train': s3_input_train, 'validation': s3_input_validation}, 
              job_name=tuning_job_name, include_cls_metadata=False)
    tuner.wait()

After the tuning finished, the top 5 performing hyperparameters can be
listed below. One can analyse the results deeper by using
`HPO_Analyze_TuningJob_Results.ipynb
notebook <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb>`__.

.. code:: ipython3

    tuner_metrics = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
    tuner_metrics.dataframe().sort_values(['FinalObjectiveValue'], ascending=False).head(5)

The total training time and training jobs status can be checked with the
following script. Because automatic early stopping is by default off,
all the training jobs should be completed normally.

.. code:: ipython3

    total_time = tuner_metrics.dataframe()['TrainingElapsedTimeSeconds'].sum() / 3600
    print("The total training time is {:.2f} hours".format(total_time))
    tuner_metrics.dataframe()['TrainingJobStatus'].value_counts()

Launch hyperparameter tuning job with automatic early stopping
--------------------------------------------------------------

Now we lunch the same tuning job with only one difference: setting
**early_stopping_type**\ =\ **‘Auto’** to enable automatic training job
early stopping.

.. code:: ipython3

    tuning_job_name_es = "imageclassif-job-{}-es".format(strftime("%d-%H-%M-%S", gmtime()))
    
    tuner_es = HyperparameterTuner(imageclassification, 
                                   objective_metric_name, 
                                   hyperparameter_ranges,
                                   objective_type='Maximize', 
                                   max_jobs=20, 
                                   max_parallel_jobs=2, 
                                   early_stopping_type='Auto')
    
    tuner_es.fit({'train': s3_input_train, 'validation': s3_input_validation}, 
                 job_name=tuning_job_name_es, include_cls_metadata=False)
    tuner_es.wait()

After the tuning job finished, we again list the top 5 performing
training jobs.

.. code:: ipython3

    tuner_metrics_es = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name_es)
    tuner_metrics_es.dataframe().sort_values(['FinalObjectiveValue'], ascending=False).head(5)

The total training time and training jobs status can be checked with the
following script.

.. code:: ipython3

    df = tuner_metrics_es.dataframe()
    total_time_es = df['TrainingElapsedTimeSeconds'].sum() / 3600
    print("The total training time with early stopping is {:.2f} hours".format(total_time_es))
    df['TrainingJobStatus'].value_counts()

The stopped training jobs can be listed using the following scripts.

.. code:: ipython3

    df[df.TrainingJobStatus == 'Stopped']

Wrap up
-------

In this notebook, we demonstrated how to use automatic early stopping to
speed up model tuning. One thing to keep in mind is as the training time
for each training job gets longer, the benefit of training job early
stopping becomes more significant. On the other hand, smaller training
jobs won’t benefit as much due to infrastructure overhead. For example,
our experiments show that the effect of training job early stopping
typically becomes noticeable when the training jobs last longer than **4
minutes**. To enable automatic early stopping, one can simply set
**early_stopping_type** to **‘Auto’**.

For more information on using SageMaker’s Automatic Model Tuning, see
our other `example
notebooks <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning>`__
and
`documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`__.
