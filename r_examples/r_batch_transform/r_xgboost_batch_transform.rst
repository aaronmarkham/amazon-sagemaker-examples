.. raw:: html

   <h1>

Batch Transform Using R with Amazon SageMaker

.. raw:: html

   </h1>

**Note:** You will need to use R kernel in SageMaker for this notebook.

This sample Notebook describes how to do batch transform to make
predictions for an abalone’s age, which is measured by the number of
rings in the shell. The notebook will use the public `abalone
dataset <https://archive.ics.uci.edu/ml/datasets/abalone>`__ hosted by
`UCI Machine Learning
Repository <https://archive.ics.uci.edu/ml/index.php>`__.

You can find more details about SageMaker’s Batch Trsnform here: -
`Batch
Transform <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html>`__
using a Transformer

We will use ``reticulate`` library to interact with SageMaker: -
```Reticulate`` library <https://rstudio.github.io/reticulate/>`__:
provides an R interface to use the `Amazon SageMaker Python
SDK <https://sagemaker.readthedocs.io/en/latest/index.html>`__ to make
API calls to Amazon SageMaker. The ``reticulate`` package translates
between R and Python objects, and Amazon SageMaker provides a serverless
data science environment to train and deploy ML models at scale.

Table of Contents: - `Reticulating the Amazon SageMaker Python
SDK <#Reticulating-the-Amazon-SageMaker-Python-SDK>`__ - `Creating and
Accessing the Data Storage <#Creating-and-accessing-the-data-storage>`__
- `Downloading and Processing the
Dataset <#Downloading-and-processing-the-dataset>`__ - `Preparing the
Dataset for Model
Training <#Preparing-the-dataset-for-model-training>`__ - `Creating a
SageMaker Estimator <#Creating-a-SageMaker-Estimator>`__ - `Batch
Transform using SageMaker
Transformer <#Batch-Transform-using-SageMaker-Transformer>`__ -
`Download the Batch Transform
Output <#Download-the-Batch-Transform-Output>`__

**Note:** The first portion of this notebook focused on data ingestion
and preparing the data for model training is inspired by the data
preparation section outlined in the `“Using R with Amazon
SageMaker” <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/r_kernel/using_r_with_amazon_sagemaker.ipynb>`__
notebook on AWS SageMaker Examples Github repository with some
modifications.

.. raw:: html

   <h3>

Reticulating the Amazon SageMaker Python SDK

.. raw:: html

   </h3>

First, load the ``reticulate`` library and import the ``sagemaker``
Python module. Once the module is loaded, use the ``$`` notation in R
instead of the ``.`` notation in Python to use available classes.

.. code:: r

    # Turn warnings off globally
    options(warn=-1)

.. code:: r

    # Install reticulate library and import sagemaker
    library(reticulate)
    sagemaker <- import('sagemaker')

.. raw:: html

   <h3>

Creating and Accessing the Data Storage

.. raw:: html

   </h3>

The ``Session`` class provides operations for working with the following
`boto3 <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`__
resources with Amazon SageMaker:

-  `S3 <https://boto3.readthedocs.io/en/latest/reference/services/s3.html>`__
-  `SageMaker <https://boto3.readthedocs.io/en/latest/reference/services/sagemaker.html>`__

Let’s create an `Amazon Simple Storage
Service <https://aws.amazon.com/s3/>`__ bucket for your data.

.. code:: r

    session <- sagemaker$Session()
    bucket <- session$default_bucket()
    prefix <- 'r-batch-transform'

**Note** - The ``default_bucket`` function creates a unique Amazon S3
bucket with the following name:

``sagemaker-<aws-region-name>-<aws account number>``

Specify the IAM role’s
`ARN <https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html>`__
to allow Amazon SageMaker to access the Amazon S3 bucket. You can use
the same IAM role used to create this Notebook:

.. code:: r

    role_arn <- sagemaker$get_execution_role()

.. raw:: html

   <h3>

Downloading and Processing the Dataset

.. raw:: html

   </h3>

The model uses the `abalone
dataset <https://archive.ics.uci.edu/ml/datasets/abalone>`__ from the
`UCI Machine Learning
Repository <https://archive.ics.uci.edu/ml/index.php>`__. First,
download the data and start the `exploratory data
analysis <https://en.wikipedia.org/wiki/Exploratory_data_analysis>`__.
Use tidyverse packages to read, plot, and transform the data into ML
format for Amazon SageMaker:

.. code:: r

    library(readr)
    data_file <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    abalone <- read_csv(file = data_file, col_names = FALSE)
    names(abalone) <- c('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
    head(abalone)

The output above shows that ``sex`` is a factor data type but is
currently a character data type (F is Female, M is male, and I is
infant). Change ``sex`` to a factor and view the statistical summary of
the dataset:

.. code:: r

    abalone$sex <- as.factor(abalone$sex)
    summary(abalone)

The summary above shows that the minimum value for ``height`` is 0.

Visually explore which abalones have height equal to 0 by plotting the
relationship between ``rings`` and ``height`` for each value of ``sex``:

.. code:: r

    library(ggplot2)
    options(repr.plot.width = 5, repr.plot.height = 4) 
    ggplot(abalone, aes(x = height, y = rings, color = sex)) + geom_point() + geom_jitter()

The plot shows multiple outliers: two infant abalones with a height of 0
and a few female and male abalones with greater heights than the rest.
Let’s filter out the two infant abalones with a height of 0.

.. code:: r

    library(dplyr)
    abalone <- abalone %>%
      filter(height != 0)

.. raw:: html

   <h3>

Preparing the Dataset for Model Training

.. raw:: html

   </h3>

The model needs three datasets: one for training, testing, and
validation. First, convert ``sex`` into a `dummy
variable <https://en.wikipedia.org/wiki/Dummy_variable_(statistics)>`__
and move the target, ``rings``, to the first column. Amazon SageMaker
algorithm require the target to be in the first column of the dataset.

.. code:: r

    abalone <- abalone %>%
      mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
             male = as.integer(ifelse(sex == 'M', 1, 0)),
             infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
      select(-sex)
    abalone <- abalone %>%
      select(rings:infant, length:shell_weight)
    head(abalone)

Next, sample 70% of the data for training the ML algorithm. Split the
remaining 30% into two halves, one for testing and one for validation:

.. code:: r

    abalone_train <- abalone %>%
      sample_frac(size = 0.7)
    abalone <- anti_join(abalone, abalone_train)
    abalone_test <- abalone %>%
      sample_frac(size = 0.5)
    abalone_valid <- anti_join(abalone, abalone_test)

Upload the training and validation data to Amazon S3 so that you can
train the model. First, write the training and validation datasets to
the local filesystem in .csv format:


Second, upload the two datasets to the Amazon S3 bucket into the
``data`` key:

.. code:: r

    write_csv(abalone_train, 'abalone_train.csv', col_names = FALSE)
    write_csv(abalone_valid, 'abalone_valid.csv', col_names = FALSE)
    
    # Remove target from test
    write_csv(abalone_test[-1], 'abalone_test.csv', col_names = FALSE)

.. code:: r

    s3_train <- session$upload_data(path = 'abalone_train.csv', 
                                    bucket = bucket, 
                                    key_prefix = paste(prefix,'data', sep = '/'))
    s3_valid <- session$upload_data(path = 'abalone_valid.csv', 
                                    bucket = bucket, 
                                    key_prefix = paste(prefix,'data', sep = '/'))
    
    s3_test <- session$upload_data(path = 'abalone_test.csv', 
                                    bucket = bucket, 
                                    key_prefix = paste(prefix,'data', sep = '/'))

Finally, define the Amazon S3 input types for the Amazon SageMaker
algorithm:

.. code:: r

    s3_train_input <- sagemaker$s3_input(s3_data = s3_train,
                                         content_type = 'csv')
    s3_valid_input <- sagemaker$s3_input(s3_data = s3_valid,
                                         content_type = 'csv')

.. raw:: html

   <hr>

.. raw:: html

   <h3>

Creating a SageMaker Estimator

.. raw:: html

   </h3>

Amazon SageMaker algorithm are available via a
`Docker <https://www.docker.com/>`__ container. To train an
`XGBoost <https://en.wikipedia.org/wiki/Xgboost>`__ model, specify the
training containers in `Amazon Elastic Container
Registry <https://aws.amazon.com/ecr/>`__ (Amazon ECR) for the AWS
Region.

.. code:: r

    registry <- sagemaker$amazon$amazon_estimator$registry(session$boto_region_name, algorithm='xgboost')
    container <- paste(registry, '/xgboost:latest', sep='')
    cat('XGBoost Container Image URL: ', container)

Define an Amazon SageMaker
`Estimator <http://sagemaker.readthedocs.io/en/latest/estimators.html>`__,
which can train any supplied algorithm that has been containerized with
Docker. When creating the Estimator, use the following arguments: \*
**image_name** - The container image to use for training \* **role** -
The Amazon SageMaker service role \* **train_instance_count** - The
number of Amazon EC2 instances to use for training \*
**train_instance_type** - The type of Amazon EC2 instance to use for
training \* **train_volume_size** - The size in GB of the `Amazon
Elastic Block Store <https://aws.amazon.com/ebs/>`__ (Amazon EBS) volume
to use for storing input data during training \* **train_max_run** - The
timeout in seconds for training \* **input_mode** - The input mode that
the algorithm supports \* **output_path** - The Amazon S3 location for
saving the training results (model artifacts and output files) \*
**output_kms_key** - The `AWS Key Management
Service <https://aws.amazon.com/kms/>`__ (AWS KMS) key for encrypting
the training output \* **base_job_name** - The prefix for the name of
the training job \* **sagemaker_session** - The Session object that
manages interactions with Amazon SageMaker API

.. code:: r

    # Model artifacts and batch output
    s3_output <- paste('s3:/', bucket, prefix,'output', sep = '/')

.. code:: r

    # Estimator
    estimator <- sagemaker$estimator$Estimator(image_name = container,
                                               role = role_arn,
                                               train_instance_count = 1L,
                                               train_instance_type = 'ml.m5.4xlarge',
                                               train_volume_size = 30L,
                                               train_max_run = 3600L,
                                               input_mode = 'File',
                                               output_path = s3_output,
                                               output_kms_key = NULL,
                                               base_job_name = NULL,
                                               sagemaker_session = NULL)

**Note** - The equivalent to ``None`` in Python is ``NULL`` in R.

Next, we Specify the `XGBoost
hyperparameters <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html>`__
for the estimator.

Once the Estimator and its hyperparamters are specified, you can train
(or fit) the estimator.

.. code:: r

    # Set Hyperparameters
    estimator$set_hyperparameters(eval_metric='rmse',
                                  objective='reg:linear',
                                  num_round=100L,
                                  rate_drop=0.3,
                                  tweedie_variance_power=1.4)

.. code:: r

    # Create a training job name
    job_name <- paste('sagemaker-r-xgboost', format(Sys.time(), '%H-%M-%S'), sep = '-')
    
    # Define the data channels for train and validation datasets
    input_data <- list('train' = s3_train_input,
                       'validation' = s3_valid_input)
    
    # train the estimator
    estimator$fit(inputs = input_data, job_name = job_name)

.. raw:: html

   <hr>

.. raw:: html

   <h3>

Batch Transform using SageMaker Transformer

.. raw:: html

   </h3>

For more details on SageMaker Batch Transform, you can visit this
example notebook on `Amazon SageMaker Batch
Transform <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_batch_transform/introduction_to_batch_transform/batch_transform_pca_dbscan_movie_clusters.ipynb>`__.

In many situations, using a deployed model for making inference is not
the best option, especially when the goal is not to make online
real-time inference but to generate predictions from a trained model on
a large dataset. In these situations, using Batch Transform may be more
efficient and appropriate.

This section of the notebook explains how to set up the Batch Transform
Job and generate predictions.

To do this, we need to identify the batch input data path in S3 and
specify where generated predictions will be stored in S3.

.. code:: r

    # Define S3 path for Test data 
    s3_test_url <- paste('s3:/', bucket, prefix, 'data','abalone_test.csv', sep = '/')

Then we create a ``Transformer``.
`Transformers <https://sagemaker.readthedocs.io/en/stable/transformer.html#transformer>`__
take multiple paramters, including the following. For more details and
the complete list visit the `documentation
page <https://sagemaker.readthedocs.io/en/stable/transformer.html#transformer>`__.

-  **model_name** (str) – Name of the SageMaker model being used for the
   transform job.
-  **instance_count** (int) – Number of EC2 instances to use.
-  **instance_type** (str) – Type of EC2 instance to use, for example,
   ‘ml.c4.xlarge’.

-  **output_path** (str) – S3 location for saving the transform result.
   If not specified, results are stored to a default bucket.

-  **base_transform_job_name** (str) – Prefix for the transform job when
   the transform() method launches. If not specified, a default prefix
   will be generated based on the training image name that was used to
   train the model associated with the transform job.

-  **sagemaker_session** (sagemaker.session.Session) – Session object
   which manages interactions with Amazon SageMaker APIs and any other
   AWS services needed. If not specified, the estimator creates one
   using the default AWS configuration chain.

Once we create a ``Transformer`` we can transform the batch input.

.. code:: r

    # Define a transformer
    transformer <- estimator$transformer(instance_count=1L, 
                                         instance_type='ml.m4.xlarge',
                                         output_path = s3_output)

.. code:: r

    # Do the batch transform
    transformer$transform(s3_test_url,
                         wait = TRUE)

.. raw:: html

   <hr>

.. raw:: html

   <h3>

Download the Batch Transform Output

.. raw:: html

   </h3>

.. code:: r

    # Download the file from S3 using S3Downloader to local SageMaker instance 'batch_output' folder
    sagemaker$s3$S3Downloader$download(paste(s3_output,"abalone_test.csv.out",sep = '/'),
                              "batch_output")

.. code:: r

    # Read the batch csv from sagemaker local files
    library(readr)
    predictions <- read_csv(file = 'batch_output/abalone_test.csv.out', col_names = 'predicted_rings')
    head(predictions)

Column-bind the predicted rings to the test data:

.. code:: r

    # Concatenate predictions and test for comparison
    abalone_predictions <- cbind(predicted_rings = predictions, 
                          abalone_test)
    # Convert predictions to Integer
    abalone_predictions$predicted_rings = as.integer(abalone_predictions$predicted_rings);
    head(abalone_predictions)

.. code:: r

    # Define a function to calculate RMSE
    rmse <- function(m, o){
      sqrt(mean((m - o)^2))
    }

.. code:: r

    # Calucalte RMSE
    abalone_rmse <- rmse(abalone_predictions$rings, abalone_predictions$predicted_rings)
    cat('RMSE for Batch Transform: ', round(abalone_rmse, digits = 2))

