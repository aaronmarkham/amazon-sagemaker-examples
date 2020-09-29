.. raw:: html

   <h1>

Using R with Amazon SageMaker

.. raw:: html

   </h1>

This sample Notebook describes how to train, deploy, and retrieve
predictions from a machine learning (ML) model using `Amazon
SageMaker <https://aws.amazon.com/sagemaker/>`__ and
`R <https://www.r-project.org/>`__. The model predicts abalone age as
measured by the number of rings in the shell. The
`reticulate <https://rstudio.github.io/reticulate/>`__ package will be
used as an R interface to `Amazon SageMaker Python
SDK <https://sagemaker.readthedocs.io/en/latest/index.html>`__ to make
API calls to Amazon SageMaker. The ``reticulate`` package translates
between R and Python objects, and Amazon SageMaker provides a serverless
data science environment to train and deploy ML models at scale.

.. raw:: html

   <h3>

Reticulating the Amazon SageMaker Python SDK

.. raw:: html

   </h3>

First, load the ``reticulate`` library and import the ``sagemaker``
Python module. Once the module is loaded, use the ``$`` notation in R
instead of the ``.`` notation in Python to use available classes.

.. code:: r

    library(reticulate)
    sagemaker <- import('sagemaker')

.. raw:: html

   <h3>

Creating and accessing the data storage

.. raw:: html

   </h3>

The ``Session`` class provides operations for working with the following
`boto3 <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`__
resources with Amazon SageMaker:

-  `S3 <https://boto3.readthedocs.io/en/latest/reference/services/s3.html>`__
-  `SageMaker <https://boto3.readthedocs.io/en/latest/reference/services/sagemaker.html>`__
-  `SageMakerRuntime <https://boto3.readthedocs.io/en/latest/reference/services/sagemaker-runtime.html>`__

Let’s create an `Amazon Simple Storage
Service <https://aws.amazon.com/s3/>`__ bucket for your data.

.. code:: r

    session <- sagemaker$Session()
    bucket <- session$default_bucket()

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

Downloading and processing the dataset

.. raw:: html

   </h3>

The model uses the `abalone
dataset <https://archive.ics.uci.edu/ml/datasets/abalone>`__ from the
`UCI Machine Learning
Repository <https://archive.ics.uci.edu/ml/index.php>`__. First,
download the data and start the `exploratory data
analysis <https://en.wikipedia.org/wiki/Exploratory_data_analysis>`__.
Use tidyverse packages to read the data, plot the data, and transform
the data into ML format for Amazon SageMaker:

.. code:: r

    library(readr)
    data_file <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
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

Preparing the dataset for model training

.. raw:: html

   </h3>

The model needs three datasets: one each for training, testing, and
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

.. code:: r

    write_csv(abalone_train, 'abalone_train.csv', col_names = FALSE)
    write_csv(abalone_valid, 'abalone_valid.csv', col_names = FALSE)

Second, upload the two datasets to the Amazon S3 bucket into the
``data`` key:

.. code:: r

    s3_train <- session$upload_data(path = 'abalone_train.csv', 
                                    bucket = bucket, 
                                    key_prefix = 'data')
    s3_valid <- session$upload_data(path = 'abalone_valid.csv', 
                                    bucket = bucket, 
                                    key_prefix = 'data')

Finally, define the Amazon S3 input types for the Amazon SageMaker
algorithm:

.. code:: r

    s3_train_input <- sagemaker$s3_input(s3_data = s3_train,
                                         content_type = 'csv')
    s3_valid_input <- sagemaker$s3_input(s3_data = s3_valid,
                                         content_type = 'csv')

.. raw:: html

   <h3>

Training the model

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

    s3_output <- paste0('s3://', bucket, '/output')
    estimator <- sagemaker$estimator$Estimator(image_name = container,
                                               role = role_arn,
                                               train_instance_count = 1L,
                                               train_instance_type = 'ml.m5.large',
                                               train_volume_size = 30L,
                                               train_max_run = 3600L,
                                               input_mode = 'File',
                                               output_path = s3_output,
                                               output_kms_key = NULL,
                                               base_job_name = NULL,
                                               sagemaker_session = NULL)

**Note** - The equivalent to ``None`` in Python is ``NULL`` in R.

Specify the `XGBoost
hyperparameters <https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html>`__
and fit the model. Set the number of rounds for training to 100 which is
the default value when using the XGBoost library outside of Amazon
SageMaker. Also specify the input data and a job name based on the
current time stamp:

.. code:: r

    estimator$set_hyperparameters(num_round = 100L)
    job_name <- paste('sagemaker-train-xgboost', format(Sys.time(), '%H-%M-%S'), sep = '-')
    input_data <- list('train' = s3_train_input,
                       'validation' = s3_valid_input)
    estimator$fit(inputs = input_data,
                  job_name = job_name)

Once training has finished, Amazon SageMaker copies the model binary (a
gzip tarball) to the specified Amazon S3 output location. Get the full
Amazon S3 path with this command:

.. code:: r

    estimator$model_data

.. raw:: html

   <h3>

Deploying the model

.. raw:: html

   </h3>

Amazon SageMaker lets you `deploy your
model <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html>`__
by providing an endpoint that consumers can invoke by a secure and
simple API call using an HTTPS request. Let’s deploy our trained model
to a ``ml.t2.medium`` instance.

.. code:: r

    model_endpoint <- estimator$deploy(initial_instance_count = 1L,
                                       instance_type = 'ml.t2.medium')

.. raw:: html

   <h3>

Generating predictions with the model

.. raw:: html

   </h3>

Use the test data to generate predictions. Pass comma-separated text to
be serialized into JSON format by specifying ``text/csv`` and
``csv_serializer`` for the endpoint:

.. code:: r

    model_endpoint$content_type <- 'text/csv'
    model_endpoint$serializer <- sagemaker$predictor$csv_serializer

Remove the target column and convert the first 500 observations to a
matrix with no column names:

.. code:: r

    abalone_test <- abalone_test[-1]
    num_predict_rows <- 500
    test_sample <- as.matrix(abalone_test[1:num_predict_rows, ])
    dimnames(test_sample)[[2]] <- NULL

**Note** - 500 observations was chosen because it doesn’t exceed the
endpoint limitation.

Generate predictions from the endpoint and convert the returned
comma-separated string:

.. code:: r

    library(stringr)
    predictions <- model_endpoint$predict(test_sample)
    predictions <- str_split(predictions, pattern = ',', simplify = TRUE)
    predictions <- as.numeric(predictions)

Column-bind the predicted rings to the test data:

.. code:: r

    abalone_test <- cbind(predicted_rings = predictions, 
                          abalone_test[1:num_predict_rows, ])
    head(abalone_test)

.. raw:: html

   <h3>

Deleting the endpoint

.. raw:: html

   </h3>

When you’re done with the model, delete the endpoint to avoid incurring
deployment costs:

.. code:: r

    session$delete_endpoint(model_endpoint$endpoint)
