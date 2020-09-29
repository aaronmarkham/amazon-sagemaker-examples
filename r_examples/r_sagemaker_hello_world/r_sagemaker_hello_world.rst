.. raw:: html

   <h1>

Using R with Amazon SageMaker - Basic Notebook

.. raw:: html

   </h1>

This sample Notebook describes how you can develop R scripts in `Amazon
SageMaker <https://aws.amazon.com/sagemaker/>`__ and
`R <https://www.r-project.org/>`__ Jupyer notebooks. In this notebook we
only focus on setting up the SageMaker environment and permissions, and
then download the `abalone
dataset <https://archive.ics.uci.edu/ml/datasets/abalone>`__ from the
`UCI Machine Learning
Repository <https://archive.ics.uci.edu/ml/index.php>`__. We then do
some basic processing and visualization on the data, and will save the
data as .CSV format to S3.

For other examples related to R on SageMaker, including end-2-end
examples for training, tuning, and deploying models, please visit the
GitHub repository located at this link:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/r_examples

**R Kernel:** For running this example, you need to select R kernel from
**Kernel** menu, then **Change kernel**, then select **R**.

For more details about the R kernel in SageMaker, please visit this news
release: `Amazon SageMaker notebooks now available with pre-installed R
kernel <https://aws.amazon.com/about-aws/whats-new/2019/08/amazon-sagemaker-notebooks-available-with-pre-installed-r-kernel/>`__

.. raw:: html

   <h3>

Reticulating the Amazon SageMaker Python SDK

.. raw:: html

   </h3>

First, load the ``reticulate`` library and import the ``sagemaker``
Python module. Once the module is loaded, use the ``$`` notation in R
instead of the ``.`` notation in Python to use available classes.

.. code:: ipython3

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

.. code:: ipython3

    session <- sagemaker$Session()
    bucket <- session$default_bucket()

**Note** - The ``default_bucket`` function creates a unique Amazon S3
bucket with the following name:

``sagemaker-<aws-region-name>-<aws account number>``

Specify the IAM role’s
`ARN <https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html>`__
to allow Amazon SageMaker to access the Amazon S3 bucket. You can use
the same IAM role used to create this Notebook:

.. code:: ipython3

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

.. code:: ipython3

    library(readr)
    data_file <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    abalone <- read_csv(file = data_file, col_names = FALSE)
    names(abalone) <- c('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
    head(abalone)

The output above shows that ``sex`` is a factor data type but is
currently a character data type (F is Female, M is male, and I is
infant). Change ``sex`` to a factor and view the statistical summary of
the dataset:

.. code:: ipython3

    abalone$sex <- as.factor(abalone$sex)
    summary(abalone)

The summary above shows that the minimum value for ``height`` is 0.

Visually explore which abalones have height equal to 0 by plotting the
relationship between ``rings`` and ``height`` for each value of ``sex``:

.. code:: ipython3

    library(ggplot2)
    options(repr.plot.width = 5, repr.plot.height = 4) 
    ggplot(abalone, aes(x = height, y = rings, color = sex, alpha=0.5)) + geom_point() + geom_jitter()

.. code:: ipython3

    # Do OneHotEncoding for Sex column
    library(dplyr)
    
    abalone <- abalone %>%
      mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
             male = as.integer(ifelse(sex == 'M', 1, 0)),
             infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
      select(-sex)
    abalone <- abalone %>%
      select(rings:infant, length:shell_weight)
    head(abalone)

Now let’s write the dataframe to a CSV file locally on the SageMaker
instance.

.. code:: ipython3

    write_csv(abalone, 'abalone.csv', col_names = TRUE)

Then, upload the csv file to the Amazon S3 default bucket into the
``data`` key:

.. code:: ipython3

    s3_train <- session$upload_data(path = 'abalone.csv', 
                                    bucket = bucket, 
                                    key_prefix = 'r_hello_world_demo/data')
    
    s3_path = paste('s3://',bucket,'/r_hello_world_demo/data/abalone.csv',sep = '')
    cat('Your CSV data is stored on S3 in this location:\n',s3_path)

Extensions
----------

This example walked you through a simple process for setting up your
SageMaker environment and write your R script. In addition, you were
able to download a dataset, process it, visualize it, and then store it
on S3.

If you are interested in learning more about how you can leverage R on
SageMaker and take advantage of SageMaker features for training, tuning,
and deploying machine learning models, visit other exampls in this
GitHub repository:

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/r_examples

