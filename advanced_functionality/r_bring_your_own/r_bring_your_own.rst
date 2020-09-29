Bring Your Own R Algorithm
==========================

**Create a Docker container for training R algorithms and hosting R
models**

--------------

--------------

Contents
--------

1.  `Background <#Background>`__
2.  `Preparation <#Preparation>`__
3.  `Code <#Code>`__
4.  `Fit <#Fit>`__
5.  `Serve <#Serve>`__
6.  `Dockerfile <#Dockerfile>`__
7.  `Publish <#Publish>`__
8.  `Data <#Data>`__
9.  `Train <#Train>`__
10. `Host <#Host>`__
11. `Predict <#Predict>`__
12. `Extensions <#Extensions>`__

Preparation
-----------

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the boto regexp with a the appropriate full
   IAM role arn string(s).

.. code:: ipython3

    # Define IAM role
    import boto3
    import re
    import sagemaker
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    bucket = sagemaker.Session().default_bucket()
    prefix = 'sagemaker/DEMO-r-byo'

Now we’ll import the libraries we’ll need for the remainder of the
notebook.

.. code:: ipython3

    import time
    import json
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

Permissions
~~~~~~~~~~~

Running this notebook requires permissions in addition to the normal
``SageMakerFullAccess`` permissions. This is because we’ll be creating a
new repository in Amazon ECR. The easiest way to add these permissions
is simply to add the managed policy
``AmazonEC2ContainerRegistryFullAccess`` to the role that you used to
start your notebook instance. There’s no need to restart your notebook
instance when you do this, the new permissions will be available
immediately.

--------------

Code
----

For this example, we’ll need 3 supporting code files.

Fit
~~~

``mars.R`` creates functions to fit and serve our model. The algorithm
we’ve chosen to use is `Multivariate Adaptive Regression
Splines <https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_splines>`__.
This is a suitable example as it’s a unique and powerful algorithm, but
isn’t as broadly used as Amazon SageMaker algorithms, and it isn’t
available in Python’s scikit-learn library. R’s repository of packages
is filled with algorithms that share these same criteria.

*The top of the code is devoted to setup. Bringing in the libraries
we’ll need and setting up the file paths as detailed in Amazon SageMaker
documentation on bringing your own container.*

::

   # Bring in library that contains multivariate adaptive regression splines (MARS)
   library(mda)

   # Bring in library that allows parsing of JSON training parameters
   library(jsonlite)

   # Bring in library for prediction server
   library(plumber)


   # Setup parameters
   # Container directories
   prefix <- '/opt/ml'
   input_path <- paste(prefix, 'input/data', sep='/')
   output_path <- paste(prefix, 'output', sep='/')
   model_path <- paste(prefix, 'model', sep='/')
   param_path <- paste(prefix, 'input/config/hyperparameters.json', sep='/')

   # Channel holding training data
   channel_name = 'train'
   training_path <- paste(input_path, channel_name, sep='/')

*Next, we define a train function that actually fits the model to the
data. For the most part this is idiomatic R, with a bit of maneuvering
up front to take in parameters from a JSON file, and at the end to
output a success indicator.*

::

   # Setup training function
   train <- function() {

       # Read in hyperparameters
       training_params <- read_json(param_path)

       target <- training_params$target

       if (!is.null(training_params$degree)) {
           degree <- as.numeric(training_params$degree)}
       else {
           degree <- 2}

       # Bring in data
       training_files = list.files(path=training_path, full.names=TRUE)
       training_data = do.call(rbind, lapply(training_files, read.csv))
       
       # Convert to model matrix
       training_X <- model.matrix(~., training_data[, colnames(training_data) != target])

       # Save factor levels for scoring
       factor_levels <- lapply(training_data[, sapply(training_data, is.factor), drop=FALSE],
                               function(x) {levels(x)})
       
       # Run multivariate adaptive regression splines algorithm
       model <- mars(x=training_X, y=training_data[, target], degree=degree)
       
       # Generate outputs
       mars_model <- model[!(names(model) %in% c('x', 'residuals', 'fitted.values'))]
       attributes(mars_model)$class <- 'mars'
       save(mars_model, factor_levels, file=paste(model_path, 'mars_model.RData', sep='/'))
       print(summary(mars_model))

       write.csv(model$fitted.values, paste(output_path, 'data/fitted_values.csv', sep='/'), row.names=FALSE)
       write('success', file=paste(output_path, 'success', sep='/'))}

*Then, we setup the serving function (which is really just a short
wrapper around our plumber.R file that we’ll
discuss*\ `next <#Serve>`__\ *.*

::

   # Setup scoring function
   serve <- function() {
       app <- plumb(paste(prefix, 'plumber.R', sep='/'))
       app$run(host='0.0.0.0', port=8080)}

*Finally, a bit of logic to determine if, based on the options passed
when Amazon SageMaker Training or Hosting call this script, we are using
the container to train an algorithm or host a model.*

::

   # Run at start-up
   args <- commandArgs()
   if (any(grepl('train', args))) {
       train()}
   if (any(grepl('serve', args))) {
       serve()}

Serve
~~~~~

``plumber.R`` uses the `plumber <https://www.rplumber.io/>`__ package to
create a lightweight HTTP server for processing requests in hosting.
Note the specific syntax, and see the plumber help docs for additional
detail on more specialized use cases.

Per the Amazon SageMaker documentation, our service needs to accept post
requests to ping and invocations. plumber specifies this with custom
comments, followed by functions that take specific arguments.

Here invocations does most of the work, ingesting our trained model,
handling the HTTP request body, and producing a CSV output of
predictions.

::

   # plumber.R


   #' Ping to show server is there
   #' @get /ping
   function() {
       return('')}


   #' Parse input and return the prediction from the model
   #' @param req The http request sent
   #' @post /invocations
   function(req) {

       # Setup locations
       prefix <- '/opt/ml'
       model_path <- paste(prefix, 'model', sep='/')

       # Bring in model file and factor levels
       load(paste(model_path, 'mars_model.RData', sep='/'))

       # Read in data
       conn <- textConnection(gsub('\\\\n', '\n', req$postBody))
       data <- read.csv(conn)
       close(conn)

       # Convert input to model matrix
       scoring_X <- model.matrix(~., data, xlev=factor_levels)

       # Return prediction
       return(paste(predict(mars_model, scoring_X, row.names=FALSE), collapse=','))}

Dockerfile
~~~~~~~~~~

Smaller containers are preferred for Amazon SageMaker as they lead to
faster spin up times in training and endpoint creation, so this
container is kept minimal. It simply starts with Ubuntu, installs R,
mda, and plumber libraries, then adds ``mars.R`` and ``plumber.R``, and
finally runs ``mars.R`` when the entrypoint is launched.

.. code:: dockerfile

   FROM ubuntu:16.04

   MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

   RUN apt-get -y update && apt-get install -y --no-install-recommends \
       wget \
       r-base \
       r-base-dev \
       ca-certificates

   RUN R -e "install.packages(c('mda', 'plumber'), repos='https://cloud.r-project.org')"

   COPY mars.R /opt/ml/mars.R
   COPY plumber.R /opt/ml/plumber.R

   ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/mars.R", "--no-save"]

Publish
~~~~~~~

Now, to publish this container to ECR, we’ll run the comands below.

This command will take several minutes to run the first time.

.. code:: sh

    %%sh
    
    # The name of our algorithm
    algorithm_name=sagemaker-rmars
    
    #set -e # stop if anything fails
    
    account=$(aws sts get-caller-identity --query Account --output text)
    
    # Get the region defined in the current configuration (default to us-west-2 if none defined)
    region=$(aws configure get region)
    region=${region:-us-west-2}
    
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
    
    # If the repository doesn't exist in ECR, create it.
    
    aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
    
    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
    fi
    
    # Get the login command from ECR and execute it directly
    $(aws ecr get-login --region ${region} --no-include-email)
    
    # Build the docker image locally with the image name and then push it to ECR
    # with the full name.
    docker build  -t ${algorithm_name} .
    docker tag ${algorithm_name} ${fullname}
    
    docker push ${fullname}

--------------

Data
----

For this illustrative example, we’ll simply use ``iris``. This a
classic, but small, dataset used to test supervised learning algorithms.
Typically the goal is to predict one of three flower species based on
various measurements of the flowers’ attributes. Further detail can be
found `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`__.

Then let’s copy the data to S3.

.. code:: ipython3

    train_file = 'iris.csv'
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_file(train_file)

*Note: Although we could, we’ll avoid doing any preliminary
transformations on the data, instead choosing to do those
transformations inside the container. This is not typically the best
practice for model efficiency, but provides some benefits in terms of
flexibility.*

--------------

Train
-----

Now, let’s setup the information needed to train a Multivariate Adaptive
Regression Splines (MARS) model on iris data. In this case, we’ll
predict ``Sepal.Length`` rather than the more typical classification of
``Species`` to show how factors might be included in a model and limit
the case to regression.

First, we’ll get our region and account information so that we can point
to the ECR container we just created.

.. code:: ipython3

    region = boto3.Session().region_name
    account = boto3.client('sts').get_caller_identity().get('Account')

-  Specify the role to use
-  Give the training job a name
-  Point the algorithm to the container we created
-  Specify training instance resources (in this case our algorithm is
   only single-threaded so stick to 1 instance)
-  Point to the S3 location of our input data and the ``train`` channel
   expected by our algorithm
-  Point to the S3 location for output
-  Provide hyperparamters (keeping it simple)
-  Maximum run time

.. code:: ipython3

    r_job = 'DEMO-r-byo-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    
    print("Training job", r_job)
    
    r_training_params = {
        "RoleArn": role,
        "TrainingJobName": r_job,
        "AlgorithmSpecification": {
            "TrainingImage": '{}.dkr.ecr.{}.amazonaws.com/sagemaker-rmars:latest'.format(account, region),
            "TrainingInputMode": "File"
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m4.xlarge",
            "VolumeSizeInGB": 10
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://{}/{}/train".format(bucket, prefix),
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None"
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": "s3://{}/{}/output".format(bucket, prefix)
        },
        "HyperParameters": {
            "target": "Sepal.Length",
            "degree": "2"
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 60 * 60
        }
    }

Now let’s kick off our training job on Amazon SageMaker Training, using
the parameters we just created. Because training is managed (AWS takes
care of spinning up and spinning down the hardware), we don’t have to
wait for our job to finish to continue, but for this case, let’s setup a
waiter so we can monitor the status of our training.

.. code:: ipython3

    %%time
    
    sm = boto3.client('sagemaker')
    sm.create_training_job(**r_training_params)
    
    status = sm.describe_training_job(TrainingJobName=r_job)['TrainingJobStatus']
    print(status)
    sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=r_job)
    status = sm.describe_training_job(TrainingJobName=r_job)['TrainingJobStatus']
    print("Training job ended with status: " + status)
    if status == 'Failed':
        message = sm.describe_training_job(TrainingJobName=r_job)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')

--------------

Host
----

Hosting the model we just trained takes three steps in Amazon SageMaker.
First, we define the model we want to host, pointing the service to the
model artifact our training job just wrote to S3.

.. code:: ipython3

    r_hosting_container = {
        'Image': '{}.dkr.ecr.{}.amazonaws.com/sagemaker-rmars:latest'.format(account, region),
        'ModelDataUrl': sm.describe_training_job(TrainingJobName=r_job)['ModelArtifacts']['S3ModelArtifacts']
    }
    
    create_model_response = sm.create_model(
        ModelName=r_job,
        ExecutionRoleArn=role,
        PrimaryContainer=r_hosting_container)
    
    print(create_model_response['ModelArn'])

Next, let’s create an endpoing configuration, passing in the model we
just registered. In this case, we’ll only use a few c4.xlarges.

.. code:: ipython3

    r_endpoint_config = 'DEMO-r-byo-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print(r_endpoint_config)
    create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName=r_endpoint_config,
        ProductionVariants=[{
            'InstanceType': 'ml.m4.xlarge',
            'InitialInstanceCount': 1,
            'ModelName': r_job,
            'VariantName': 'AllTraffic'}])
    
    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

Finally, we’ll create the endpoints using our endpoint configuration
from the last step.

.. code:: ipython3

    %%time
    
    r_endpoint = 'DEMO-r-endpoint-' + time.strftime("%Y%m%d%H%M", time.gmtime())
    print(r_endpoint)
    create_endpoint_response = sm.create_endpoint(
        EndpointName=r_endpoint,
        EndpointConfigName=r_endpoint_config)
    print(create_endpoint_response['EndpointArn'])
    
    resp = sm.describe_endpoint(EndpointName=r_endpoint)
    status = resp['EndpointStatus']
    print("Status: " + status)
    
    try:
        sm.get_waiter('endpoint_in_service').wait(EndpointName=r_endpoint)
    finally:
        resp = sm.describe_endpoint(EndpointName=r_endpoint)
        status = resp['EndpointStatus']
        print("Arn: " + resp['EndpointArn'])
        print("Status: " + status)
    
        if status != 'InService':
            raise Exception('Endpoint creation did not succeed')

--------------

Predict
-------

To confirm our endpoints are working properly, let’s try to invoke the
endpoint.

*Note: The payload we’re passing in the request is a CSV string with a
header record, followed by multiple new lines. It also contains text
columns, which the serving code converts to the set of indicator
variables needed for our model predictions. Again, this is not a best
practice for highly optimized code, however, it showcases the
flexibility of bringing your own algorithm.*

.. code:: ipython3

    iris = pd.read_csv('iris.csv')
    
    runtime = boto3.Session().client('runtime.sagemaker')
    
    payload = iris.drop(['Sepal.Length'], axis=1).to_csv(index=False)
    response = runtime.invoke_endpoint(EndpointName=r_endpoint,
                                       ContentType='text/csv',
                                       Body=payload)
    
    result = json.loads(response['Body'].read().decode())
    result 

We can see the result is a CSV of predictions for our target variable.
Let’s compare them to the actuals to see how our model did.

.. code:: ipython3

    plt.scatter(iris['Sepal.Length'], np.fromstring(result[0], sep=','))
    plt.show()

--------------

Extensions
----------

This notebook showcases a straightforward example to train and host an R
algorithm in Amazon SageMaker. As mentioned previously, this notebook
could also be written in R. We could even train the algorithm entirely
within a notebook and then simply use the serving portion of the
container to host our model.

Other extensions could include setting up the R algorithm to train in
parallel. Although R is not the easiest language to build distributed
applications on top of, this is possible. In addition, running multiple
versions of training simultaneously would allow for parallelized grid
(or random) search for optimal hyperparamter settings. This would more
fully realize the benefits of managed training.

(Optional) Clean-up
~~~~~~~~~~~~~~~~~~~

If you’re ready to be done with this notebook, please run the cell
below. This will remove the hosted endpoint you created and avoid any
charges from a stray instance being left on.

.. code:: ipython3

    sm.delete_endpoint(EndpointName=r_endpoint)
