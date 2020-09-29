Using R in SageMaker Processing
===============================

Amazon SageMaker Processing is a capability of Amazon SageMaker that
lets you easily run your preprocessing, postprocessing and model
evaluation workloads on fully managed infrastructure. In this example,
we’ll see how to use SageMaker Processing with the R programming
language.

The workflow for using R with SageMaker Processing involves the
following steps:

-  Writing a R script.
-  Building a Docker container.
-  Creating a SageMaker Processing job.
-  Retrieving and viewing job results.

The R script
------------

To use R with SageMaker Processing, first prepare a R script similar to
one you would use outside SageMaker. Below is the R script we’ll be
using. It performs operations on data and also saves a .png of a plot
for retrieval and display later after the Processing job is complete.
This enables you to perform any kind of analysis and feature engineering
at scale with R, and also create visualizations for display anywhere.

.. code:: ipython3

    %%writefile preprocessing.R
    
    library(readr)
    library(dplyr)
    library(ggplot2)
    library(forcats)
    
    input_dir <- "/opt/ml/processing/input/"
    filename <- Sys.glob(paste(input_dir, "*.csv", sep=""))
    df <- read_csv(filename)
    
    plot_data <- df %>%
      group_by(state) %>%
      count()
    
    write_csv(plot_data, "/opt/ml/processing/csv/plot_data.csv")
    
    plot <- plot_data %>% 
      ggplot()+
      geom_col(aes(fct_reorder(state, n), 
                   n, 
                   fill = n))+
      coord_flip()+
      labs(
        title = "Number of people by state",
        subtitle = "From US-500 dataset",
        x = "State",
        y = "Number of people"
      )+ 
      theme_bw()
    
    ggsave("/opt/ml/processing/images/census_plot.png", width = 10, height = 8, dpi = 100)

Building a Docker container
---------------------------

Next, there is a one-time step to create a R container. For subsequent
SageMaker Processing jobs, you can just reuse this container (unless you
need to add further dependencies, in which case you can just add them to
the Dockerfile and rebuild). To start, set up a local directory for
Docker-related files.

.. code:: ipython3

    !mkdir docker

A simple Dockerfile can be used to build a Docker container for
SageMaker Processing. For this example, we’ll use a parent Docker image
from the Rocker Project, which provides a set of convenient R Docker
images. There is no need to include your R script in the container
itself because SageMaker Processing will ingest it for you. This gives
you the flexibility to modify the script as needed without having to
rebuild the Docker image every time you modify it.

.. code:: ipython3

    %%writefile docker/Dockerfile
    
    FROM rocker/tidyverse:latest
    
    # tidyverse has all the packages we need, otherwise we could install more as follows
    # RUN install2.r --error \
    #    jsonlite \
    #    tseries
    
    ENTRYPOINT ["Rscript"]

The Dockerfile is now used to build the Docker image. We’ll also create
an Amazon Elastic Container Registry (ECR) repository, and push the
image to ECR so it can be accessed by SageMaker.

.. code:: ipython3

    import boto3
    
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name
    
    ecr_repository = 'r-in-sagemaker-processing'
    tag = ':latest'
    
    uri_suffix = 'amazonaws.com'
    processing_repository_uri = '{}.dkr.ecr.{}.{}/{}'.format(account_id, region, uri_suffix, ecr_repository + tag)
    
    # Create ECR repository and push Docker image
    !docker build -t $ecr_repository docker
    !$(aws ecr get-login --region $region --registry-ids $account_id --no-include-email)
    !aws ecr create-repository --repository-name $ecr_repository
    !docker tag {ecr_repository + tag} $processing_repository_uri
    !docker push $processing_repository_uri

Creating a SageMaker Processing job
-----------------------------------

With our Docker image in ECR, we now prepare for the SageMaker
Processing job by specifying Amazon S3 buckets for output and input, and
downloading the raw dataset.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    
    role = get_execution_role()
    session = sagemaker.Session()
    s3_output = session.default_bucket()
    s3_prefix = 'R-in-Processing'
    s3_source = 'sagemaker-workshop-pdx'
    session.download_data(path='./data', bucket=s3_source, key_prefix='R-in-Processing/us-500.csv')

Before setting up the SageMaker Processing job, the raw dataset is
uploaded to S3 so it is accessible to SageMaker Processing.

.. code:: ipython3

    rawdata_s3_prefix = '{}/data/raw'.format(s3_prefix)
    raw_s3 = session.upload_data(path='./data', key_prefix=rawdata_s3_prefix)
    print(raw_s3)

The ``ScriptProcessor`` class of the SageMaker SDK lets you run a
command inside a Docker container. We’ll use this to run our own script
using the ``Rscript`` command. In the ``ScriptProcessor`` you also can
specify the type and number of instances to be used in the SageMaker
Processing job.

.. code:: ipython3

    from sagemaker.processing import ScriptProcessor
    
    script_processor = ScriptProcessor(command=['Rscript'],
                    image_uri=processing_repository_uri,
                    role=role,
                    instance_count=1,
                    instance_type='ml.c5.xlarge')

We can now start the SageMaker Processing job. The main aspects of the
code below are specifying the input and output locations, and the name
of our R preprocessing script.

.. code:: ipython3

    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from time import gmtime, strftime 
    
    processing_job_name = "R-in-Processing-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    output_destination = 's3://{}/{}/data'.format(s3_output, s3_prefix)
    
    script_processor.run(code='preprocessing.R',
                          job_name=processing_job_name,
                          inputs=[ProcessingInput(
                            source=raw_s3,
                            destination='/opt/ml/processing/input')],
                          outputs=[ProcessingOutput(output_name='csv',
                                                    destination='{}/csv'.format(output_destination),
                                                    source='/opt/ml/processing/csv'),
                                   ProcessingOutput(output_name='images',
                                                    destination='{}/images'.format(output_destination),
                                                    source='/opt/ml/processing/images')])
    
    preprocessing_job_description = script_processor.jobs[-1].describe()

Retrieving and viewing job results
----------------------------------

From the SageMaker Processing job description, we can look up the S3
URIs of the output, including the output plot .png file.

.. code:: ipython3

    output_config = preprocessing_job_description['ProcessingOutputConfig']
    for output in output_config['Outputs']:
        if output['OutputName'] == 'csv':
            preprocessed_csv_data = output['S3Output']['S3Uri']
        if output['OutputName'] == 'images':
            preprocessed_images = output['S3Output']['S3Uri']

Now we can display the plot produced by the SageMaker Processing job. A
similar workflow applies to retrieving and working with any other output
from a job, such as the transformed data itself.

.. code:: ipython3

    from PIL import Image
    from IPython.display import display
    
    plot_key = 'census_plot.png'
    plot_in_s3 = '{}/{}'.format(preprocessed_images, plot_key)
    !aws s3 cp {plot_in_s3} .
    im = Image.open(plot_key)
    display(im)
