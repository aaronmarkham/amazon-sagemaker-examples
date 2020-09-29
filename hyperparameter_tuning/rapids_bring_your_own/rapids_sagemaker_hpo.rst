`Hyper Parameter
Optimization <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`__
(HPO) improves model quality by searching over hyperparameters,
parameters not typically learned during the training process but rather
values that control the learning process itself (e.g., model size,
learning rate, regularization). This search can significantly boost
model quality relative to default settings and non-expert tuning;
however, HPO can take a very long time on a non-accelerated platform. In
this notebook, we containerize a RAPIDS workflow and run
Bring-Your-Own-Container SageMaker HPO to show how we can overcome the
computational complexity of model search.

We accelerate HPO in two key ways: \* by *scaling within a node* (e.g.,
multi-GPU where each GPU brings a magnitude higher core count relative
to CPUs), and \* by *scaling across nodes* and running parallel trials
on cloud instances.

By combining these two powers HPO experiments that feel unapproachable
and may take multiple days on CPU instances can complete in just hours.
For example, we find a 12X speedup in wall clock time and a 4.5x
reduction in cost when comparing between GPU and CPU `EC2 Spot
instances <https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html>`__
on 100 XGBoost HPO trials using 10 parallel workers on 10 years of the
Airline Dataset (~63M flights) hosted in a S3 bucket. For additional
details refer to the end of the notebook.

With all these powerful tools at our disposal, every data scientist
should feel empowered to up-level their model before serving it to the
world!



Preamble

To get things rolling let’s make sure we can query our AWS SageMaker
execution role and session as well as our account ID and AWS region.

.. code:: ipython3

    import sagemaker
    from helper_functions import *
    working_directory = get_notebook_path()

.. code:: ipython3

    execution_role = sagemaker.get_execution_role()
    session = sagemaker.Session()
    
    account=!(aws sts get-caller-identity --query Account --output text)
    region=!(aws configure get region)

.. code:: ipython3

    account, region

Key Choices

Let’s go ahead and choose the configuration options for our HPO run.

Below are two reference configurations showing a small and a large scale
HPO (sized in terms of total experiments/compute).

The default values in the notebook are set for the small HPO
configuration, however you are welcome to scale them up.

   **small HPO**: 1_year, XGBoost, 3 CV folds, singleGPU, max_jobs = 10,
   max_parallel_jobs = 2

..

   **large HPO**: 10_year, XGBoost, 10 CV folds, multiGPU, max_jobs =
   100, max_parallel_jobs = 10

[ Dataset and S3 Bucket ]

We target a large real-world structured dataset of flight logs for US
airlines ( published monthly since 1987 by the Bureau of Transportation
`dataset
link <https://www.transtats.bts.gov/DatabaseInfo.asp?DB_ID=120&DB_URL=>`__,
additional details in Section 1.1).

We offer several sizes of this dataset in parquet (compressed column
storage) format: ``1_year`` (2019, 7.2M flights), ``3_year`` (2016-2019,
18M flights) or ``10_year`` (2009-2019, 63M flights).

.. code:: ipython3

    dataset_directory = '1_year'
    
    assert( dataset_directory in [ '1_year', '3_year', '10_year'] )

For demo purposes we host these pre-built datasets in public S3 buckets
in the **us-east-1** (N. Virginia) and **us-west-2** (Oregon).

Make sure you are in one of these two regions if you plan to use our
buckets since SageMaker requires that the S3 dataset and the compute
instacnes you’ll be renting are co-located. You are welcome to
remove/comment out the ``validate_region`` check if you plan to run on a
different region with your own dataset.

.. code:: ipython3

    data_bucket = 'sagemaker-rapids-hpo-' + region[0]
    s3_data_input = f"s3://{data_bucket}/{dataset_directory}"
    
    model_output_bucket = session.default_bucket()
    s3_model_output = f"s3://{model_output_bucket}/trained-models"
    
    validate_region(region[0])

[ Algorithm ]

From a ML/algorithm perspective, we offer
`XGBoost <https://xgboost.readthedocs.io/en/latest/#>`__ and
`RandomForest <https://docs.rapids.ai/api/cuml/stable/cuml_blogs.html#tree-and-forest-models>`__
decision tree models which do quite well on this structured dataset. You
are free to switch between these two algorithm choices and everything in
the example will continue to work.

.. code:: ipython3

    algorithm_choice = 'XGBoost'
    
    assert ( algorithm_choice in [ 'XGBoost', 'RandomForest' ])

We can also optionally increase robustness via reshuffles of the
train-test split (i.e., `cross-validation
folds <https://scikit-learn.org/stable/modules/cross_validation.html>`__).
Typical values here are between 3 and 10 folds.

.. code:: ipython3

    cv_folds = 3
    
    assert ( cv_folds >= 1 )

[ Compute Choice ]

We enable the option of running different code variations that unlock
increasing amounts of parallelism in the compute workflow.

-  ``singleCPU``\ \*\* = `pandas <https://pandas.pydata.org/>`__ +
   `sklearn <https://scikit-learn.org/stable/>`__
-  ``multiCPU`` = `dask <https://dask.org/>`__ +
   `pandas <https://pandas.pydata.org/>`__ +
   `sklearn <https://scikit-learn.org/stable/>`__

-  ``singleGPU`` = `cudf <https://github.com/rapidsai/cudf>`__ +
   `cuml <https://github.com/rapidsai/cuml>`__
-  ``multiGPU`` = `dask <https://dask.org/>`__ +
   `cudf <https://github.com/rapidsai/cudf>`__ +
   `cuml <https://github.com/rapidsai/cuml>`__

All of these code paths are integrated in the ``rapids_cloud_ml.py``
file for your reference. For some context, cuDF and cuML are part of the
RAPIDS library ecosystem. cuDF is a GPU accelerated dataframe library
aimed to mirror pandas, while cuML is a GPU accelerated machine-learning
library aimed to mirror sklearn.

   \**Note that the single-CPU option will leverage multiple cores in
   the model training portion of the workflow; however, to unlock full
   parallelism in each stage of the workflow we use
   `Dask <https://dask.org/>`__.

.. code:: ipython3

    code_choice = 'singleGPU' 
    
    assert ( code_choice in [ 'singleCPU', 'singleGPU', 'multiCPU', 'multiGPU'])

[ Search Ranges and Strategy ]

One of the most important choices when running HPO is to choose the
bounds of the hyperparameter search process. Below we’ve set the ranges
of the hyperparameters to allow for interesting variation, you are of
course welcome to revise these ranges based on domain knowledge
especially if you plan to plug in your own dataset.

   Note that we support additional algorithm specific parameters (refer
   to the ``parse_hyper_parameter_inputs`` function in
   ``rapids_cloud_ml.py``), but for demo purposes have limited our
   choice to the three parameters that overlap between the XGBoost and
   RandomForest algorithms. For more details see the documentation for
   `XGBoost
   parameters <https://xgboost.readthedocs.io/en/latest/parameter.html>`__
   and `RandomForest
   parameters <https://docs.rapids.ai/api/cuml/stable/api.html#random-forest>`__.

.. code:: ipython3

    hyperparameter_ranges = {
        'max_depth'    : sagemaker.parameter.IntegerParameter        ( 5, 15 ),
        'n_estimators' : sagemaker.parameter.IntegerParameter        ( 100, 500 ),
        'max_features' : sagemaker.parameter.ContinuousParameter     ( 0.1, 1.0 ),    
    } # see note above for adding additional parameters

.. code:: ipython3

    if 'XGBoost' in algorithm_choice: 
        # number of trees parameter name difference b/w XGBoost and RandomForest
        hyperparameter_ranges['num_boost_round'] = hyperparameter_ranges.pop('n_estimators')

We can also choose between a Random and Bayesian search strategy for
picking parameter combinations.

**Random Search**: Choose a random combination of values from within the
ranges for each training job it launches. The choice of hyperparameters
doesn’t depend on previous results so you can run the maximum number of
concurrent workers without affecting the performance of the search.

**Bayesian Search**: Make a guess about which hyperparameter
combinations are likely to get the best results. After testing the first
set of hyperparameter values, hyperparameter tuning uses regression to
choose the next set of hyperparameter values to test.

.. code:: ipython3

    search_strategy = 'Random'
    
    assert ( search_strategy in [ 'Random', 'Bayesian' ])

[ Experiment Scale ]

We also need to decide how may total experiments to run, and how many
should run in parallel. Below we have a very conservative number of
maximum jobs to run so that you don’t accidently spawn large
computations when starting out, however for meaningful HPO searches this
number should be much higher (e.g., in our experiments we often run 100
max_jobs). Note that you may need to request a `quota limit
increase <https://docs.aws.amazon.com/general/latest/gr/sagemaker.html>`__
for additional ``max_parallel_jobs`` parallel workers.

.. code:: ipython3

    max_jobs = 2

.. code:: ipython3

    max_parallel_jobs = 2

Let’s also set the max duration for an individual job to 24 hours so we
don’t have run-away compute jobs taking too long.

.. code:: ipython3

    max_duration_of_experiment_seconds = 60 * 60 * 24

[ Compute Platform ]

Based on the dataset size and compute choice we will try to recommend an
instance choice*, you are of course welcome to select alternate
configurations. > e.g., For the 10_year dataset option, we suggest
ml.p3.8xlarge instances (4 GPUs) and ml.m5.24xlarge CPU instances (
we’ll need upwards of 200GB CPU RAM during model training).

.. code:: ipython3

    instance_type = recommend_instance_type ( code_choice, dataset_directory  ) 

In addition to choosing our instance type, we can also enable
significant savings by leveraging `AWS EC2 Spot
Instances <https://aws.amazon.com/ec2/spot/>`__.

We **highly recommend** that you set this flag to ``True`` as it
typically leads to 60-70% cost savings. Note, however that you may need
to request a `quota limit
increase <https://docs.aws.amazon.com/general/latest/gr/sagemaker.html>`__
to enable Spot instances in SageMaker.

.. code:: ipython3

    use_spot_instances_flag = True

Validate

.. code:: ipython3

    summarize_choices( s3_data_input, s3_model_output, code_choice, algorithm_choice, 
                       cv_folds, instance_type, use_spot_instances_flag, search_strategy, 
                       max_jobs, max_parallel_jobs, max_duration_of_experiment_seconds )

1. ML Workflow



1.1 - Dataset

In this demo we’ll utilize the Airline dataset (Carrier On-Time
Performance 1987-2020, available from the `Bureau of Transportation
Statistics <https://transtats.bts.gov/Tables.asp?DB_ID=120&DB_Name=Airline%20On-Time%20Performance%20Data&DB_Short_Name=On-Time#>`__).

The public dataset contains logs/features about flights in the United
States (17 airlines) including:

-  Locations and distance ( ``Origin``, ``Dest``, ``Distance`` )
-  Airline / carrier ( ``Reporting_Airline`` )
-  Scheduled departure and arrival times ( ``CRSDepTime`` and
   ``CRSArrTime`` )
-  Actual departure and arrival times ( ``DpTime`` and ``ArrTime`` )
-  Difference between scheduled & actual times ( ``ArrDelay`` and
   ``DepDelay`` )
-  Binary encoded version of late, aka our target variable (
   ``ArrDelay15`` )

Using these features we’ll build a classifier model to predict whether a
flight is going to be more than 15 minutes late on arrival as it
prepares to depart.

1.2 - Python ML Workflow

To build a RAPIDS enabled SageMaker HPO we first need to build an
Estimator. An Estimator is a container image that captures all the
software needed to run an HPO experiment. The container is augmented
with entrypoint code that will be trggered at runtime by each worker.
The entrypoint code enables us to write custom models and hook them up
to data.

In order to work with SageMaker HPO, the entrypoint logic should parse
hyperparameters (supplied by AWS SageMaker), load and split data, build
and train a model, score/evaluate the trained model, and emit an output
representing the final score for the given hyperparameter setting. We’ve
already built multiple variations of this code.

If you would like to make changes by adding your custom model logic feel
free to modify the **train.py** and **rapids_cloud_ml.py** files in the
code directory. You are also welcome to uncomment the cells below to
load the read/review the code.

.. code:: ipython3

    # %load code/train.py

.. code:: ipython3

    # %load code/rapids_cloud_ml.py

2. Build Estimator



As we’ve already mentioned, the SageMaker Estimator represents the
containerized software stack that AWS SageMaker will replicate to each
worker node.

The first step to building our Estimator, is to augment a RAPIDS
container with our ML Workflow code from above, and push this image to
Amazon Elastic Cloud Registry so it is available to SageMaker.

For additional options and details see the `Estimator
documentation <https://sagemaker.readthedocs.io/en/stable/estimators.html#sagemaker.estimator.Estimator>`__.

2.1 - Containerize and Push to ECR

Now let’s turn to building our container so that it can integrate with
the AWS SageMaker HPO API.

Our container can either be built on top of the latest RAPIDS [ nightly
] image as a starting layer or the RAPIDS stable image.

.. code:: ipython3

    rapids_stable = 'rapidsai/rapidsai:0.14-cuda10.1-runtime-ubuntu18.04-py3.7'
    rapids_nightly = 'rapidsai/rapidsai-nightly:0.15-cuda10.1-runtime-ubuntu18.04-py3.7'
    
    rapids_base_container = rapids_stable
    assert ( rapids_base_container in [ rapids_stable, rapids_nightly ] )

Let’s also decide on the full name of our container.

.. code:: ipython3

    image_base = 'cloud-ml-sagemaker'
    image_tag  = rapids_base_container.split(':')[1]

.. code:: ipython3

    ecr_fullname = f"{account[0]}.dkr.ecr.{region[0]}.amazonaws.com/{image_base}:{image_tag}"

.. code:: ipython3

    ecr_fullname

2.1.1 - Write Dockerfile

We write out the Dockerfile in this cell, write it to disk, and in the
next cell execute the docker build command.

First, let’s switch our working directory to the location of the
Estimator entrypoint and library code.

.. code:: ipython3

    %cd code

Let’s now write our selected RAPDIS image layer as the first FROM
statement in the the Dockerfile.

.. code:: ipython3

    with open('Dockerfile', 'w') as dockerfile_handle: 
        dockerfile_handle.writelines( 'FROM ' + rapids_base_container + '\n')

Next let’s write the remaining pieces of the Dockerfile, namely adding
the sagemaker-training-toolkit and copying our python code.

.. code:: ipython3

    %%writefile -a Dockerfile
    
    # install https://github.com/aws/sagemaker-training-toolkit
    RUN apt-get update && apt-get install -y --no-install-recommends build-essential \ 
        && source activate rapids && pip3 install sagemaker-training
    
    # path where sagemaker looks for our code
    ENV CLOUD_PATH="/opt/ml/code"
    
    # copy our latest [local] code into the container 
    COPY rapids_cloud_ml.py $CLOUD_PATH/rapids_cloud_ml.py
    COPY train.py $CLOUD_PATH/train.py
    
    # sagemaker entrypoint will be train.py
    ENV SAGEMAKER_PROGRAM train.py 
    
    WORKDIR $CLOUD_PATH

Lastly, let’s ensure that our Dockerfile correctly captured our base
image selection.

.. code:: ipython3

    validate_dockerfile( rapids_base_container )
    !cat Dockerfile

2.1.2 Build and Tag

The build step will be dominated by the download of the RAPIDS image
(base layer). If it’s already been downloaded the build will take less
than 1 minute.

.. code:: ipython3

    !docker pull $rapids_base_container

.. code:: ipython3

    %%time
    !docker build . -t $ecr_fullname -f Dockerfile

2.1.3 - Publish to Elastic Cloud Registry (ECR)

Now that we’ve built and tagged our container its time to push it to
Amazon’s container registry (ECR). Once in ECR, AWS SageMaker will be
able to leverage our image to build Estimators and run experiments.

Docker Login to ECR

.. code:: ipython3

    docker_login_str = !(aws ecr get-login --region {region[0]} --no-include-email)

.. code:: ipython3

    !{docker_login_str[0]}

Create ECR repository [ if it doesn’t already exist]

.. code:: ipython3

    repository_query = !(aws ecr describe-repositories --repository-names $image_base)
    if repository_query[0] == '':
        !(aws ecr create-repository --repository-name $image_base)

Let’s now actually push the container to ECR > Note the first push to
ECR may take some time (hopefully less than 10 minutes).

.. code:: ipython3

    ecr_fullname

.. code:: ipython3

    !docker push $ecr_fullname

2.2 - Create Estimator

Having built our container [ +custom logic] and pushed it to ECR, we can
finally compile all of efforts into an Estimator instance.

.. code:: ipython3

    estimator_params = {
        'image_name' : ecr_fullname,
        
        'train_use_spot_instances': use_spot_instances_flag,
        'train_instance_type' : instance_type,
        'train_instance_count' : 1,
        
        'train_max_run'  : max_duration_of_experiment_seconds, # 24 hours 
        
        'input_mode'  : 'File',
        'output_path' : s3_model_output,
        
        'sagemaker_session' : session,
        'role' : execution_role,
    }
    
    if use_spot_instances_flag == True:
        estimator_params.update ( { 'train_max_wait' : max_duration_of_experiment_seconds + 1 })

.. code:: ipython3

    estimator = sagemaker.estimator.Estimator( **estimator_params  )

2.3 - Test Estimator

Now we are ready to test by asking SageMaker to run the BYOContainer
logic inside our Estimator. This is a useful step if you’ve made changes
to your custom logic and are interested in making sure everything works
before launching a large HPO search.

   Note: This verification step will use the default hyperparameter
   values declared in our custom train code, as SageMaker HPO will not
   be orchestrating a search for this single run.

.. code:: ipython3

    summarize_choices( s3_data_input, s3_model_output, code_choice, algorithm_choice, 
                       cv_folds, instance_type, use_spot_instances_flag, search_strategy, 
                       max_jobs, max_parallel_jobs, max_duration_of_experiment_seconds )

.. code:: ipython3

    assert ( input('confirm test run? [ y / n ] : ').lower() == 'y' )
    
    job_name = new_job_name_from_config( dataset_directory, code_choice, 
                                         algorithm_choice, cv_folds,
                                         instance_type  )
    
    estimator.fit( inputs = s3_data_input, job_name = job_name.lower() )

3. Run HPO

With a working SageMaker Estimator in hand, the hardest part is behind
us. In the key choices section we already defined our search strategy
and hyperparameter ranges, so all that remains is to choose a metric to
evaluate performance on. For more documentation check out the AWS
SageMaker `Hyperparameter Tuner
documentation <https://sagemaker.readthedocs.io/en/stable/tuner.html>`__.



3.1 - Define Metric

We only focus on a single metric, which we call ‘final-score’, that
captures the accuracy of our model on the test data unseen during
training. You are of course welcome to add aditional metrics, see `AWS
SageMaker documentation on
Metrics <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-metrics.html>`__.
When defining a metric we provide a regular expression (i.e., string
parsing rule) to extract the key metric from the output of each
Estimator/worker.

.. code:: ipython3

    metric_definitions = [{'Name': 'final-score', 'Regex': 'final-score: (.*);'}]

.. code:: ipython3

    objective_metric_name = 'final-score'

3.2 - Define Tuner

Finally we put all of the elements we’ve been building up together into
a HyperparameterTuner declaration.

.. code:: ipython3

    hpo = sagemaker.tuner.HyperparameterTuner( estimator = estimator,
                                               metric_definitions = metric_definitions, 
                                               objective_metric_name = objective_metric_name,
                                               objective_type = 'Maximize',
                                               hyperparameter_ranges = hyperparameter_ranges,
                                               strategy = search_strategy,  
                                               max_jobs = max_jobs,
                                               max_parallel_jobs = max_parallel_jobs)

3.3 - Run HPO

.. code:: ipython3

    summarize_choices( s3_data_input, s3_model_output, code_choice, algorithm_choice, 
                       cv_folds, instance_type, use_spot_instances_flag, search_strategy, 
                       max_jobs, max_parallel_jobs, max_duration_of_experiment_seconds )

Let’s be sure we take a moment to confirm before launching all of our
HPO experiments. Depending on your configuration options running this
cell can kick off a massive amount of computation! > Once this process
begins, we recommend that you use the SageMaker UI to keep track of the
health of the HPO process and the individual workers.

.. code:: ipython3

    assert ( input('confirm HPO launch? [ y / n ] : ').lower() == 'y' )
    
    tuning_job_name = new_job_name_from_config( dataset_directory, code_choice, 
                                                algorithm_choice, cv_folds, 
                                                instance_type )
    hpo.fit( inputs = s3_data_input, 
             job_name = tuning_job_name, 
             wait = True, logs = 'All')
    
    hpo.wait() # block until the .fit call above is completed

3.4 - Results and Summary

Once your job is complete there are multiple ways to analyze the
results. Below we display the performance of the best job, as well
printing each HPO trial/job as a row of a dataframe.

.. code:: ipython3

    hpo_results = summarize_hpo_results ( tuning_job_name )

.. code:: ipython3

    sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name).dataframe()

For a more in depth look at the HPO process we invite you to check out
the HPO_Analyze_TuningJob_Results.ipynb notebook which shows how we can
explore interesting things like the impact of each individual
hyperparameter on the performance metric.

3.5 - Getting the best Model

Next let’s download the best trained model from our HPO runs.

.. code:: ipython3

    %cd $working_directory

.. code:: ipython3

    local_filename = download_best_model( model_output_bucket, s3_model_output, 
                                          hpo_results, working_directory )

3.6 - Model Serving

With your best model in hand, you can now move on to serving this model.

The `SageMaker Inference
Toolkit <https://github.com/aws/sagemaker-inference-toolkit>`__
implements a model serving stack and can be easily added to any Docker
container, making it deployable to SageMaker. This library’s serving
stack is built on Multi Model Server, and it can serve your own models
or those you trained on SageMaker using machine learning frameworks with
native SageMaker support.

We’ll leave you with pointers to documentation in case you want to go
further, however a full implementation is out of scope for our HPO
notebook.

   Note that the best model we just downloaded is stored in a
   `Treelite <https://treelite.readthedocs.io/en/latest/>`__ format and
   can run optimized inference using the GPU or the CPU.

Summary

We’ve now successfully built a RAPIDS ML workflow, containerized it (as
a SageMaker Estimator), and launched a set of HPO experiments to find
the best hyperparamters for our model.

If you are curious to go further, we invite you to plug in your own
dataset and tweak the configuration settings to find your champion
model!

**HPO Experiment Details**

As mentioned in the introduction we find a 12X speedup in wall clock
time and a 4.5x reduction in cost when comparing between GPU and CPU
instances on 100 HPO trials using 10 parallel workers on 10 years of the
Airline Dataset (~63M flights). In these experiments we used the XGBoost
algorithm with the multi-GPU vs multi-CPU Dask cluster and 10 cross
validaiton folds. Below we offer a table with additional details.



In the case of the CPU runs, 12 jobs were stopped since they exceeded
the 24 hour limit we set. CPU Job Summary Image

In the case of the GPU runs, no jobs were stopped. GPU Job Summary Image

Note that in both cases 1 job failed because a spot instance was
terminated. But 1 failed job out of 100 is a minimal tradeoff for the
significant cost savings.

Rapids References

   `cloud-ml-examples <http://github.com/rapidsai/cloud-ml-examples>`__

..

   `RAPIDS HPO <https://rapids.ai/hpo>`__

   `cuML Documentation <https://docs.rapids.ai/api/cuml/stable/>`__

SageMaker References

   `SageMaker Training
   Toolkit <https://github.com/aws/sagemaker-training-toolkit>`__

..

   `Estimator
   Parameters <https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html>`__

   Spot Instances
   `docs <https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html>`__,
   and `blog <>`__
