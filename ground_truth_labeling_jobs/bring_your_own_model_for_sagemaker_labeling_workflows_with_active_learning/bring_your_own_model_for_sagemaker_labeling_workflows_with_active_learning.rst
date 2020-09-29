Part 1: Create Resources Needed for an Active Learning Workflow
---------------------------------------------------------------

Use this part of the notebook to create the resources required to create
an automated labeling workflow for a text-classification labeling job.
Specifically, we will create:

-  An input manifest file using the UCI News Dataset with 20% of the
   data labeled
-  A CreateLabelingJob request

**This notebook is intended to be used along side the blog
post**\ `Bring your own model for SageMaker labeling workflows with
Active
Learning <https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-for-amazon-sagemaker-labeling-workflows-with-active-learning/>`__\ **,
Part 1: Create an Active Learning Workflow with BlazingText**.

While following along with this blog post, we recommend that you leave
most of the cells unmodified. However, the notebook will indicate where
you can modify variables to create the resources needed for a custom
labeling job.

If you plan to customize the Ground Truth labeling job request
configuration below, you will also need the resources required to create
a labeling job. For more information, see `Use Amazon SageMaker Ground
Truth for Data
Labeling <https://docs.aws.amazon.com/sagemaker/latest/dg/sms.html>`__.

Using this Notebook
~~~~~~~~~~~~~~~~~~~

Please set the kernel to *conda_tensorflow_p36* when running this
notebook.

Run the code cells in this notebook to configure a Labeling Job request
in JSON format. This request JSON can be used in an active learning
workflow and will determine how your labeling job task appears to human
workers.

To customize this notebook, you will need to modify the the cells below
and configure the Ground Truth labeling job request
(``human_task_config``) to meet your requirements. To learn how to
create a Ground Truth labeling job using the Amazon SageMaker API, see
`CreateLabelingJob <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateLabelingJob.html>`__.

First, we will set up our environment.

.. code:: ipython3

    import os,sys,sagemaker, tensorflow as tf, pandas as pd, boto3, numpy as np
    from sagemaker import get_execution_role
    from sagemaker.tensorflow import TensorFlow
    
    sess = sagemaker.Session()
    
    role = get_execution_role()
    region = sess.boto_session.region_name
    bucket= sess.default_bucket(); key='sagemaker-byoal'

Prepare labeling input manifest file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will create an input manifest file for our active learning workflow
using the newsCorpora.csv file from the `UCI News
Dataset <https://archive.ics.uci.edu/ml/datasets/News+Aggregator>`__.
This dataset contains a list of about 420,000 articles that fall into
one of four categories: Business (b), Science & Technology (t),
Entertainment (e) and Health & Medicine (m). We will randomly choose
10,000 articles from that file to create our dataset.

For the active learning loop to start, 20% of the data must be labeled.
To quickly test the active learning component, we will include 20%
(``labeled_count``) of the original labels provided in the dataset in
our input manifest. We use this partially-labeled dataset as the input
to the active learning loop.

.. code:: ipython3

    ! wget https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip --no-check-certificate && unzip NewsAggregatorDataset.zip

.. code:: ipython3

    column_names = ["TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
    manifest_file = "partially-labeled.manifest"
    news_data_all = pd.read_csv('newsCorpora.csv', names=column_names, header=None, delimiter='\t')
    news_data = news_data_all.sample(n=10000, random_state=42)
    news_data = news_data[["TITLE","CATEGORY"]]

We will clean our data set using *pandas*.

.. code:: ipython3

    news_data["TITLE"].replace('"','',inplace=True,regex=True)
    news_data["TITLE"].replace('[^\w\s]','',inplace=True,regex=True)
    news_data["TITLE"] = news_data["TITLE"].str.split('\n').str[0]
    news_data['CATEGORY'] = news_data['CATEGORY'].astype("category").cat.codes

.. code:: ipython3

    fixed = news_data["TITLE"].str.lower().replace('"','')

.. code:: ipython3

    news_data.to_csv("news_subset.csv", index=False)

The following cell will create our partially-labeled input manifest
file, and push it to our S3 bucket.

.. code:: ipython3

    import json
    
    total=len(news_data)
    labeled_count = int(total / 5) #20% of the dataset is labeled.
    label_map = {
                 "b": "Business",
                 "e": "Entertainment",
                 "m": "Health & Medicine",
                 "t": "Science and Technology"
              }
    labeled_series=pd.Series(data=news_data.iloc[:labeled_count].TITLE.values,index=news_data.iloc[:labeled_count].CATEGORY.values)
    annotation_metadata = b"""{ "category-metadata" : { "confidence": 1.0, "human-annotated": "yes", "type": "groundtruth/text-classification"} }"""
    annotation_metadata_dict = json.loads(annotation_metadata)
    with open(manifest_file, 'w') as outfile:
        for items in labeled_series.iteritems():
            labeled_record = dict()
            labeled_record["source"] = items[1]
            labeled_record["category"] =  int(items[0])
            labeled_record.update(annotation_metadata_dict)
            outfile.write(json.dumps(labeled_record) + "\n")
    
    unlabeled_series=pd.Series(data=news_data.iloc[labeled_count:].TITLE.values,index=news_data.iloc[labeled_count:].CATEGORY.values)
    with open(manifest_file, 'a') as outfile:
        for items in unlabeled_series.iteritems():
            outfile.write("{\"source\":\""+items[1]+"\"}\n")    
        
    boto3.resource('s3').Bucket(bucket).upload_file(manifest_file,key+ "/" + manifest_file)
    manifest_file_uri =  "s3://{}/{}".format(bucket,key+ "/" + manifest_file)

.. code:: ipython3

    # Use s3 client to upload relevant json strings to s3.
    s3_client = boto3.client('s3')

This cell will specify the labels that workers will use to categorize
the articles. To customize your labeling job, add your own labels here.
To learn more, see
`LabelCategoryConfigS3Uri <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateLabelingJob.html#sagemaker-CreateLabelingJob-request-LabelCategoryConfigS3Uri>`__.

.. code:: ipython3

    label_file_name = "class_labels.json"
    label_file = """{
        "document-version": "2018-11-28",
        "labels": [
            {
                "label": "Business"
            },
            {
                "label": "Entertainment"
            },
            {
                "label": "Health & Medicine"
            },
            {
                "label": "Science and Technology"
            }
        ]
    }"""
    
    s3_client.put_object(Body=label_file, Bucket=bucket, Key=key+ "/" + label_file_name)
    label_file_uri =  "s3://{}/{}".format(bucket,key+ "/" + label_file_name)

The following cell will specify our custom worker task template. This
template will configure the UI that workers will see when they open our
text classification labeling job tasks. To learn how to customize this
cell, see `Creating your custom labeling task
template <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step2.html>`__.

.. code:: ipython3

    template_file_name = "instructions.template"
    template_file = r"""
    <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
    <crowd-form>
      <crowd-classifier
        name="crowd-classifier"
        categories="{{ task.input.labels | to_json | escape }}"
        header="Select the news title corresponding to the 4 categories. (b) for Business, (e) for Entertainment, (m) for Health and Medicine and (t) for Science and Technology."
      >
        <classification-target> {{ task.input.taskObject }} </classification-target>
        <full-instructions header="Classifier instructions">
          <ol><li><strong>Read</strong> the text carefully.</li><li><strong>Read</strong> the examples to understand more about the options.</li><li><strong>Choose</strong> the appropriate label that best suits the text.</li></ol>
        </full-instructions>
        <short-instructions>
          <p>Example Business title:</p><p>US open: Stocks fall after Fed official hints at accelerated tapering.</p><p><br>
          </p><p>Example Entertainment title:</p><p>CBS negotiates three more seasons for The Big Bang Theory</p><p><br>
          </p><p>Example Health & Medicine title:</p><p>Blood Test Could Predict Alzheimer's. Good News? </p><p><br>
          </p><p>Example Science and Technology (t) title:</p><p>Elephants tell human friend from foe by voice.</p><p><br>
          </p>
        </short-instructions>
      </crowd-classifier>
    </crowd-form>
    """
    
    s3_client.put_object(Body=template_file, Bucket=bucket, Key=key+ "/" + template_file_name)
    template_file_uri =  "s3://{}/{}".format(bucket,key+ "/" + template_file_name)

To use a private work team to labeling your data objects, set
``USE_PRIVATE_WORKFORCE`` to ``True`` and input your work team ARN for
``private_workteam_arn``. You must have a private workforce in the same
AWS Region as your labeling job task request to use a private work team.
To learn more see `Use a Private
Workforce <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-private.html>`__

.. code:: ipython3

    USE_PRIVATE_WORKFORCE = False
    private_workteam_arn = ''

This cell will automatically configure a public workforce ARN and pre-
and post-annotation ARNs (``prehuman_arn`` and ``acs_arn``
respectively). If ``USE_PRIVATE_WORKFORCE`` is ``False`` a public
workforce will be used to create your labeling job request.

To customize your labeling job task type, you will need to modify
``prehuman_arn`` and ``acs_arn``.

If you are using one of the Ground Truth built-in task types, you can
find pre- and post-annotation lambda ARNs using the following links. \*
Pre-annotation lambda ARNs for built in task types can be found in
`HumanTaskConfig <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_HumanTaskConfig.html#API_HumanTaskConfig_Contents>`__.
\* Post-annotation lambda ARNs (Annotation Consolidation Lambda) for
built in task types can be found in
`AnnotationConsolidationConfig <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AnnotationConsolidationConfig.html#sagemaker-Type-AnnotationConsolidationConfig-AnnotationConsolidationLambdaArn>`__.

If you are creating a custom labeling job task, see `Step 3: Processing
with AWS
Lambda <https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html>`__
learn how to create custom pre- and post-annotation lambda ARNs.

.. code:: ipython3

    # Specify ARNs for resources needed to run a text classification job.
    ac_arn_map = {'us-west-2': '081040173940',
                  'us-east-1': '432418664414',
                  'us-east-2': '266458841044',
                  'eu-west-1': '568282634449',
                  'ap-northeast-1': '477331159723'}
    
    public_workteam_arn = 'arn:aws:sagemaker:{}:394669845002:workteam/public-crowd/default'.format(region)
    prehuman_arn = 'arn:aws:lambda:{}:{}:function:PRE-TextMultiClass'.format(region, ac_arn_map[region])
    acs_arn = 'arn:aws:lambda:{}:{}:function:ACS-TextMultiClass'.format(region, ac_arn_map[region])

The following cell specifies our labeling job name, the description
workers see, and tags that workers can use to find our labeling job
task.

.. code:: ipython3

    job_name_prefix = "byoal-news"
    task_description = 'Classify news title to one of these 4 categories.'
    task_keywords = ['text', 'classification', 'humans', 'news']
    task_title = task_description

Modify the following request to customize your labeling job request. For
more information on the parameters below, see
`CreateLabelingJob <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateLabelingJob.html>`__.

.. code:: ipython3

    human_task_config = {
          "AnnotationConsolidationConfig": {
            "AnnotationConsolidationLambdaArn": acs_arn,
          },
          "PreHumanTaskLambdaArn": prehuman_arn,
          "MaxConcurrentTaskCount": 200, # 200 texts will be sent at a time to the workteam.
          "NumberOfHumanWorkersPerDataObject": 1, # 1 workers will be enough to label each text.
          "TaskAvailabilityLifetimeInSeconds": 21600, # Your work team has 6 hours to complete all pending tasks.
          "TaskDescription": task_description,
          "TaskKeywords": task_keywords,
          "TaskTimeLimitInSeconds": 300, # Each text must be labeled within 5 minutes.
          "TaskTitle": task_title,
          "UiConfig": {
            "UiTemplateS3Uri": template_file_uri,
          }
        }
    
    if not USE_PRIVATE_WORKFORCE:
        human_task_config["PublicWorkforceTaskPrice"] = {
            "AmountInUsd": {
               "Dollars": 0,
               "Cents": 1,
               "TenthFractionsOfACent": 2,
            }
        } 
        human_task_config["WorkteamArn"] = public_workteam_arn
    else:
        human_task_config["WorkteamArn"] = private_workteam_arn
    
    ground_truth_request = {
            "InputConfig" : {
              "DataSource": {
                "S3DataSource": {
                  "ManifestS3Uri": manifest_file_uri,
                }
              },
              "DataAttributes": {
                "ContentClassifiers": [
                  "FreeOfPersonallyIdentifiableInformation",
                  "FreeOfAdultContent"
                ]
              },  
            },
            "OutputConfig" : {
              "S3OutputPath": 's3://{}/{}/output/'.format(bucket, key),
            },
            "HumanTaskConfig" : human_task_config,
            "LabelingJobNamePrefix": job_name_prefix,
            "RoleArn": role, 
            "LabelAttributeName": "category",
            "LabelCategoryConfigS3Uri": label_file_uri,
        }
        


.. code:: ipython3

    print(json.dumps(ground_truth_request, indent=2))

Do the following steps to trigger the Active Learning loop.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open the AWS Step Functions console:
   http://console.aws.amazon.com/states
2. The Cloud Formation stack provided in the blog post has generated two
   step function in the State Machines section:
   **ActiveLearningLoop-**\ \* and **ActiveLearning-**\ \* where \* will
   be replaced with the name you used when you launched your Cloud
   Formation stack.
3. Select **ActiveLearningLoop**-*.
4. Choose **Start Execution**.
5. Paste the JSON above in **Input – optional code-block**.
6. Select **Start execution**.

These manual steps could be automated by using the data science SDK.
Please refer to the details
`here <https://aws.amazon.com/about-aws/whats-new/2019/11/introducing-aws-step-functions-data-science-sdk-amazon-sagemaker/>`__
for more information.

On successful completion of the active learning loop, the state machine
will output the final output manifest file and the latest trained model
output.

Part 2: Bring Your Own Model to an Active Learning Workflow
-----------------------------------------------------------

Use this part of the notebook to learn how to containerize your own
Machine Learning model and push it to `Amazon Elastic Container Registry
(ERC) <https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html>`__.
This notebook will produce an ECR ID that you can use to integrate your
model into an active learning workflow.

**This notebook is intended to be used along side the blog
post**\ `Bring your own model for Amazon SageMaker labeling workflows
with Active
Learning <https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-for-amazon-sagemaker-labeling-workflows-with-active-learning/>`__\ **,
Part 2: Create a Custom Model and Integrate it into an Active Learning
Workflow**.

Permissions
~~~~~~~~~~~

**Please update your role with AmazonEC2ContainerRegistryFullAccess
before proceeding**

Running this notebook requires permissions in addition to the normal
SageMakerFullAccess permissions. This is because it creates new
repositories in Amazon ECR. The easiest way to add these permissions is
simply to add the managed policy
**AmazonEC2ContainerRegistryFullAccess** to the role that you used to
start your notebook instance. There’s no need to restart your notebook
instance when you do this, the new permissions will be available
immediately. To access the role associated with your notebook instance,
select “Notebook instances” from the SageMaker console, select the name
of your instance, and finally select the link under “IAM role ARN” in
the “Permissions and encryption” section.

To Use this Notebook
~~~~~~~~~~~~~~~~~~~~

We use this notebook to tokenize our dataset and create a training
dataset, add a containerized model to ERC, and train the model. The
notebook will produce an image name in ECR which can be used for
training and inference across Amazon SageMaker.

We use a Keras deep learning model for demonstration purposes only. The
methodology for developing and containerizing our model was inspired by
the tutorial `Take an ML from idea to production using Amazon
SageMaker <https://github.com/aws-samples/amazon-sagemaker-keras-text-classification>`__
and is not included in the notebook.

To customize this notebook, you will need to create your own machine
learning model and add it to a Docker container. Use the blog post above
to learn how to do this with Amazon SageMaker.

First we will set up our environment and extract our account number. We
will use the account number to define an image name for the Elastic
Container Repository (ECR).

.. code:: ipython3

    region = sess.boto_session.region_name
    account = sess.boto_session.client('sts').get_caller_identity()['Account']
    image = '{}.dkr.ecr.{}.amazonaws.com/news-classifier'.format(account, region)


Preprocessing and Tokenizing the data
-------------------------------------

First we read the csv news dataset using pandas and clean the data:

-  We make all alphanumeric characters lowercase and replace undesired
   characters.
-  We remove stop words and empty records.

The result is saved into a JSON formatted file.

Next, we use the `Keras Tokenizer
class <https://keras.io/preprocessing/text/>`__ to tokenize our dataset
and upload it to S3.

.. code:: ipython3

    import os, pickle
    from sklearn.feature_extraction import stop_words
    stop_words=stop_words.ENGLISH_STOP_WORDS
    import os,sys,sagemaker, tensorflow as tf, pandas as pd, boto3, numpy as np
    
    train_s3_key = 'sagemaker/news_subset.csv'
    boto3.resource('s3').Bucket(bucket).upload_file('news_subset.csv',train_s3_key)
    
    column_names = ["TITLE", "CATEGORY"]
    tf_train = pd.read_csv('news_subset.csv', names=column_names, header=None, skiprows=[0], delimiter=',')
    tf_train= tf_train[column_names]
    
    tf_train["TITLE"]=tf_train["TITLE"].str.lower().replace('[^\w\s]','')
    tf_train["TITLE"]= tf_train["TITLE"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    tf_train.dropna(inplace=True)
    
    cat=tf_train['CATEGORY'].astype("category").cat.categories
    tf_train['CATEGORY']=tf_train['CATEGORY'].astype("category").cat.codes
    y=tf_train['CATEGORY'].values
    
    
    max_features=5000 #we set maximum number of words to 5000
    maxlen=100 #and maximum sequence length to 100
    embedding_dim = 50 #this is the final dimension of the embedding space.
    tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_features) #tokenizer step
    tok.fit_on_texts(list(tf_train['TITLE'])) #fit to cleaned text
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
    boto3.resource('s3').Bucket(bucket).upload_file('tokenizer.pickle',key+'/tokenizer.pickle')


The next cell will update the value of *tokenizer_bucket* in our
training and prediction scripts within the container.

.. code:: ipython3

    def inplace_string_replace(filename, old_string, new_string):
        with open(filename) as f:
            updated_text=f.read().replace(old_string, new_string)
    
        with open(filename, "w") as f:
            f.write(updated_text)
    
    old_code = "tokenizer_bucket = '<Update tokenizer bucket here>'"
    new_code = "tokenizer_bucket = '{}'".format(bucket)
    inplace_string_replace("./container/news-classifier/train", old_code, new_code)
    inplace_string_replace("./container/news-classifier/predictor.py", old_code, new_code)

We extract the first 1000 entries for training and add them to a
manifest file. Then, we save our training manifest file in S3.

.. code:: ipython3

    column_names = ["TITLE", "CATEGORY"]
    
    tf_train = pd.read_csv('news_subset.csv', names=column_names, header=None, skiprows=[0], delimiter=',')
    tf_train= tf_train[["TITLE","CATEGORY"]]
    tf_train["TITLE"]=tf_train["TITLE"].str.replace('"','').replace('\r', '')
    tf_train['CATEGORY']=tf_train['CATEGORY'].astype("category").cat.codes
    
    val_file = "validation-manifest"
    series=pd.Series(data=tf_train.iloc[:1000].TITLE.values,index=tf_train.iloc[:1000].CATEGORY.values)
    with open(val_file, 'w') as outfile:
        for items in series.iteritems():
            outfile.write("{\"category\":"+str(items[0])+",\"source\":\""+items[1]+"\"}\n")
    boto3.resource('s3').Bucket(bucket).upload_file(val_file,key+ "/" + val_file)
    valdiate_s3_uri =  "s3://{}/{}".format(bucket,key+ "/" + val_file)
    
    train_file = "train-manifest"
    series=pd.Series(data=tf_train.iloc[1000:7000].TITLE.values,index=tf_train.iloc[1000:7000].CATEGORY.values)
    with open(train_file, 'w') as outfile:
        for items in series.iteritems():
            outfile.write("{\"category\":"+str(items[0])+",\"source\":\""+items[1]+"\"}\n")
    
    boto3.resource('s3').Bucket(bucket).upload_file(train_file,key+ "/" + train_file)
    train_s3_uri =  "s3://{}/{}".format(bucket,key+ "/" + train_file)

Adding the Containerized ML Model to ECR
----------------------------------------

The next cell will create a repository in ECR (if it does not exist
already), build our docker image locally, and then `push it to
ECR <https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html>`__.

.. code:: sh

    %%sh
    
    # The name of our algorithm
    algorithm_name=news-classifier
    
    cd container
    
    chmod +x ${algorithm_name}/train
    chmod +x ${algorithm_name}/serve
    
    account=$(aws sts get-caller-identity --query Account --output text)
    
    # Get the region defined in the current configuration (default to us-west-2 if none defined)
    region=$(aws configure get region)
    
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
    
    # On a SageMaker Notebook Instance, the docker daemon may need to be restarted in order
    # to detect your network configuration correctly.  (This is a known issue.)
    if [ -d "/home/ec2-user/SageMaker" ]; then
      sudo service docker restart
    fi
    
    docker build  -t ${algorithm_name} .
    docker tag ${algorithm_name} ${fullname}
    
    docker push ${fullname}

**Confirm** the push to the ecr repository happened successfully before
proceeding to the next section.

Training our Model
------------------

We train our model on the training data that we extracted above and see
the accuracy returned by our algorithm in Amazon SageMaker:

.. code:: ipython3

    from sagemaker.estimator import Estimator
    
    estimator = Estimator(image_name= 'news-classifier',
                          role=role,
                          train_instance_count=1,
                          train_instance_type='local')
    
    estimator.fit({'training': train_s3_uri, 'validation': valdiate_s3_uri})

Print the Image name in ECR
---------------------------

The cell below will print our image’s name in ECR. This image can now be
used for both training and inference across Amazon SageMaker.

.. code:: ipython3

    print(image)

To add this image to an active learning workflow follow the instructions
in *Step 1: Update the container ECR reference* in the blog.

The active learning workflow resources produced by the Cloud Formation
Stack provided in **Bring your own model for SageMaker labeling
workflows with Active Learning** defaults to a ``MultiRecord`` `batch
strategy <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTransformJob.html#sagemaker-CreateTransformJob-request-BatchStrategy>`__.
If your model only support a ``SingleRecord`` batch strategy, change
your batch strategy by following the instructions in *Step 2: Change
batch strategy*.
