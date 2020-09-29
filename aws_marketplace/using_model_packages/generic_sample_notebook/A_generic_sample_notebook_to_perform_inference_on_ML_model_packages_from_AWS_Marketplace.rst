Deploy and perform inference on ML Model packages from AWS Marketplace.
-----------------------------------------------------------------------

There are two simple ways to try/deploy `ML model packages from AWS
Marketplace <https://aws.amazon.com/marketplace/search/results?page=1&filters=FulfillmentOptionType%2CSageMaker::ResourceType&FulfillmentOptionType=SageMaker&SageMaker::ResourceType=ModelPackage>`__,
either using AWS console to deploy an ML model package (see `this
blog <https://aws.amazon.com/blogs/machine-learning/adding-ai-to-your-applications-with-ready-to-use-models-from-aws-marketplace/>`__)
or via code written typically in a Jupyter notebook. Many listings have
a high-quality sample Jupyter notebooks provided by the seller itself,
usually, these sample notebooks are linked to the AWS Marketplace
listing (E.g. `Source
Separation <https://aws.amazon.com/marketplace/pp/prodview-23n4vi2zw67we?qid=1579739476471&sr=0-1&ref_=srh_res_product_title>`__),
If a sample notebook exists, try it out.

If such a sample notebook does not exist and you want to deploy and try
an ML model package via code written in python language, this generic
notebook can guide you on how to deploy and perform inference on an ML
model package from AWS Marketplace.

   **Note**:If you are facing technical issues while trying an ML model
   package from AWS Marketplace and need help, please open a support
   ticket or write to the team on aws-mp-bd-ml@amazon.com for additional
   assistance.

Pre-requisites:
^^^^^^^^^^^^^^^

1. Open this notebook from an Amazon SageMaker Notebook instance.
2. Ensure that Amazon SageMaker notebook instance used has
   IAMExecutionRole with **AmazonSageMakerFullAccess**
3. Your IAM role has these three permisions -
   **aws-marketplace:ViewSubscriptions**,
   **aws-marketplace:Unsubscribe**, **aws-marketplace:Subscribe** and
   you have authority to make AWS Marketplace subscriptions in the AWS
   account used.

..

   **Note**: If you are viewing this notebook from a GitHub repository,
   then to try this notebook successfully, `create an Amazon SageMaker
   Notebook
   Instance <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html>`__
   and then `access Notebook
   Instance <https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-access-ws.html>`__
   you just created. Next, upload this Jupyter notebook to your notebook
   instance.

Additional Resources:
^^^^^^^^^^^^^^^^^^^^^

**Background on Model Packages**: 1. An ML model can be created from a
Model Package, to know how, see `Use a Model Package to Create a
Model <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-mkt-model-pkg-model.html>`__.
2. An ML Model accepts data and generates predictions. 3. To perform
inference, you first need to deploy the ML Model. An ML model typically
supports two types of predictions: 1. `Use Batch
Transform <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html>`__
to asynchronously generate predictions for multiple input data
observations. 2. Send input data to Amazon SageMaker endpoint to
synchronously generate predictions for individual data observations. For
information, see `Deploy a Model on Amazon SageMaker Hosting
Services <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html>`__

**Background on AWS Marketplace Model packages**: If you are new to
Model packages from AWS Marketplace, here are some additional resources.
\* For a high level overview of how AWS Marketplace for Machine Learning
see the `Using AWS Marketplace for machine learning
workloads <https://aws.amazon.com/blogs/awsmarketplace/using-aws-marketplace-for-machine-learning-workloads/>`__
blog post. \* For a high level overview on Model packages from AWS
Marketplace, see `this blog
post <https://aws.amazon.com/blogs/aws/new-machine-learning-algorithms-and-model-packages-now-available-in-aws-marketplace/>`__.
\* For an overview on how to deploy a Model package using AWS Console
and using AWS CLI for performing inference, see the `Adding AI to your
applications with ready-to-use models from AWS
Marketplace <https://aws.amazon.com/blogs/machine-learning/adding-ai-to-your-applications-with-ready-to-use-models-from-aws-marketplace/>`__
blog post. \* For a Jupyter notebook of the sample solution for
**Automating auto insurance claim processing workflow** outlined in
`this re:Mars session <https://www.youtube.com/watch?v=GkKZt0s_ku0>`__,
see
`amazon-sagemaker-examples/aws-marketplace <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/aws_marketplace/using_model_packages/auto_insurance>`__
GitHub repository. \* For a Jupyter notebook of the sample solution for
**Improving workplace safety solution** outlined in `this re:Invent
session <https://www.youtube.com/watch?v=iLOXaWpK6ag>`__, see
`amazon-sagemaker-examples/aws-marketplace <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/aws_marketplace/using_model_packages/improving_industrial_workplace_safety>`__
GitHub repository.

Contents:
^^^^^^^^^

1. `Subscribe to the model package <#Subscribe-to-the-model-package>`__

   1. `Identify compatible
      instance-type <#A.-Identify-compatible-instance-type>`__
   2. `Identify content-type <#B.-Identify-content_type>`__
   3. `Specify model-package-arn <#C.-Specify-model-package-arn>`__

2. `Create an Endpoint and perform real-time
   inference <#2.-Create-an-Endpoint-and-perform-real-time-inference>`__

   1. `Create an Endpoint <#A.-Create-an-Endpoint>`__
   2. `Create input payload <#B.-Create-input-payload>`__
   3. `Perform Real-time inference <#C.-Perform-Real-time-inference>`__
   4. `Visualize output <#D.-Visualize-output>`__
   5. `Delete the endpoint <#E.-Delete-the-endpoint>`__

3. `Perform Batch inference <#3.-Perform-Batch-inference>`__

   1. `Prepare input payload <#A.-Prepare-input-payload>`__
   2. `Run a batch-transform job <#B.-Run-a-batch-transform-job>`__
   3. `Visualize output <#C.-Visualize-output>`__

4. `Delete the model <#4.-Delete-the-model>`__
5. `Unsubscribe to the model
   package <#Unsubscribe-to-the-model-package>`__

Usage instructions
^^^^^^^^^^^^^^^^^^

You can run this notebook one cell at a time (By using Shift+Enter for
running a cell).

.. code:: ipython3

    #Following boilerplate code includes all major libraries that you might need.
    import base64
    import json 
    import uuid
    from sagemaker import ModelPackage
    import sagemaker as sage
    from sagemaker import get_execution_role
    from sagemaker import ModelPackage
    from urllib.parse import urlparse
    import boto3
    from IPython.display import Image
    from PIL import Image as ImageEdit
    import urllib.request
    import numpy as np
    role = get_execution_role()
    
    sagemaker_session = sage.Session()

.. code:: ipython3

    bucket=sagemaker_session.default_bucket()
    bucket

1. Subscribe to the model package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you can deploy the model, your account needs to be subscribed to
it. This section covers instructions for populating necessary parameters
and for subscribing to the Model package, if the subscription does not
already exist.

1. Open the Model Package listing page (E.g. `GluonCV YOLOv3 Object
   Detector <https://aws.amazon.com/marketplace/pp/prodview-5jlvp43tsn3ny?qid=1578429923058&ref_=srh_res_product_title&sr=0-1>`__)
   from AWS Marketplace that you wish to try/use.
2. Read the **product overview** section and **Highlights** section of
   the listing to understand the value proposition of the model package.
3. View **usage information** and then **additional resources**
   sections. These sections will contain following things:

   1. Input content-type
   2. Sample input file (optional)
   3. Sample Jupyter notebook
   4. Output format
   5. Any additional information.

A. Identify compatible instance-type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. On the listing, Under **Pricing Information**, you will see
   **software pricing** for **real-time inference** as well as
   **batch-transform usage** for specific instance-types.

..

   **Note**: Software pricing is in addition to regular SageMaker
   infrastructure charges.

2. In the pricing chart, you will also notice the **vendor recommended
   instance-type** . E.g `GluonCV YOLOv3 Object
   Detector <https://aws.amazon.com/marketplace/pp/prodview-5jlvp43tsn3ny?qid=1578429923058&ref_=srh_res_product_title&sr=0-1>`__
   has recommended real-time inference instance-type as **ml.m4.xlarge**
   and recommended batch transform inference as **ml.m4.xlarge**

3. Specify the recommended instance-types in the following cell and then
   run the cell.

.. code:: ipython3

    real_time_inference_instance_type=''
    batch_transform_inference_instance_type=''
    #real_time_inference_instance_type='ml.m4.xlarge'
    #batch_transform_inference_instance_type='ml.m4.xlarge'

B. Identify content_type
^^^^^^^^^^^^^^^^^^^^^^^^

You need to specify input content-type and payload while performing
inference on the model. In this sub-section you will identify input
content type that is accepted by the model you wish to try.

Sellers has provided content_type information via: 1. a sample
invoke_endpoint api/CLI call in the **usage instructions** section, of
the listing. E.g `GluonCV YOLOv3 Object
Detector <https://aws.amazon.com/marketplace/pp/prodview-5jlvp43tsn3ny?qid=1578429923058&sr=0-1&ref_=srh_res_product_title>`__
has following AWS CLI snippet, with –content-type specified as
**image/jpeg**.
>\ ``Bash aws sagemaker-runtime invoke-endpoint --endpoint-name your_endpoint_name --body fileb://img.jpg --content-type image/jpeg --custom-attributes '{"threshold": 0.2}' --accept json  >(cat) 1>/dev/null``

2. plain-text information in the **usage instructions** section, of the
   listing. E.g. `Lyrics Generator
   (CPU) <https://aws.amazon.com/marketplace/pp/prodview-qqzh5iao6si4c?qid=1578429518061&sr=0-2&ref_=srh_res_product_title>`__
   has following snippet which indicates that **application/json** is
   the content-type.

..

   \```Javascript Input (application/json): Artist name and seed lyrics
   (start of song). Payload: {“instances”: [{“artist”:“”, “seed”: “”}]}

::


   3. Sample notebook, linked under **usage instructions**/**additional information**/**support information** and the sample notebook might use AWS CLI/Boto3 or SDK to perform inference.
       
       1. E.g., [Vehicle Damage Inspection](https://aws.amazon.com/marketplace/pp/prodview-xhj66rbazm6oe?qid=1579723100840&sr=0-1&ref_=srh_res_product_title) has a link to a file under **Additional resources** section that containing **Vehicle-Damage-Inspection.ipynb**, a jupyter notebook that has following snippet with ContentType specified as **image/jpeg**.
   > ```Python
   invocation_api_handle.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='image/jpeg', ... 

::

   2. [A Sample notebook from sagemaker-examples repo](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/aws_marketplace/using_model_packages/auto_insurance/automating_auto_insurance_claim_processing.ipynb) uses Python SDK to perform inference, and the predictor class defined shows that content type is **image/jpeg**  

..

   .. code:: python

      def damage_detection_predict_wrapper(endpoint, session):
      return sage.RealTimePredictor(endpoint, session,content_type='image/jpeg')

Once you have identified the input content type, specify the same in
following cell.

.. code:: ipython3

    content_type=''
    #content_type='image/jpeg'

C. Specify model-package-arn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A model-package-arn is a unique identifier for each ML model package
from AWS Marketplace within a chosen region.

1. On the AWS Marketplace listing, click on **Continue to subscribe**
   button.
2. On the **Subscribe to this software** page (E.g.
   `here <https://aws.amazon.com/marketplace/ai/procurement?productId=d9949c88-fe3b-4a2d-923e-9458fe7e9f2c>`__),
   Review the **End user license agreement**, **support terms**, as well
   as **pricing information**.
3. **“Accept Offer”** button needs to be clicked if your organization
   agrees with EULA, pricing information as well as support terms.
4. Next, **Continue to configuration** button becomes activated and when
   you click on the button, you will see that a **Product Arn** will
   appear. In the **Region** dropdown, Choose the region in which you
   have opened this notebook from, Copy the product ARN and replace it
   in the next cell.

.. code:: ipython3

    model_package_arn=''
    #model_package_arn='arn:aws:sagemaker:us-east-1:865070037744:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2'

Congratulations, you have identified necessary information to be able to
create an endpoint for performing real-time inference.

2. Create an Endpoint and perform real-time inference.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, you will stand up an Amazon SageMaker endpoint. Each
endpoint must have a unique name which you can use for performing
inference.

Specify a short name you wish to use for naming endpoint.

.. code:: ipython3

    model_name=''
    #model_name='gluoncv-object-detector'

A. Create an Endpoint
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    def predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session,content_type)
    
    #create a deployable model from the model package.
    model = ModelPackage(role=role,
                        model_package_arn=model_package_arn,
                        sagemaker_session=sagemaker_session,
                        predictor_cls=predict_wrapper)
    
    #Deploy the model
    predictor = model.deploy(1, real_time_inference_instance_type, endpoint_name=model_name)

Once endpoint has been created, you would be able to perform real-time
inference.

B. Create input payload
^^^^^^^^^^^^^^^^^^^^^^^

**Background**: A Machine Learning model accepts a payload and returns
an inference. E.g. `Deep Vision vehicle
recognition <https://aws.amazon.com/marketplace/pp/prodview-a7wgrolhu54ts?qid=1579728052169&sr=0-1&ref_=srh_res_product_title>`__
accepts an image as a payload and returns an inference containing
make,model, and year of the car.

In this step, you will prepare a payload to perform a prediction. This
step varies from model to model.

Identify a sample Input file you can use:
'''''''''''''''''''''''''''''''''''''''''

1. Sometimes file is available in **additional resources** section of
   the listing. E.g. `Mphasis DeepInsights Document
   Classifier <https://aws.amazon.com/marketplace/pp/prodview-u5jlb2ba6xmaa?qid=1579793398686&sr=0-1&ref_=srh_res_product_title>`__
   has multiple sample files in an archieve.

2. Sometimes file is available in a Github Repo link associated with the
   listing. E.g. `Source
   Separation <https://aws.amazon.com/marketplace/pp/prodview-23n4vi2zw67we?qid=1579739476471&sr=0-1&ref_=srh_res_product_title>`__
   has a sample file in the GitHUB repo. In which case, please copy the
   link to the raw data file.

3. Sometimes a sample file is not available, however, clear instructions
   on how to prepare the payload are available. E.g. `Face Anonymizer
   (CPU) <https://aws.amazon.com/marketplace/pp/prodview-3olpixsfcqfq6?qid=1560287886810&sr=0-3&ref_=srh_res_product_title>`__),
   then you would need to manually identify an input file you can use. I
   identified that I can use an image shown on `this
   blog <https://aws-preview.aka.amazon.com/blogs/machine-learning/adding-ai-to-your-applications-with-ready-to-use-models-from-aws-marketplace/>`__,
   and then manually prepare a payload for performing inference

4. For models for which there is no sample file (E.g. `Demisto Phishing
   Email
   Classifier <https://aws.amazon.com/marketplace/pp/prodview-k5354ho27eyps>`__)
   but it accepts a simple input, jump to `Step
   B.2 <#Step-B.2-Manually-prepare-data-(applicable-only-if-your-payload-is-not-ready-yet)>`__

Specify the URL of the sample file you identified in the following cell
to download the file for creating payload.

.. code:: ipython3

    url=''
    #url='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg'
    #url='https://d1.awsstatic.com/webteam/architecture-icons/AWS-Architecture-Icons-Deck_For-Light-BG_20191031.pptx.6fcecd0cf65442a1ada0ce1674bc8bfc8de0cb1d.zip'

Next, specify a file_name that you would like to save the file to.

.. code:: ipython3

    file_name=''
    #file_name='input.json'
    #file_name='input.jpg'
    #file_name='file.zip'

.. code:: ipython3

    #Download the file
    urllib.request.urlretrieve(url,file_name)

View the file you just downloaded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on the type of file used, uncomment, modify, and run appropriate
code snippets.

ZIP/Tar file
            

If your input file was inside a zip file, uncomment appropriate line
from following two lines.

.. code:: ipython3

    #!unzip $file_name
    #!tar -xvf $file_name

.. code:: ipython3

    #Update the file_name variable with an appropriate file-path from the folder created by unzipping the archieve
    #file_name=''
    #file_name='images/AWS-Architecture-Icons-Deck_For-Light-BG_20191031.pptx'

Image File
          

.. code:: ipython3

    #Uncomment and run the following line to view the image
    #Image(url= file_name, width=400, height=800)

Text/Json/CSV File
                  

If your input file is a text/json/csv file, view the file by
un-commenting following line. If your file contains multiple payloads,
consider keeping just one.

.. code:: ipython3

    #!head $file_name

Video File
''''''''''

.. code:: ipython3

    #View and play the video by uncommenting following two lines
    
    #from IPython.display import HTML
    #HTML('<iframe width="560" height="315" src="'+file_name+'?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')

Audio File
''''''''''

.. code:: ipython3

    #Uncomment following two lines to view and play the audio
    
    #import IPython.display as ipd
    #ipd.Audio(file_name)

If your model’s input **content-type** is one of the following and
file_name variable is pointing to a file that can directly be sent to ML
model, congratulations, you have prepared the payload, you can jump to
Step `C. Perform Real-time
inference <#C.-Perform-Real-time-inference>`__: \* **wav/mp3** \*
**application/pdf** \* **image/png** \* **image/jpeg**: \*
**text/plain** \* **text/csv** \* **application/json** (Only if
file_name variable is pointing to a JSON file that can directly be sent
to ML model)

If your content-type is any other, your model might need additional
pre-processing, proceed to `Step
B.1 <#Step-B.1-Pre-process-the-data-(Optional-for-some-models)>`__

Step B.1 Pre-process the data (Optional for some models)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some models require preprocessing, others dont. If model you want to
try/use requires additional pre-processing, Please refer to sample
notebook or usage instructions for pre-processing required. This section
contains some re-usable code that you might need to tweak as per your
requirement. Uncomment, tweak and use following as required. Ensure that
final payload is written to a variable with name ‘payload’.

Some models require Base64 encoded data
'''''''''''''''''''''''''''''''''''''''

   `Background Noise Classifier
   (CPU) <https://aws.amazon.com/marketplace/pp/prodview-vpd6qdjm4d7u4?qid=1579792115621&sr=0-2&ref_=srh_res_product_title>`__
   requires payload to be in following format

.. code:: javascript

       Payload: {"instances": [{"audio": {"b64": "BASE_64_ENCODED_WAV_FILE_CONTENTS"}}]}

..

   `Neural Style
   Transfer <https://aws.amazon.com/marketplace/pp/prodview-g5i35lg4qmplu>`__
   requires payload to be in following format. You would need to tweak
   code appropriately to convert two images into base64 format for this
   model.

.. code:: javascript

       {
           "content": "base64 characters of your content image",
           "style": "base64 characters of your style image",
           "iterations": 2
       }

.. code:: ipython3

    #Here is a sample code that does Base64 encoding
    
    #file_read = open(file_name, 'rb').read() 
    #base64_encoded_value = base64.b64encode(file_read).decode('utf-8')
    
    #payload="{\"style\":\""+str(style_image_base64_encoded_value)+"\", \"iterations\": 2,\"content\":\""+str(base64_encoded_value)+"\"}"

Some models require images in serialized format
'''''''''''''''''''''''''''''''''''''''''''''''

   E.g. `Mphasis DeepInsights Damage
   Prediction <https://aws.amazon.com/marketplace/pp/prodview-2f5br37zmuk2y?qid=1576781776298&sr=0-1&ref_=srh_res_product_title>`__
   requires the image to be re-sized to (300 x 300) and then JSON
   serialised before it can be fed to the model. To make it easy to do
   so, they also have provided snippet identical to following one in the
   sample jupyter notebook.

.. code:: ipython3

    #from PIL import Image
    
    #image = Image.open(file_name).convert(mode = 'RGB')
    #resized_image = image.resize((300,300))
    
    #image_array = np.array(resized_image).tolist()
    #payload = json.dumps({'instances': [{'input_image': image_array}]})

Next, jump to `Step
B.3 <#Step-B.3-Write-payload-you-created-to-a-file>`__

Step B.2 Manually prepare data (applicable only if your payload is not ready yet)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

If sample notebook is not available but input format is simple, write
code required for creating the input file. E.g. `Demisto Phishing Email
Classifier <https://aws.amazon.com/marketplace/pp/prodview-k5354ho27eyps>`__
does not have a sample file but sample notebook has some code that can
be used for prepared payload.

.. code:: javascript

   email1 = "<Email content shortened for brevity>"
   email2 = "<Email content shortened for brevity>"
   emails = [email1, email2]
   json.dumps(emails)

Prepare appropriate payload and store the same in a variable called
‘payload’.

.. code:: ipython3

    #Write your code here.

Jump to `Step B.3 <#Step-B.3-Write-payload-you-created-to-a-file>`__ to
write your payload to a file.

Step B.3 Write payload you created to a file
''''''''''''''''''''''''''''''''''''''''''''

Assuming that you have populated payload json/csv in a variable called
‘payload’, here is a sample generic code that writes the payload to a
file you can un-comment and reuse.

.. code:: ipython3

    #file_name='output_file'
    
    #file = open(file_name, "w") #Change w to wb if you intend to write bytes isntead of text.
    #file.write(payload)
    #file.close()

Once your payload is ready and is written to a file referenced by the
file_name variable, you are ready to perform an inference. Proceed to
Step `C. Perform Real-time
inference <#C.-Perform-Real-time-inference>`__.

C. Perform Real-time inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify name and extension of the file you would like to store the
inference output to. The output type varies from model to model and this
information is usually available in the **Usage instructions**/**sample
notebook** associated with the listing.

For Example: \* `Neural Style
Transfer <https://aws.amazon.com/marketplace/pp/prodview-g5i35lg4qmplu>`__
model’s output type is image, so specify **.png** as file extension. \*
`Source
Separation <https://aws.amazon.com/marketplace/pp/prodview-23n4vi2zw67we?qid=1579807024600&sr=0-1&ref_=srh_res_product_title>`__
model’s output type is a zip file, so specify **.zip** as file
extension. \* `Mphasis DeepInsights Address
Extraction <https://aws.amazon.com/marketplace/pp/prodview-z4wgslad4b27g?qid=1579802907920&sr=0-2&ref_=srh_res_product_title>`__
model’s output type is text/plain, so specify **.txt** as file
extension. \* Sample notebook provided by seller usually makes it
evident what the output type is. If one doesnt exist, and instructions
are unclear, try a few options, start with text - Many ML models return
response in a simple textual format.

.. code:: ipython3

    real_time_inference_output_file_name=''
    #real_time_inference_output_file_name='output.json'
    #real_time_inference_output_file_name='output.zip'
    #real_time_inference_output_file_name='output.txt'
    #real_time_inference_output_file_name='output.png'

The following AWS CLI command sends the **payload** and the
**content-type** to the model hosted on the endpoint. > **Note on Custom
Attributes**: Some models accept additional attributes such as `GluonCV
YOLOv3 Object
Detector <https://aws.amazon.com/marketplace/pp/prodview-5jlvp43tsn3ny?qid=1578429923058&ref_=srh_res_product_title&sr=0-1>`__
accepts a custom attribute called threshold as specified in following
sample code snippet. >
``Bash aws sagemaker-runtime invoke-endpoint --endpoint-name your_endpoint_name --body fileb://img.jpg --content-type image/jpeg --custom-attributes '{"threshold": 0.2}' --accept json  >(cat) 1>/dev/null``
Please modify the following AWS-CLI command appropriately if the model
you wish to perform inference on requires any custom attribute, if not,
execute following command to perform inference.

Once inference has been performed, the output gets written to the output
file.

.. code:: ipython3

    !aws sagemaker-runtime invoke-endpoint \
        --endpoint-name $model_name \
        --body fileb://$file_name \
        --content-type $content_type \
        --region $sagemaker_session.boto_region_name \
        $real_time_inference_output_file_name

If the above invocation shows a snippet such as following, it means the
command executed successfully. Otherwise, check whether input payload is
in correct format.

.. code:: javascript

   {
       "ContentType": "<content_type>; charset=utf-8",
       "InvokedProductionVariant": "<Variant>"
   }

View the output available in file referenced by
**real_time_inference_output_file_name** variable.

D. Visualize output
^^^^^^^^^^^^^^^^^^^

If the output is in **text**/**CSV**/**JSON** format, view the output
file by uncommenting and running following command. Otherwise use an
appropriate command (Please see reference commands from step `View the
file you just downloaded <#View-the-file-you-just-downloaded>`__) for
viewing the output OR open the output file directly from Jupyter
console.

.. code:: ipython3

    #f=open(real_time_inference_output_file_name, "r")
    #data=f.read()
    #print(data)
    #Sometimes output is a json, load it into a variable with json.loads(data) and then print the variable to see formatted output.

E. Delete the endpoint
^^^^^^^^^^^^^^^^^^^^^^

Now that you have successfully performed a real-time inference, you do
not need the endpoint any more. you can terminate the same to avoid
being charged.

.. code:: ipython3

    predictor=sage.RealTimePredictor(model_name, sagemaker_session,content_type)
    predictor.delete_endpoint(delete_endpoint_config=True)

3. Perform Batch inference
~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a batch transform job, we will use the same payload we used for
performing real-time inference. file_name variable points to the
payload.

   **Note**: If you followed instructions closely, your input file
   contains a single payload. However, batch-transform can be used to
   perform a batch inference on multiple records at a time. To know
   more, see documentation.

.. code:: ipython3

    #upload the file to S3
    transform_input = sagemaker_session.upload_data(file_name, key_prefix=model_name) 
    print("Transform input uploaded to " + transform_input)

.. code:: ipython3

    #Run a batch-transform job
    transformer = model.transformer(1, batch_transform_inference_instance_type)
    transformer.transform(transform_input, content_type=content_type)
    transformer.wait()

.. code:: ipython3

    #output is available on following path
    transformer.output_path

C. Visualize output
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from urllib.parse import urlparse
    
    parsed_url = urlparse(transformer.output_path)
    bucket_name = parsed_url.netloc
    file_key = '{}/{}.out'.format(parsed_url.path[1:], file_name.split("/")[-1])
    print(file_key)
    s3_client = sagemaker_session.boto_session.client('s3')
    
    response = s3_client.get_object(Bucket = sagemaker_session.default_bucket(), Key = file_key)

If the output is in **text**/**CSV**/**JSON** format, view the output
file by uncommenting and running following command. Otherwise go to S3,
download the file and open it using appropriate editor.

.. code:: ipython3

    response_bytes = response['Body'].read().decode('utf-8')
    print(response_bytes)

4. Delete the model
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model.delete_model()

**Note** - You need to write appropriate code here to clean-up any files
you may have uploaded/created while trying out this notebook.

5. Cleanup
~~~~~~~~~~

Finally, if the AWS Marketplace subscription was created just for the
experiment and you would like to unsubscribe to the product, here are
the steps that can be followed. Before you cancel the subscription,
ensure that you do not have any `deployable
model <https://console.aws.amazon.com/sagemaker/home#/models>`__ created
from the model package or using the algorithm. Note - You can find this
information by looking at the container name associated with the model.

**Steps to un-subscribe to product from AWS Marketplace**: 1. Navigate
to **Machine Learning** tab on `Your Software subscriptions
page <https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=lbr_tab_ml>`__
2. Locate the listing that you would need to cancel subscription for,
and then **Cancel Subscription** can be clicked to cancel the
subscription.
