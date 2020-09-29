Goal: Automate Auto Insurance Claim Processing Using Pre-trained Models
-----------------------------------------------------------------------

Auto insurance claim process requires extracting metadata from images
and performing validations to ensure that the claim is not fraudulent.
This sample notebook shows how third party pre-trained machine learning
models can be used to extract such metadata from images.

This notebook uses `Vehicle Damage
Inspection <https://aws.amazon.com/marketplace/pp/Persistent-Systems-Vehicle-Damage-Inspection/prodview-xhj66rbazm6oe>`__
model to identify the type of damage and `Deep Vision vehicle
recognition <https://aws.amazon.com/marketplace/pp/prodview-a7wgrolhu54ts?qid=1558356141251&sr=0-4&ref_=srh_res_product_title>`__
to identify the make, model, year, and bounding box of the car. This
notebook also shows how to use the bounding box to extract license
information from the using `Amazon
Rekognition <https://aws.amazon.com/rekognition/>`__.

Pre-requisites:
~~~~~~~~~~~~~~~

This sample notebook requires subscription to following pre-trained
machine learning model packages from AWS Marketplace:

1. `Vehicle Damage
   Inspection <https://aws.amazon.com/marketplace/pp/Persistent-Systems-Vehicle-Damage-Inspection/prodview-xhj66rbazm6oe>`__
2. `Deep Vision vehicle
   recognition <https://aws.amazon.com/marketplace/pp/prodview-a7wgrolhu54ts?qid=1558356141251&sr=0-4&ref_=srh_res_product_title>`__

If your AWS account has not been subscribed to these listings, here is
the process you can follow for each of the above mentioned listings: 1.
Open the listing from AWS Marketplace 2. Read the **Highlights** section
and then **product overview** section of the listing. 3. View **usage
information** and then **additional resources**. 4. Note the supported
instance types. 5. Next, click on **Continue to subscribe**. 6. Review
**End user license agreement**, **support terms**, as well as **pricing
information**. 7. **“Accept Offer”** button needs to be clicked if your
organization agrees with EULA, pricing information as well as support
terms.

**Notes**: 1. If **Continue to configuration** button is active, it
means your account already has a subscription to this listing. 2. Once
you click on **Continue to configuration** button and then choose
region, you will see that a **Product Arn** will appear. This is the
model package ARN that you need to specify while creating a deployable
model. However, for this notebook, the algorithm ARN has been specified
in **src/model_package_arns.py** file and you do not need to specify the
same explicitly.

Set up environment and view a sample image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we will import necessary libraries and define variables
such as an S3 bucket, an IAM role, and SageMaker session to be used.

.. code:: ipython3

    import base64
    import json 
    import uuid
    from sagemaker import ModelPackage
    from src.model_package_arns import ModelPackageArnProvider
    import sagemaker as sage
    from sagemaker import get_execution_role
    from sagemaker import ModelPackage
    from urllib.parse import urlparse
    import boto3
    from IPython.display import Image
    from PIL import Image as ImageEdit
    
    role = get_execution_role()
    
    sagemaker_session = sage.Session()
    bucket=sagemaker_session.default_bucket()

For your convenience sample images which depict damage (manually added
using a photo editor tool), have been provided with this notebook. Next,
view the image to be processed.

.. code:: ipython3

    vehicle_image_path='img/car_damage.jpg'
    vehicle_image_damage_closeup_path='img/closeup.png'
    
    #View the image
    Image(url= vehicle_image_path, width=400, height=800)

.. code:: ipython3

    #View the close-up image of the damaged part
    Image(url= vehicle_image_damage_closeup_path, width=400, height=800)

Step 1: Deploy Vehicle Damage Inspection model
----------------------------------------------

In this step, we will deploy the `Vehicle Damage
Inspection <https://aws.amazon.com/marketplace/pp/Persistent-Systems-Vehicle-Damage-Inspection/prodview-xhj66rbazm6oe>`__
model package. The model package can be used to detect following types
of car damages: 1. Normal image 2. Broken headlight 3. Broken windshield
4. Full front damage.

Step 1.1: Deploy the model for performing real-time inference.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Get the model_package_arn
    damage_detection_modelpackage_arn = ModelPackageArnProvider.get_vehicle_damage_detection_model_package_arn(sagemaker_session.boto_region_name)
    
    #Define predictor wrapper class
    def damage_detection_predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session,content_type='image/jpeg')
    
    #create a deployable model for damage inspection model package.
    damage_detection_model = ModelPackage(role=role,
                                          model_package_arn=damage_detection_modelpackage_arn,
                                          sagemaker_session=sagemaker_session,
                                          predictor_cls=damage_detection_predict_wrapper)
    
    #Deploy the model
    predictor_damage_detection = damage_detection_model.deploy(1, 'ml.m4.xlarge', endpoint_name='vehicle-damage-detection-endpoint')


Step 1.2: Perform a prediction on Amazon Sagemaker Endpoint created.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, we will prepare a payload and perform a prediction.

.. code:: ipython3

    # Open the file and read the image into a bytearray.
    with open(vehicle_image_damage_closeup_path, "rb") as image:
      b = bytearray(image.read())
    
    #Perform a prediction
    damage_detection_result = predictor_damage_detection.predict(b).decode('utf-8')
    
    #View the prediction
    print(damage_detection_result)

Step 2: Deploy the Vehicle recognition model.
---------------------------------------------

In this step, we will deploy the `Deep Vision vehicle
recognition <https://aws.amazon.com/marketplace/pp/prodview-a7wgrolhu54ts?qid=1558356141251&sr=0-4&ref_=srh_res_product_title>`__
model package.

We will use it to detect year, make, model, and angle (such as front
right, front left, front center, rear right, rear left, rear center,
side left, side right) of the car in picture.

Step 2.1: Deploy the model for performing real-time inference.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Get the model_package_arn
    vehicle_recognition_modelpackage_arn = ModelPackageArnProvider.get_vehicle_recognition_model_package_arn(sagemaker_session.boto_region_name)
    
    #Define predictor wrapper class
    def vehicle_recognition_predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session,content_type='application/json')
    
    #create a deployable model.
    vehicle_recognition_model = ModelPackage(role=role,
                                             model_package_arn=vehicle_recognition_modelpackage_arn,
                                             sagemaker_session=sagemaker_session,
                                             predictor_cls=vehicle_recognition_predict_wrapper)
    
    #Deploy the model
    predictor_vehicle_recognition = vehicle_recognition_model.deploy(1, 'ml.p2.xlarge', endpoint_name='vehicle-recognition-endpoint')


Step 2.2: Perform real-time inference on the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Read the image and prepare the payload
    image = open(vehicle_image_path, 'rb') 
    image_64_encode = base64.b64encode(image.read()).decode('utf-8')
    
    #Prepare payload for prediction
    payload="{\"source\": \""+str(image_64_encode)+"\"}"
    
    
    #Perform a prediction
    result = predictor_vehicle_recognition.predict(payload).decode('utf-8')
    vehicle_mmy_result= json.loads(result)
    #View the prediction
    print(json.dumps(vehicle_mmy_result, indent=2))

Step 2.3: Store the precise car image for further processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Extract the bounding box of the first result.
    left_top_x=int(vehicle_mmy_result['result'][0]['bbox']['left'])
    left_top_y=int(vehicle_mmy_result['result'][0]['bbox']['top'])
    
    
    right_bottom_x=int(vehicle_mmy_result['result'][0]['bbox']['right'])
    right_bottom_y=int(vehicle_mmy_result['result'][0]['bbox']['bottom'])

.. code:: ipython3

    #Let us crop the image based on bounding box and use the same for extracting license information.
    vehicle_image = ImageEdit.open(vehicle_image_path)
    
    vehicle_image_bounding_box_path="vehicle_image_bounding_box_2.jpg"
    
    vehicle_image_bounding_box = vehicle_image.crop((left_top_x,left_top_y,right_bottom_x,right_bottom_y))
    vehicle_image_bounding_box.save(vehicle_image_bounding_box_path)

Step 3. Extract labels from the picture (optional)
--------------------------------------------------

Let us use the car image extracted from the original image for
extracting license information using `Amazon
Rekognition <https://aws.amazon.com/rekognition/>`__.

**Note**:

This step requires the IAM role associated with this notebook to have
**rekognition:DetectText** IAM permission.

.. code:: ipython3

    client=boto3.client('rekognition')
    
    recognized_word=''
    with open(vehicle_image_bounding_box_path, 'rb') as image:
        response = client.detect_text(Image={'Bytes': image.read()})
               
    for label in response['TextDetections']:
        if(label['Confidence']>99 and label['Type']== 'WORD'):
            print(label['DetectedText'])
            recognized_word=label['DetectedText']

Step 4: View all outputs
------------------------

View the original image.

.. code:: ipython3

    Image(url= vehicle_image_path, width=400, height=800)

Look at the metadata the metadata we have extracted so far.

.. code:: ipython3

    print("Vehicle Make found: "+ vehicle_mmy_result['result'][0]['mmy']['make'])
    print("Vehicle Model found: "+ vehicle_mmy_result['result'][0]['mmy']['model'])
    print("Vehicle Year found: "+ vehicle_mmy_result['result'][0]['mmy']['year'])
    print("Damage detection probabilities: "+ json.loads(damage_detection_result)['Results'])
    print("License detected: "+recognized_word)

Note how we were able extract information such as car’s make, model,
year, and damage-type using pre-trained machine learning models.

5. Cleanup
~~~~~~~~~~

.. code:: ipython3

    predictor_damage_detection.delete_endpoint()
    predictor_damage_detection.delete_model()

.. code:: ipython3

    predictor_vehicle_recognition.delete_endpoint()
    predictor_vehicle_recognition.delete_model()

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
