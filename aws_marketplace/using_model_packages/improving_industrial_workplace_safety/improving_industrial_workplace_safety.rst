Demonstrating Industrial Workplace Safety using Pre-trained Machine Learning Models
-----------------------------------------------------------------------------------

Introduction
~~~~~~~~~~~~

This sample notebook shows how to use pre-trained model packages from
`AWS
Marketplace <https://aws.amazon.com/marketplace/search/results?page=1&filters=FulfillmentOptionType&FulfillmentOptionType=SageMaker&ref_=mlmp_gitdemo_indust>`__
to detect industrial workspace safety related object labels, such as
hard-hat, personal protective equipment, construction machinery, and
construction worker in an image. The notebook also shows an approach to
perform inference on a video by taking snapshots from the video file to
generate an activity/status log. At the end of this you will become
familiar on steps to integrate inferences from pre-trained models into
your application. This notebook is intended for demonstration, we highly
recommend you to evaluate the accuracy of machine learning models to see
if they meet your expectations.

Pre-requisites:
~~~~~~~~~~~~~~~

This sample notebook requires you to subscribe to pre-trained machine
learning model packages. Follow the following steps to subscribe to the
listings:

1.  Open the following model package product detail pages, in separate
    tabs, in your web browser.

2.  `Construction Worker
    Detection <https://aws.amazon.com/marketplace/pp/prodview-6utmzaproaqhs?qid=1563547984309&sr=0-5&ref_=mlmp_gitdemo_indust>`__
    to identify construction workers in an image.

3.  `Hard Hat Detector for Worker
    Safety <https://aws.amazon.com/marketplace/pp/prodview-jd5tj2egpxxum?qid=1563547984309&sr=0-2&ref_=mlmp_gitdemo_indust>`__
    model to infer if construction workers are wearing hard hats.

4.  `Personal Protective
    Equipments <https://aws.amazon.com/marketplace/pp/prodview-2inbkii6o24k4?qid=1563547984309&sr=0-6&ref_=mlmp_gitdemo_indust>`__
    to infer if a person is wearing a high visibility safety vest.

5.  `Construction Machines
    Detector <https://aws.amazon.com/marketplace/pp/prodview-fuukizaiq5o7c?qid=1563549078039&sr=0-1&ref_=mlmp_gitdemo_indust>`__
    to identify construction machines in an image.

6.  For each of the model packages, follow these steps:
7.  Review the information available on the product details page
    including **Support Terms** .
8.  Click on **“Continue to Subscribe”**. You will now see the
    **“Subscribe to this software”** page.
9.  Review **End User License Agreement** and **Pricing Terms**.
10. **“Accept Offer”** button needs to be clicked if your organization
    agrees with EULA, pricing information and support terms.

Notes: 1. Once you click on **Continue to configuration** button and
then choose a region, you will see a **Product Arn** displayed. This is
the model package ARN that you need to specify while creating a
deployable model using Boto3. However, for this notebook, the model ARNs
have been specified in **src/model_package_arns.py** file and you need
not specify them explicitly. The configuration page also provides a
**“View in SageMaker”** button to navigate to Amazon SageMaker to deploy
via Amazon SageMaker Console. 1. Products with **Free Trials**, do not
incur hourly software charges during free trial period, but AWS
infrastructure charges still apply. Free Trials will automatically
convert to a paid hourly subscription upon expiration. We have included
steps below to cancel subscription at the end of this exercise.

Step 1: Set up environment and view sample images
-------------------------------------------------

Step 1.1: Set up environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we will first import necessary libraries and define
variables such as an S3 bucket, an IAM role, and an Amazon SageMaker
session.

.. code:: ipython3

    #Import necessary libraries and declare variables
    import json 
    from sagemaker import ModelPackage
    from src.model_package_arns import ModelPackageArnProvider
    import sagemaker as sage
    from sagemaker import get_execution_role
    from sagemaker import ModelPackage
    import boto3
    from IPython.display import Image
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    
    role = get_execution_role()
    
    sagemaker_session = sage.Session()
    bucket=sagemaker_session.default_bucket()
    region=sagemaker_session.boto_region_name

Next, we will create utility functions to: 1. Return the predictor
wrapper for an image payload. 2. Draw a bounding box on an image based
on coordinates. 3. Display an image.

.. code:: ipython3

    #Define a generic image predictor wrapper which accepts endpoint & session object, and returns a predictor wrapper
    def image_predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session,content_type='image/jpeg')

.. code:: ipython3

    #This function accepts an image, bounding box co-ordinates, label, label probability,  
    # and returns the image that has the bounding box along with the label and its probability.
    def draw_bounding_box(img,x1,y1,x2,y2,class_name,probability):
        #truncate probability to two decimal places
        img = cv2.rectangle(img,(x1,y1) , (x2,y2), (0,215,255), 2)
        if probability is not None:
            img = cv2.putText(img, '{} {}'.format(
                class_name, float(str(probability)[:4])),
                (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,255,0), 2)
        
        else:
            img = cv2.putText(img, '{}'.format(class_name),(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,255,0), 2)
        return img

.. code:: ipython3

    #This function accepts image along with a title and displays the same.
    def show_image(img, title):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        figure = plt.figure(figsize = (12,18)) 
        axis = figure.add_subplot(111)
        axis.imshow(rgb_img,interpolation='none')
        axis.set_title(title)
        plt.show()

Step 1.2: View sample images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will now view the sample images used to perform an inference.

Step 1.2.1: View construction site image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the next cell to view an image of a construction site with workers
and a truck. The workers are wearing personal protective equipment -
hard hat, and safety vest.

.. code:: ipython3

    construction_image={'path':'img/construction-2578410_640.jpg'}
    
    with open(construction_image['path'], "rb") as image:
      construction_image['byte_array'] = bytearray(image.read())
    
    Image(url= construction_image['path'], width=600)

Courtesy - https://pixabay.com/photos/construction-worker-safety-2578410

Step 1.2.2: View an image with a worker and a person at a workplace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following image shows two people, a worker wearing a high-visibility
vest and a person.

.. code:: ipython3

    workers_image={'path':'img/two-employees.jpg'}
    
    with open(workers_image['path'], "rb") as image:
      workers_image['byte_array'] = bytearray(image.read())
    
    Image(url= workers_image['path'], width=600)

Courtesy -
https://www.pexels.com/photo/two-men-wearing-white-hard-hat-901941

Step 1.2.3: View an image with an excavator and a truck at work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following image shows a truck and an excavator.

.. code:: ipython3

    machines_image={'path':'img/earth-2579434_1280.jpg'}
    
    with open(machines_image['path'], "rb") as image:
      machines_image['byte_array'] = bytearray(image.read())
    
    Image(url= machines_image['path'], width=600)

Courtesy -
https://pixabay.com/photos/earth-390f-hydraulic-excavators-2579434/

We will deploy pre-trained models to generate inferences using sample
images.

Step 2: Deploy construction worker detection model
--------------------------------------------------

In this step, you will deploy the `Construction Worker
Detection <https://aws.amazon.com/marketplace/pp/prodview-6utmzaproaqhs?qid=1563547984309&sr=0-5&ref_=mlmp_gitdemo_indust>`__
model package and perform an inference using sample images.

Step 2.1: Deploy the model for performing real-time inference.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Get the model_package_arn.
    construction_worker_detection_modelpackage_arn = ModelPackageArnProvider.get_construction_worker_model_package_arn(region)
    
    #create a deployable model for damage inspection model package.
    construction_worker_detection_model = ModelPackage(role=role,
                                          model_package_arn=construction_worker_detection_modelpackage_arn,
                                          sagemaker_session=sagemaker_session,
                                          predictor_cls=image_predict_wrapper)
    
    #Deploy the model.
    predictor_construction_worker_detection = construction_worker_detection_model.deploy(1, 'ml.c5.xlarge', endpoint_name='construction-worker-detection-endpoint')


While the model is deploying, review the **Usage Information** and
**Additional Resources** section from the `model package detail
page <https://aws.amazon.com/marketplace/pp/prodview-6utmzaproaqhs?qid=1563547984309&sr=0-5&ref_=mlmp_gitdemo_indust>`__
to understand the I/O interface of the model.

Step 2.2: Perform a prediction (Test 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, we will perform a prediction using the construction-site
image.

.. code:: ipython3

    #Perform a prediction.
    construction_worker_detection_result_1 = json.loads(predictor_construction_worker_detection.predict(construction_image['byte_array']).decode('utf-8'))
    #Un-comment the following line to view the result returned by the model.
    #print(json.dumps(construction_worker_detection_result,indent=2))
    
    #Read original image.
    image=cv2.imread(construction_image['path'])
    
    #Plot the inference on the image
    for output in construction_worker_detection_result_1['output']:
        x1=int(output['bbox'][0])
        y1=int(output['bbox'][1])
        x2=int(output['bbox'][2])
        y2=int(output['bbox'][3])
        image=draw_bounding_box(image,x1,y1,x2,y2,output['class'],None)
    
    show_image(image,'Worker detection Test 1')

You can see that the model recognized all the three workers found in the
image.

Step 2.3: Perform a prediction (Test 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us perform inference using the worker/person image and see how the
model can identify a worker and a person.

.. code:: ipython3

    #Perform a prediction
    construction_worker_detection_result_2 = json.loads(predictor_construction_worker_detection.predict(workers_image['byte_array']).decode('utf-8'))
    
    #Un-comment the following line to view the result returned by the model.
    #print(json.dumps(construction_worker_detection_result,indent=2))
    
    #Read original image.
    
    image=cv2.imread(workers_image['path'])
    
    #Plot the inference on the image
    for output in construction_worker_detection_result_2['output']:
        x1=int(output['bbox'][0])
        y1=int(output['bbox'][1])
        x2=int(output['bbox'][2])
        y2=int(output['bbox'][3])
        image=draw_bounding_box(image,x1,y1,x2,y2,output['class'],None)
    
    show_image(image,'Worker detection Test 2')

You can see that the model can differentiate between a construction
worker (on the left) and a non-construction worker (person on the
right).

AWS Marketplace also contains another model you may want to try for
`construction worker
detection <https://aws.amazon.com/marketplace/pp/prodview-labdyzgb3z6fe?qid=1563562334851&sr=0-2&ref_=mlmp_gitdemo_indust>`__.

Step 3: Deploy the hard-hat detection model.
--------------------------------------------

In this step, we will deploy the `Hard Hat Detector for Worker
Safety <https://aws.amazon.com/marketplace/pp/prodview-jd5tj2egpxxum?qid=1563547984309&sr=0-2&ref_=mlmp_gitdemo_indust>`__
model to identify whether people in the image are wearing `hard
hats <https://en.wikipedia.org/wiki/Hard_hat>`__.

Step 3.1: Deploy the model for performing real-time inference.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Get the model_package_arn
    hard_hat_detection_modelpackage_arn = ModelPackageArnProvider.get_hard_hat_detection_model_package_arn(region)
    
    #create a deployable model.
    hard_hat_detection_model = ModelPackage(role=role,
                                             model_package_arn=hard_hat_detection_modelpackage_arn,
                                             sagemaker_session=sagemaker_session,
                                             predictor_cls=image_predict_wrapper)
    
    #Deploy the model
    predictor_hard_hat_detection = hard_hat_detection_model.deploy(1, 'ml.p2.xlarge', endpoint_name='hardhat-detection-endpoint')


Step 3.2: Perform real-time inference on the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Perform a prediction
    hard_hat_detection_result = json.loads(predictor_hard_hat_detection.predict(construction_image['byte_array']).decode('utf-8'))
    #Un-comment the following line to view the result returned by the model.
    #print(json.dumps(hard_hat_detection_result,indent=2))
    
    #Read original image.
    image=cv2.imread(construction_image['path'])
    
    #Plot the inference on the image
    width=image.shape[1]
    height=image.shape[0]
    
    for i in range(len(hard_hat_detection_result['boxes'])):
        output = hard_hat_detection_result['boxes'][i]
        x1=int(round(output[0]*width, 2))
        y1=int(round(output[1]*height, 2))
        x2=int(round(output[2]*width, 2))
        y2=int(round(output[3]*height, 2))
        image=draw_bounding_box(image,x1,y1,x2,y2,'hard-hat',hard_hat_detection_result['scores'][i])
    #Display result
    show_image(image,'hard-hat detection')

Note, the pre-trained model could identify all three hard-hats found in
the picture with high probabilities.

Step 4. Deploy the Personal Protective Equipment (PPE) detection model
----------------------------------------------------------------------

Next, we will deploy `Personal Protective
Equipment <https://aws.amazon.com/marketplace/pp/prodview-2inbkii6o24k4?qid=1563547984309&sr=0-6&ref_=mlmp_gitdemo_indust>`__
machine learning model to identify whether the person in the image is
wearing PPE such as a high visibility vest.

Step 4.1: Deploy the model for performing real-time inference.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Get the model_package_arn.
    ppe_detection_modelpackage_arn = ModelPackageArnProvider.get_ppe_detection_model_package_arn(region)
    
    #create a deployable model.
    ppe_detection_model = ModelPackage(role=role,
                                             model_package_arn=ppe_detection_modelpackage_arn,
                                             sagemaker_session=sagemaker_session,
                                             predictor_cls=image_predict_wrapper)
    
    #Deploy the model.
    predictor_ppe_detection = ppe_detection_model.deploy(1, 'ml.c5.xlarge', endpoint_name='personal-protective-equip-detection-endpoint')


Step 4.2: Perform real-time inference on the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Perform a prediction
    ppe_detection_result = json.loads(predictor_ppe_detection.predict(construction_image['byte_array']).decode('utf-8'))
    #Un-comment the following line to view the result returned by the model.
    #print(json.dumps(ppe_detection_result,indent=2))
    
    #Read original image.
    image=cv2.imread(construction_image['path'])
    
    #Plot inference result on the image
    for output in ppe_detection_result['output']:
        
        x1=int(output['bbox'][0])
        y1=int(output['bbox'][1])
        x2=int(output['bbox'][2])
        y2=int(output['bbox'][3])
        image=draw_bounding_box(image,x1,y1,x2,y2,'PPE',output['score'])
    
    #Display result
    show_image(image,'Personal protective equipments')

Note how the pre-trained model could identify the PPEs in the image with
high probabilities.

Step 5. Deploy the Construction Machines detection model
--------------------------------------------------------

Next, you will deploy `Construction Machines
Detector <https://aws.amazon.com/marketplace/pp/prodview-fuukizaiq5o7c?qid=1563549078039&sr=0-1&ref_=mlmp_gitdemo_indust>`__
to identify construction machines from an image.

Step 5.1: Deploy the model for performing real-time inference.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Get the model_package_arn
    machine_detection_modelpackage_arn = ModelPackageArnProvider.get_machine_detection_model_package_arn(region)
    
    #Define predictor wrapper class
    def machine_detection_predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session,content_type='image/jpeg')
    
    #create a deployable model.
    machine_detection_model = ModelPackage(role=role,
                                             model_package_arn=machine_detection_modelpackage_arn,
                                             sagemaker_session=sagemaker_session,
                                             predictor_cls=image_predict_wrapper)
    
    #Deploy the model
    predictor_machine_detection = machine_detection_model.deploy(1, 'ml.p3.2xlarge', endpoint_name='machine-detection-endpoint')


Step 5.2: Perform real-time inference on the model (Test 1).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Perform a prediction
    machine_detection_result = json.loads(predictor_machine_detection.predict(construction_image['byte_array']).decode('utf-8'))
    #Un-comment the following line to view the result returned by the model.
    #print(json.dumps(machine_detection_result,indent=2))
    
    #Read original image.
    image=cv2.imread(construction_image['path'])
    
    #Plot inference result on the image
    for output in machine_detection_result['outputs']['detections']:
        x1=output[0]
        y1=output[1]
        x2=output[2]
        y2=output[3]
        image=draw_bounding_box(image,x1,y1,x2,y2,output[4],output[5])
    
    #Display result
    show_image(image,'Construction machines test 1')

Model did not detect any construction machines since there were none. We
will now perform inference on one more image that shows construction
machinery such as an excavator.

Step 5.3: Perform real-time inference on the model (Test 2).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #Perform a prediction
    machine_detection_result = json.loads(predictor_machine_detection.predict(machines_image['byte_array']).decode('utf-8'))
    #print(json.dumps(machine_detection_result,indent=2))
    
    #Read original image.
    image=cv2.imread(machines_image['path'])
    
    #Plot the inference on the image
    for output in machine_detection_result['outputs']['detections']:
        x1=output[0]
        y1=output[1]
        x2=output[2]
        y2=output[3]
        image=draw_bounding_box(image,x1,y1,x2,y2,output[4],output[5])
    
    #Display result
    show_image(image,'Construction machines Test 2')

Note how the pre-trained model could detect both, a truck, and an
excavator from the picture.

Step 6. Generate actionable insights on video input
---------------------------------------------------

The pre-trained models demonstrated above accept images as an input.
However, the input data can also be in the form of a video. In this
section, you will see how to extract actionable insights from a video by
performing inference on snapshots.

.. code:: ipython3

    from IPython.display import HTML
    video_path='./video/construction-video.mp4'
    
    HTML('<iframe width="560" height="315" src="'+video_path+'?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')

Courtesy -
https://pixabay.com/videos/construction-road-excavator-worker-26239/
(Edited)

Analyzing hours of video footage can be tedious. Status summary reports
and rules can help detect non-compliance to trigger alarms. In this
section, we will generate following status summary from the video:

**Sample Summary report**\  No Alarm : 1 truck(s), 1 excavator(s), no
workers found. No Alarm : 2 truck(s), 1 excavator(s), 1 workers found.
No Alarm : 1 truck(s), 1 excavator(s), 1 workers found. No Alarm : 1
truck(s), 1 excavator(s), no workers found. **ALARM** : 1 worker(s)
wearing PPE but 0 wearing hard hats, 1 truck(s), 1 excavator(s) found.
No Alarm : 1 truck(s), 1 excavator(s), no workers found. **ALARM** : 1
worker(s) wearing PPE but 0 wearing hard hats, 1 truck(s), 1
excavator(s) found.

**Note**: There are couple instances in the video when the worker is not
visible because of an obstruction.

In this task, we will take a snapshot from the video every 1.5 seconds
and then perform inference on each snapshot to identify actionable
insights. The snapshot images from the video enable you to to generate
inferences from model packages that only support image payloads. In some
cases, this approach may help you scale usage of endpoints.

.. code:: ipython3

    capture = cv2.VideoCapture(video_path) 
    
    #Get number of frames from the video.
    framecount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #Take snapshot every 1.5 second(s)
    num_seconds=1.5
    
    skip_frames=capture.get(cv2.CAP_PROP_FPS)*num_seconds
    
    num_snapshots=int(framecount/skip_frames)
    
    #For this experiment, we extract an image every second so that we can utilize the endpoints more efficiently.
    for i in range(num_snapshots):
        flag, frame = capture.read()
        if flag:
            path = './video/snapshots/frame' + str(i) + '.jpg'
            print ('Creating snapshot on path - ' + path) 
            cv2.imwrite(path, frame) 
            capture.set(cv2.CAP_PROP_POS_FRAMES, ((i+1)*skip_frames))
    capture.release() 

Now that we have created snapshots from the video, let us create a
utility function that generates status summary

.. code:: ipython3

    #The following method accepts path of an image, performs inference on 
    #construction machines detector, PPE, hard-hat detector models and generates a status summary.
    def generate_status_summary(image_path):
        image_byte_array=[]
        
        num_trucks=0
        num_excavator=0
        num_ppe=0
        num_hard_hat=0
        
        # Open the image.
        with open(image_path, "rb") as image:
          image_byte_array = bytearray(image.read())
        
        # Count number of machines
        machine_detection_result = json.loads(predictor_machine_detection.predict(image_byte_array).decode('utf-8'))
        for output in machine_detection_result['outputs']['detections']:
            if output[5]>0.65:
                if output[4] =='TRUCK':
                    num_trucks+= 1
                if output[4] =='EXCAVATOR':
                    num_excavator+= 1
    
        # Count number of personal protective equipments(PPEs)
        ppe_detection_result = json.loads(predictor_ppe_detection.predict(image_byte_array).decode('utf-8'))
        for output in ppe_detection_result['output']:
            if output['score']>0.5:
                num_ppe+= 1
        
        # Count number of hard-hats
        hard_hat_detection_result = json.loads(predictor_hard_hat_detection.predict(image_byte_array).decode('utf-8'))
        for i in range(len(hard_hat_detection_result['boxes'])):
            if hard_hat_detection_result['scores'][i]>0.5:
                num_hard_hat+= 1
                
        # Create and return the summary.
        if num_ppe == num_hard_hat ==0:
            current_status="No Alarm : "+str(num_trucks)+" truck(s), "+str(num_excavator)+" excavator(s), no workers found."
        elif(num_ppe == num_hard_hat):
            current_status="No Alarm : "+str(num_trucks)+" truck(s), "+str(num_excavator)+" excavator(s), "+str(num_ppe)+" workers found."
        elif num_ppe>num_hard_hat:
            current_status="ALARM    : "+str(num_ppe)+" worker(s) wearing PPE but " +str(num_hard_hat)+" wearing hard hats, "+str(num_trucks)+" truck(s), "+str(num_excavator)+" excavator(s) found."
        elif num_hard_hat>num_ppe:
            current_status="ALARM    : "+str(num_hard_hat)+" worker(s) wearing hard hats but "+str(num_ppe)+" workers wearing PPE, "+str(num_trucks)+" truck(s), and "+str(num_excavator)+" excavator(s) found."
        return current_status


Next, we will run the utility function on each snapshot to generate
status summary log from the video.

.. code:: ipython3

    #Initialize start-time with timestamp for first entry.
    start_time='00:00:{:0>3d}'.format(0)
    previous_status=''
    
    print("(Start)HH:mm:SSS-(End)HH:mm:SSS : Alarm/No alarm : Status Details")
    print("---------------------------------------------------------------")
    
    #next, we loop on each of the screenshot and extract summary. If summary for a screenshot 
    #matches with summary of previous screenshot, then we simply record the duration instead of
    #adding a duplicate summary record.
    
    for j in range(num_snapshots):
        
        image_path='./video/snapshots/frame' + str(j) + '.jpg'
        current_status = generate_status_summary(image_path)
        
        if previous_status=='':
            #For first record, populate the previous_status as current_status.
            previous_status=current_status
        
        #This means that summary status of the picture has changed. print the previous status and
        #start tracking new status.
        elif previous_status!=current_status:
            
            #map j to seconds value.
            end_time='00:00:{:0>3d}'.format(int(j*num_seconds*10))
            
            #print the previous status.
            print(start_time+"-"+(end_time)+" : " +previous_status)
            
            #Update end-time
            start_time=end_time
            previous_status=current_status
    
    #Print the final summary.
    print(start_time+"-"+"End"+" : "+ previous_status)


Step 7. Explore other relevant models!
--------------------------------------

You just learnt how pre-trained machine learning models can identify
metadata from workplace pictures (or snapshots of a video). This
metadata can be used to set up alarms to detect non-compliance.

Checkout these additional relevant models: 1. `Person and Truck
Detector <https://aws.amazon.com/marketplace/pp/prodview-mxkmbwcmojzg4?qid=1563549078039&sr=0-5&ref_=mlmp_gitdemo_indust>`__
to identify trucks and people from an image. 2. `Modjoul Geo Fence
model <https://aws.amazon.com/marketplace/pp/prodview-bspkbdfyfj42e?qid=1567887787959&sr=0-4&ref_=mlmp_gitdemo_indust>`__
informs an organization of employee and equipment location and the
activities and movements within that location. 3. `Modjoul Automotive
Telematics
Model <https://aws.amazon.com/marketplace/pp/prodview-cj46uchjavfa6?qid=1567887787959&sr=0-6&ref_=mlmp_gitdemo_indust>`__
can identify aggressive events such as hard braking and hard
acceleration, duration of driving and distance of driving. 4. `Modjoul
Asset Utilization
Model <https://aws.amazon.com/marketplace/pp/prodview-6ay5xkpc6lqbi?qid=1567887787959&sr=0-1&ref_=mlmp_gitdemo_indust>`__
to understand the utilization of heavy equipments such as back hoes,
generators, dump trucks, etc.

Step 8. Cleanup
---------------

Next, clean-up deployable models as well as endpoints from your account.

.. code:: ipython3

    predictor_construction_worker_detection.delete_endpoint()
    predictor_construction_worker_detection.delete_model()

.. code:: ipython3

    predictor_hard_hat_detection.delete_endpoint()
    predictor_hard_hat_detection.delete_model()

.. code:: ipython3

    predictor_ppe_detection.delete_endpoint()
    predictor_ppe_detection.delete_model()

.. code:: ipython3

    predictor_machine_detection.delete_endpoint()
    predictor_machine_detection.delete_model()

If you would like to unsubscribe to the model, follow these steps.
Before you cancel the subscription, ensure that you do not have any
`deployable
model <https://console.aws.amazon.com/sagemaker/home#/models>`__ created
from the model package or using the algorithm. Note - You can find this
information by looking at the container name associated with the model.

**Steps to unsubscribe to product from AWS Marketplace**: 1. Navigate to
**Machine Learning** tab on `Your Software subscriptions
page <https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=mlmp_gitdemo_indust>`__
2. Locate the listing that you would need to cancel subscription for,
and then choose **Cancel Subscription** to cancel the subscription.

