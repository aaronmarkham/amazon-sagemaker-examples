Deploy a previously created model in SageMaker
==============================================

Sagemaker decouples model creation/fitting and model deployment. **This
short notebook shows how you can deploy a model that you have already
created**. It is assumed that you have already created the model and it
appears in the ``Models`` section of the SageMaker console. Obviously,
before you deploy a model the model must exist, so please go back and
make sure you have already fit/created the model before proceeding. For
more information about deploying models, see
https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-deploy-model.html

.. code:: ipython3

    import boto3
    from time import gmtime,strftime

.. code:: ipython3

    #configs for model, endpoint and batch transform
    model_name='ENTER MODEL NAME' #enter name of a model from the 'Model panel' in the AWS SageMaker console.
    sm=boto3.client('sagemaker')

Deploy using an inference endpoint
----------------------------------

.. code:: ipython3

    #set endpoint name/config.
    endpoint_config_name = 'DEMO-model-config-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    endpoint_name = 'DEMO-model-config-'  + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

.. code:: ipython3

    create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[{
            'InstanceType':'ml.m4.xlarge',
            'InitialVariantWeight':1,
            'InitialInstanceCount':1,
            'ModelName':model_name,
            'VariantName':'AllTraffic'}])
    
    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])
    
    
    create_endpoint_response = sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print(create_endpoint_response['EndpointArn'])
    
    resp = sm.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print("Status: " + status)

If you go to the AWS SageMaker service console now, you should see that
the endpoint creation is in progress.

Deploy using a batch transform job
----------------------------------

A batch transform job should be used for when you want to create
inferences on a dateset and then shut down the resources when inference
is finished.

.. code:: ipython3

    #config for batch transform
    batch_job_name='ENTER_JOB_NAME'
    output_location='ENDER_OUTPUT_LOCATION' #S3 bucket/location
    input_location= 'ENTER_INPUT_LOCATION'  #S3 bucket/location

.. code:: ipython3

    request = {
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "TransformOutput": {
            "S3OutputPath": output_location,
            "Accept": "text/csv",
            "AssembleWith": "Line"
        },
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_location 
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
            "CompressionType": "None"
        },
        "TransformResources": {
                "InstanceType": "ml.m4.xlarge", #change this based on what resources you want to request
                "InstanceCount": 1
        }
    }
    sm.create_transform_job(**request)
