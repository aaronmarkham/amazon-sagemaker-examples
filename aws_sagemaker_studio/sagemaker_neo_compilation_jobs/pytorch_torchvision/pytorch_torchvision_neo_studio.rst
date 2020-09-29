Deploying pre-trained PyTorch vision models with Amazon SageMaker Neo
=====================================================================

Amazon SageMaker Neo is API to compile machine learning models to
optimize them for our choice of hardward targets. Currently, Neo
supports pre-trained PyTorch models from
`TorchVision <https://pytorch.org/docs/stable/torchvision/models.html>`__.
General support for other PyTorch models is forthcoming.

.. code:: ipython3

    %cd /root/amazon-sagemaker-examples/aws_sagemaker_studio/sagemaker_neo_compilation_jobs/pytorch_torchvision

.. code:: ipython3

    import sys
    !{sys.executable} -m pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

Import ResNet18 from TorchVision
--------------------------------

We’ll import `ResNet18 <https://arxiv.org/abs/1512.03385>`__ model from
TorchVision and create a model artifact ``model.tar.gz``:

.. code:: ipython3

    import torch
    from torchvision import models
    import tarfile
    
    resnet18 = models.resnet18(pretrained=True)
    input_shape = [1,3,224,224]
    trace = torch.jit.trace(resnet18.float().eval(), torch.zeros(input_shape).float())
    trace.save('model.pth')
    
    with tarfile.open('model.tar.gz', 'w:gz') as f:
        f.add('model.pth')

Invoke Neo Compilation API
--------------------------

We then forward the model artifact to Neo Compilation API:

.. code:: ipython3

    import boto3
    import sagemaker
    import time
    from sagemaker.utils import name_from_base
    
    role = sagemaker.get_execution_role()
    sess = sagemaker.Session()
    region = sess.boto_region_name
    bucket = sess.default_bucket()
    
    compilation_job_name = name_from_base('TorchVision-ResNet18-Neo')
    
    model_key = '{}/model/model.tar.gz'.format(compilation_job_name)
    model_path = 's3://{}/{}'.format(bucket, model_key)
    boto3.resource('s3').Bucket(bucket).upload_file('model.tar.gz', model_key)
    
    sm_client = boto3.client('sagemaker')
    data_shape = '{"input0":[1,3,224,224]}'
    target_device = 'ml_c5'
    framework = 'PYTORCH'
    framework_version = '1.2.0'
    compiled_model_path = 's3://{}/{}/output'.format(bucket, compilation_job_name)

.. code:: ipython3

    response = sm_client.create_compilation_job(
        CompilationJobName=compilation_job_name,
        RoleArn=role,
        InputConfig={
            'S3Uri': model_path,
            'DataInputConfig': data_shape,
            'Framework': framework
        },
        OutputConfig={
            'S3OutputLocation': compiled_model_path,
            'TargetDevice': target_device
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 300
        }
    )
    print(response)
    
    # Poll every 30 sec
    while True:
        response = sm_client.describe_compilation_job(CompilationJobName=compilation_job_name)
        if response['CompilationJobStatus'] == 'COMPLETED':
            break
        elif response['CompilationJobStatus'] == 'FAILED':
            raise RuntimeError('Compilation failed')
        print('Compiling ...')
        time.sleep(30)
    print('Done!')
    
    # Extract compiled model artifact
    compiled_model_path = response['ModelArtifacts']['S3ModelArtifacts']

Create prediction endpoint
--------------------------

To create a prediction endpoint, we first specify two additional
functions, to be used with Neo Deep Learning Runtime:

-  ``neo_preprocess(payload, content_type)``: Function that takes in the
   payload and Content-Type of each incoming request and returns a NumPy
   array. Here, the payload is byte-encoded NumPy array, so the function
   simply decodes the bytes to obtain the NumPy array.
-  ``neo_postprocess(result)``: Function that takes the prediction
   results produced by Deep Learining Runtime and returns the response
   body

.. code:: ipython3

    !pygmentize resnet18.py

Upload the Python script containing the two functions to S3:

.. code:: ipython3

    source_key = '{}/source/sourcedir.tar.gz'.format(compilation_job_name)
    source_path = 's3://{}/{}'.format(bucket, source_key)
    
    with tarfile.open('sourcedir.tar.gz', 'w:gz') as f:
        f.add('resnet18.py')
    
    boto3.resource('s3').Bucket(bucket).upload_file('sourcedir.tar.gz', source_key)

We then create a SageMaker model record:

.. code:: ipython3

    from sagemaker.model import NEO_IMAGE_ACCOUNT
    from sagemaker.fw_utils import create_image_uri
    
    model_name = name_from_base('TorchVision-ResNet18-Neo')
    
    image_uri = create_image_uri(region, 'neo-' + framework.lower(), target_device.replace('_', '.'),
                                 framework_version, py_version='py3', account=NEO_IMAGE_ACCOUNT[region])
    
    response = sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': compiled_model_path,
            'Environment': { 'SAGEMAKER_SUBMIT_DIRECTORY': source_path }
        },
        ExecutionRoleArn=role
    )
    print(response)

Then we create an Endpoint Configuration:

.. code:: ipython3

    config_name = model_name
    
    response = sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                'VariantName': 'default-variant-name',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.c5.xlarge',
                'InitialVariantWeight': 1.0
            },
        ],
    )
    print(response)

Finally, we create an Endpoint:

.. code:: ipython3

    endpoint_name = model_name + '-Endpoint'
    
    response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )
    print(response)
    
    print('Creating endpoint ...')
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    
    response = sm_client.describe_endpoint(EndpointName=endpoint_name)
    print(response)

Send requests
-------------

Let’s try to send a cat picture.

.. figure:: cat.jpg
   :alt: title

   title

.. code:: ipython3

    import json
    import numpy as np
    
    sm_runtime = boto3.Session().client('sagemaker-runtime')
    
    with open('cat.jpg', 'rb') as f:
        payload = f.read()
    
    response = sm_runtime.invoke_endpoint(EndpointName=endpoint_name,
                                          ContentType='application/x-image',
                                          Body=payload)
    print(response)
    result = json.loads(response['Body'].read().decode())
    print('Most likely class: {}'.format(np.argmax(result)))

.. code:: ipython3

    # Load names for ImageNet classes
    object_categories = {}
    with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
        for line in f:
            key, val = line.strip().split(':')
            object_categories[key] = val
    print("Result: label - " + object_categories[str(np.argmax(result))]+ " probability - " + str(np.amax(result)))

Delete the Endpoint
-------------------

Having an endpoint running will incur some costs. Therefore as a
clean-up job, we should delete the endpoint.

.. code:: ipython3

    sess.delete_endpoint(endpoint_name)
