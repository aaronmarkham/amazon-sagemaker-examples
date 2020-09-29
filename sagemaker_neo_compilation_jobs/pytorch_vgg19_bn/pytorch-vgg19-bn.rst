Deploying pre-trained PyTorch VGG19 model with Amazon SageMaker Neo
===================================================================

Amazon SageMaker Neo is API to compile machine learning models to
optimize them for our choice of hardward targets. Currently, Neo
supports pre-trained PyTorch models from
`TorchVision <https://pytorch.org/docs/stable/torchvision/models.html>`__.
General support for other PyTorch models is forthcoming.

In this example notebook, we will compare the performace of PyTorch
pretrained Vgg19_bn model before versus after compilation using Neo.

Pytorch Vgg19_bn model is one of the models that benefits a lot from
compilation with Neo. Here we will verify that in end to end compilation
and inference on sagemaker endpoints, Neo compiled model can get seven
times speedup with no loss in accuracy.

.. code:: ipython3

    !~/anaconda3/envs/pytorch_p36/bin/pip install torch==1.2.0 torchvision==0.4.0

Import VGG19 from TorchVision
-----------------------------

We’ll import `VGG19_bn <https://arxiv.org/pdf/1409.1556.pdf>`__ model
from TorchVision and create a model artifact ``model.tar.gz``:

.. code:: ipython3

    import torch
    import torchvision.models as models
    import tarfile

.. code:: ipython3

    vgg19_bn = models.vgg19_bn(pretrained=True)
    input_shape = [1,3,224,224]
    trace = torch.jit.trace(vgg19_bn.float().eval(), torch.zeros(input_shape).float())
    trace.save('model.pth')
    
    with tarfile.open('model.tar.gz', 'w:gz') as f:
        f.add('model.pth')

Set up the environment
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import boto3
    import sagemaker
    import time
    from sagemaker.utils import name_from_base
    
    role = sagemaker.get_execution_role()
    sess = sagemaker.Session()
    region = sess.boto_region_name
    bucket = sess.default_bucket()
    
    compilation_job_name = name_from_base('TorchVision-vgg19-Neo')
    prefix = compilation_job_name+'/model'
    
    model_path = sess.upload_data(path='model.tar.gz', key_prefix=prefix)
    
    data_shape = '{"input0":[1,3,224,224]}'
    target_device = 'ml_m5'
    framework = 'PYTORCH'
    framework_version = '1.2.0'
    compiled_model_path = 's3://{}/{}/output'.format(bucket, compilation_job_name)

Use sagemaker PyTorchModel to load pretained PyTorch model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from sagemaker.pytorch.model import PyTorchModel
    
    pt_vgg = PyTorchModel(model_data=model_path,
                          framework_version=framework_version,
                          role=role,                               
                          entry_point='vgg19_bn.py',
                          sagemaker_session=sess,
                          py_version='py3'
                         )

Deploy the pretrained model to prepare for predictions(the old way)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    vgg_predictor = pt_vgg.deploy(initial_instance_count = 1,
                                  instance_type = 'ml.m5.12xlarge'
                                 )

Invoke the endpoint
~~~~~~~~~~~~~~~~~~~

Let’s test with a cat image.

.. code:: ipython3

    from IPython.display import Image
    Image('cat.jpg')  

Image Pre-processing
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    import torch
    from PIL import Image
    from torchvision import transforms
    import numpy as np
    input_image = Image.open('cat.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

Measure Inference Lantency
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    import time
    start = time.time()
    for _ in range(1000):
        output = vgg_predictor.predict(input_batch)
    inference_time = (time.time()-start)
    print('Inference time is ' + str(inference_time) + 'millisecond')

.. code:: ipython3

    _, predicted = torch.max(torch.from_numpy(np.array(output)), 1)

.. code:: ipython3

    # Load names for ImageNet classes
    object_categories = {}
    with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
        for line in f:
            key, val = line.strip().split(':')
            object_categories[key] = val

.. code:: ipython3

    print("Result: label - " + object_categories[str(predicted.item())])

Clean-up
~~~~~~~~

Deleting the local endpoint when you’re finished is important since you
can only run one local endpoint at a time.

.. code:: ipython3

    sess.delete_endpoint(vgg_predictor.endpoint)

Neo optimization
----------------

Fetch neo container image for PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from sagemaker.model import NEO_IMAGE_ACCOUNT
    from sagemaker.fw_utils import create_image_uri
    
    image_uri = create_image_uri(region, 'neo-' + framework.lower(), target_device.replace('_', '.'),
                                 framework_version, py_version='py3', account=NEO_IMAGE_ACCOUNT[region])

.. code:: ipython3

    from sagemaker.pytorch.model import PyTorchModel
    from sagemaker.predictor import RealTimePredictor
    
    sagemaker_model = PyTorchModel(model_data=model_path,
                                   image=image_uri,
                                   predictor_cls=RealTimePredictor,
                                   framework_version = framework_version,
                                   role=role,
                                   sagemaker_session=sess,
                                   entry_point='vgg19_bn.py',
                                   py_version='py3'
                                  )

Use Neo compiler to compile the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    compiled_model = sagemaker_model.compile(target_instance_family=target_device, 
                                             input_shape=data_shape,
                                             job_name=compilation_job_name,
                                             role=role,
                                             framework=framework,
                                             framework_version=framework_version,
                                             output_path=compiled_model_path
                                            )

.. code:: ipython3

    predictor = compiled_model.deploy(initial_instance_count = 1,
                                      instance_type = 'ml.m5.12xlarge'
                                     )

.. code:: ipython3

    import json
    
    with open('cat.jpg', 'rb') as f:
        payload = f.read()
        payload = bytearray(payload) 

.. code:: ipython3

    predictor.content_type = 'application/x-image'

Measure Inference Lantency
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    import time
    start = time.time()
    for _ in range(1000):
        response = predictor.predict(payload)
    neo_inference_time = (time.time()-start)
    print('Neo optimized inference time is ' + str(neo_inference_time) + 'millisecond')

.. code:: ipython3

    result = json.loads(response.decode())
    print('Most likely class: {}'.format(np.argmax(result)))
    print("Result: label - " + object_categories[str(np.argmax(result))]+ " probability - " + str(np.amax(result)))

.. code:: ipython3

    sess.delete_endpoint(predictor.endpoint)
