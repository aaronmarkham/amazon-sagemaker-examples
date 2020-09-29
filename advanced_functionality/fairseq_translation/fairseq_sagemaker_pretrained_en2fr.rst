Fairseq in Amazon SageMaker: Pre-trained English to French translation model
============================================================================

In this notebook, we will show you how to serve an English to French
translation model using pre-trained model provided by the `Fairseq
toolkit <https://github.com/pytorch/fairseq>`__

Permissions
-----------

Running this notebook requires permissions in addition to the regular
SageMakerFullAccess permissions. This is because it creates new
repositories in Amazon ECR. The easiest way to add these permissions is
simply to add the managed policy AmazonEC2ContainerRegistryFullAccess to
the role that you used to start your notebook instance. There’s no need
to restart your notebook instance when you do this, the new permissions
will be available immediately.

Download pre-trained model
--------------------------

Fairseq maintains their pre-trained models
`here <https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md>`__.
We will use the model that was pre-trained on the `WMT14
English-French <http://statmt.org/wmt14/translation-task.html#Download>`__
dataset. As the models are archived in .bz2 format, we need to convert
them to .tar.gz as this is the format supported by Amazon SageMaker.

Convert archive
~~~~~~~~~~~~~~~

.. code:: sh

    %%sh
    
    wget https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2
    
    tar xvjf wmt14.v2.en-fr.fconv-py.tar.bz2 > /dev/null
    cd wmt14.en-fr.fconv-py
    mv model.pt checkpoint_best.pt
    
    tar czvf wmt14.en-fr.fconv-py.tar.gz checkpoint_best.pt dict.en.txt dict.fr.txt bpecodes README.md > /dev/null

The pre-trained model has been downloaded and converted. The next step
is upload the data to Amazon S3 in order to make it available for
running the inference.

Upload data to Amazon S3
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    region =  sagemaker_session.boto_session.region_name
    account = sagemaker_session.boto_session.client('sts').get_caller_identity().get('Account')
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-fairseq/pre-trained-models'
    
    role = sagemaker.get_execution_role()

.. code:: ipython3

    trained_model_location = sagemaker_session.upload_data(
        path='wmt14.en-fr.fconv-py/wmt14.en-fr.fconv-py.tar.gz',
        bucket=bucket,
        key_prefix=prefix)

Build Fairseq serving container
-------------------------------

Next we need to register a Docker image in Amazon SageMaker that will
contain the Fairseq code and that will be pulled at inference time to
perform the of the precitions from the pre-trained model we downloaded.

.. code:: sh

    %%sh
    chmod +x create_container.sh 
    
    ./create_container.sh pytorch-fairseq-serve

The Fairseq serving image has been pushed into Amazon ECR, the registry
from which Amazon SageMaker will be able to pull that image and launch
both training and prediction.

Hosting the pre-trained model for inference
-------------------------------------------

We first needs to define a base JSONPredictor class that will help us
with sending predictions to the model once it’s hosted on the Amazon
SageMaker endpoint.

.. code:: ipython3

    from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer
    
    class JSONPredictor(RealTimePredictor):
        def __init__(self, endpoint_name, sagemaker_session):
            super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)

We can now use the Model class to deploy the model artificats (the
pre-trained model), and deploy it on a CPU instance. Let’s use a
``ml.m5.xlarge``.

.. code:: ipython3

    from sagemaker import Model
    
    algorithm_name = "pytorch-fairseq-serve"
    image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)
    
    model = Model(model_data=trained_model_location,
                  role=role,
                  image=image,
                  predictor_cls=JSONPredictor,
                 )

.. code:: ipython3

    predictor = model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')

Now it’s your time to play. Input a sentence in English and get the
translation in French by simply calling predict.

.. code:: ipython3

    import html
    
    result = predictor.predict("I love translation")
    # Some characters are escaped HTML-style requiring to unescape them before printing
    print(html.unescape(result))

Once you’re done with getting predictions, remember to shut down your
endpoint as you no longer need it.

Delete endpoint
---------------

.. code:: ipython3

    model.sagemaker_session.delete_endpoint(predictor.endpoint)

Voila! For more information, you can check out the `Fairseq toolkit
homepage <https://github.com/pytorch/fairseq>`__.
