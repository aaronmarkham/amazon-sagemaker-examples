FAIRSeq in Amazon SageMaker: Translation task - German to English - Distributed / multi machine training
========================================================================================================

The Facebook AI Research (FAIR) Lab made available through the `FAIRSeq
toolkit <https://github.com/pytorch/fairseq>`__ their state-of-the-art
Sequence to Sequence models.

In this notebook, we will show you how to train a German to English
translation model using a fully convolutional architecture on multiple
GPUs and machines.

Permissions
-----------

Running this notebook requires permissions in addition to the regular
SageMakerFullAccess permissions. This is because it creates new
repositories in Amazon ECR. The easiest way to add these permissions is
simply to add the managed policy AmazonEC2ContainerRegistryFullAccess to
the role that you used to start your notebook instance. There’s no need
to restart your notebook instance when you do this, the new permissions
will be available immediately.

Prepare dataset
---------------

To train the model, we will be using the IWSLT’14 dataset as descibed
`here <https://github.com/pytorch/fairseq/tree/master/examples/translation#prepare-iwslt14sh>`__.
This was used in the IWSLT’14 German to English translation task:
`“Report on the 11th IWSLT evaluation campaign” by Cettolo et
al <http://workshop2014.iwslt.org/downloads/proceeding.pdf>`__.

First, we’ll download the dataset and start the pre-processing. Among
other steps, this pre-processing cleans the tokens and applys BPE
encoding as you can see
`here <https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh>`__.

.. code:: sh

    %%sh
    cd data
    chmod +x prepare-iwslt14.sh
    
    # Download dataset and start pre-processing
    ./prepare-iwslt14.sh

Next step is to apply the second set of pre-processing, which binarizes
the dataset based on the source and target language. Full information on
this script
`here <https://github.com/pytorch/fairseq/blob/master/preprocess.py>`__.

.. code:: sh

    %%sh
    
    # First we download fairseq in order to have access to the scripts
    git clone https://github.com/pytorch/fairseq.git fairseq-git
    cd fairseq-git
    
    # Binarize the dataset:
    TEXT=../data/iwslt14.tokenized.de-en
    python preprocess.py --source-lang de --target-lang en \
      --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
      --destdir ..data/iwslt14.tokenized.de-en

The dataset is now all prepared for training on one of the FAIRSeq
translation models. The next step is upload the data to Amazon S3 in
order to make it available for training.

Upload data to Amazon S3
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import sagemaker
    
    sagemaker_session = sagemaker.Session()
    region =  sagemaker_session.boto_session.region_name
    account = sagemaker_session.boto_session.client('sts').get_caller_identity().get('Account')
    
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-pytorch-fairseq/datasets/iwslt14'
    
    role = sagemaker.get_execution_role()

.. code:: ipython3

    inputs = sagemaker_session.upload_data(path='data/iwslt14.tokenized.de-en', bucket=bucket, key_prefix=prefix)

Next we need to register a Docker image in Amazon SageMaker that will
contain the FAIRSeq code and that will be pulled at training and
inference time to perform the respective training of the model and the
serving of the precitions.

Build FAIRSeq Translation task container
----------------------------------------

.. code:: sh

    %%sh
    chmod +x create_container.sh 
    
    ./create_container.sh pytorch-fairseq

The FAIRSeq image has been pushed into Amazon ECR, the registry from
which Amazon SageMaker will be able to pull that image and launch both
training and prediction.

Training on Amazon SageMaker
----------------------------

Next we will set the hyper-parameters of the model we want to train.
Here we are using the recommended ones from the `FAIRSeq
example <https://github.com/pytorch/fairseq/tree/master/examples/translation#prepare-iwslt14sh>`__.
The full list of hyper-parameters available for use can be found
`here <https://fairseq.readthedocs.io/en/latest/command_line_tools.html>`__.
Please note you can use dataset, training, and generation parameters.
For the distributed backend, **gloo** is the only supported option and
is set as default.

.. code:: ipython3

    hyperparameters = {
        "lr": 0.25,    
        "clip-norm": 0.1,
        "dropout": 0.2,
        "max-tokens": 4000,
        "criterion": "label_smoothed_cross_entropy",
        "label-smoothing": 0.1,
        "lr-scheduler": "fixed",
        "force-anneal": 200,
        "arch": "fconv_iwslt_de_en"
    }

We are ready to define the Estimator, which will encapsulate all the
required parameters needed for launching the training on Amazon
SageMaker.

For training, the FAIRSeq toolkit recommends to train on GPU instances,
such as the ``ml.p3`` instance family `available in Amazon
SageMaker <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__.
In this example, we are training on 2 instances.

.. code:: ipython3

    from sagemaker.estimator import Estimator
    
    algorithm_name = "pytorch-fairseq"
    image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)
    
    estimator = Estimator(image,
                         role,
                         train_instance_count=2,
                         train_instance_type='ml.p3.8xlarge',
                         train_volume_size=100, 
                         output_path='s3://{}/output'.format(bucket),
                         hyperparameters=hyperparameters)

The call to fit will launch the training job and regularly report on the
different performance metrics related to the training.

.. code:: ipython3

    estimator.fit(inputs=inputs)

Once the model has finished training, we can go ahead and test its
translation capabilities by deploying it on an endpoint.

Hosting the model
-----------------

We first need to define a base JSONPredictor class that will help us
with sending predictions to the model once it’s hosted on the Amazon
SageMaker endpoint.

.. code:: ipython3

    from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer
    
    class JSONPredictor(RealTimePredictor):
        def __init__(self, endpoint_name, sagemaker_session):
            super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)

We can now use the estimator object to deploy the model artificats (the
trained model), and deploy it on a CPU instance as we no longer need a
GPU instance for simply infering from the model. Let’s use a
``ml.m5.xlarge``.

.. code:: ipython3

    predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge', predictor_cls=JSONPredictor)

Now it’s your time to play. Input a sentence in German and get the
translation in English by simply calling predict.

.. code:: ipython3

    import html
    
    text_input = 'Guten Morgen'
    
    result = predictor.predict(text_input)
    #  Some characters are escaped HTML-style requiring to unescape them before printing
    print(html.unescape(result))

Once you’re done with getting predictions, remember to shut down your
endpoint as you no longer need it.

Delete endpoint
---------------

.. code:: ipython3

    sagemaker_session.delete_endpoint(predictor.endpoint)

Voila! For more information, you can check out the `FAIRSeq toolkit
homepage <https://github.com/pytorch/fairseq>`__.
