Fairseq in Amazon SageMaker: Translation task - English to French
=================================================================

In this notebook, we will show you how to train an English to French
translation model using a fully convolutional architecture using the
`Fairseq toolkit <https://github.com/pytorch/fairseq>`__

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

To train the model, we will be using the WMT’14 dataset as descibed
`here <https://github.com/pytorch/fairseq/tree/master/examples/translation#prepare-wmt14en2frsh>`__.

First, we’ll download the dataset and start the pre-processing. Among
other steps, this pre-processing cleans the tokens and applys BPE
encoding as you can see
`here <https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh>`__.

.. code:: sh

    %%sh
    cd data
    chmod +x prepare-wmt14en2fr.sh
    
    # Download dataset and start pre-processing
    ./prepare-wmt14en2fr.sh

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
    TEXT=../data/wmt14_en_fr
    python preprocess.py --source-lang en --target-lang fr \
      --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
      --destdir ../data/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0

The dataset is now all prepared for training on one of the Fairseq
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
    prefix = 'sagemaker/DEMO-pytorch-fairseq/datasets/wmt14_en_fr'
    
    role = sagemaker.get_execution_role()

.. code:: ipython3

    inputs = sagemaker_session.upload_data(path='data/wmt14_en_fr', bucket=bucket, key_prefix=prefix)

Build Fairseq Translation task container
----------------------------------------

Next we need to register a Docker image in Amazon SageMaker that will
contain the Fairseq code and that will be pulled at training and
inference time to perform the respective training of the model and the
serving of the precitions.

.. code:: sh

    %%sh
    chmod +x create_container.sh 
    
    ./create_container.sh pytorch-fairseq

The Fairseq image has been pushed into Amazon ECR, the registry from
which Amazon SageMaker will be able to pull that image and launch both
training and prediction.

Training on Amazon SageMaker
----------------------------

Next we will set the hyper-parameters of the model we want to train.
Here we are using the recommended ones from the `Fairseq
example <https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh>`__.

.. code:: ipython3

    hyperparameters = {
        "lr": 0.5,    
        "clip-norm": 0.1,
        "dropout": 0.1,
        "max-tokens": 3000,
        "criterion": "label_smoothed_cross_entropy",
        "label-smoothing": 0.1,
        "lr-scheduler": "fixed",
        "force-anneal": 50,
        "arch": "fconv_wmt_en_fr"
    }

We are ready to define the Estimator, which will encapsulate all the
required parameters needed for launching the training on Amazon
SageMaker. For training, the Fairseq toolkit recommends to train on GPU
instances, such as the ``ml.p3`` instance family `available in Amazon
SageMaker <https://aws.amazon.com/sagemaker/pricing/instance-types/>`__.

.. code:: ipython3

    from sagemaker.estimator import Estimator
    
    algorithm_name = "pytorch-fairseq"
    image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)
    
    estimator = Estimator(image,
                         role,
                         train_instance_count=1,
                         train_instance_type='ml.p3.8xlarge',
                         output_path='s3://{}/output'.format(bucket),
                         sagemaker_session=sagemaker_session,
                         hyperparameters=hyperparameters)

The call to fit will launch the training job and regularly report on the
different performance metrics such as losses.

.. code:: ipython3

    estimator.fit(inputs=inputs)

The model has finished training, we can go ahead and test its
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

Now it’s your time to play. Input a sentence in English and get the
translation in French by simply calling predict.

.. code:: ipython3

    import html
    
    text_input = "Hey, how you're doing?"
    
    result = predictor.predict(text_input)
    #  Some characters are escaped HTML-style requiring to unescape them before printing
    print(html.unescape(result))

Once you’re done with getting predictions, remember to shut down your
endpoint as you no longer need it.

Delete endpoint
---------------

.. code:: ipython3

    sagemaker_session.delete_endpoint(predictor.endpoint)

Voila! For more information, you can check out the `Fairseq toolkit
homepage <https://github.com/pytorch/fairseq>`__.
