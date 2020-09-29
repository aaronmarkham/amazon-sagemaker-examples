Introduction
------------

In this notebook, we demonstrate how BlazingText supports hosting of
pre-trained Text Classification and Word2Vec models `FastText
models <https://fasttext.cc/docs/en/english-vectors.html>`__.
BlazingText is a GPU accelerated version of FastText. FastText is a
shallow Neural Network model used to perform both word embedding
generation (unsupervised) and text classification (supervised).
BlazingText uses custom CUDA kernels to accelerate the training process
of FastText but the underlying algorithm is same for both the
algorithms. Therefore, if you have a model trained with FastText or if
one of the pre-trained models made available by FastText team is
sufficient for your use case, then you can take advantage of Hosting
support for BlazingText to setup SageMaker endpoints for realtime
predictions using FastText models. It can help you avoid to train with
BlazingText algorithm if your use-case is covered by the pre-trained
models available from FastText.

To start the proceedings, we will specify few of the important parameter
like IAM Role and S3 bucket location which is required for SageMaker to
facilitate model hosting. SageMaker Python SDK helps us to retrieve the
IAM role and also helps you to operate easily with S3 resources.

.. code:: ipython3

    import sagemaker
    from sagemaker import get_execution_role
    import boto3
    import json
    
    sess = sagemaker.Session()
    
    role = get_execution_role()
    print(role) # This is the role that SageMaker would use to leverage AWS resources (S3, CloudWatch) on your behalf
    
    bucket = sess.default_bucket() # Replace with your own bucket name if needed
    print(bucket)
    prefix = 'fasttext/pretrained' #Replace with the prefix under which you want to store the data if needed

.. code:: ipython3

    region_name = boto3.Session().region_name

.. code:: ipython3

    container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")
    print('Using SageMaker BlazingText container: {} ({})'.format(container, region_name))

Hosting the `Language Idenfication model <https://fasttext.cc/docs/en/language-identification.html>`__ by FastText
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the example, we will leverage the pre-trained model available by
FastText for Language Identification. Language Identification is the
first step of many NLP applications where after the language of the
input text is identified, specific models for that language needs to be
applied for various other downstream tasks. Language Identification
underneath is a Text Classification model which uses the language IDs as
the class labels and hence FastText can be directly used for the
training. FastText pretrained language model supports identification of
176 different languages.

Here we will download the Language Identification (Text Classification)
model [1] from `FastText
website <https://fasttext.cc/docs/en/language-identification.html>`__.

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for
Efficient Text Classification

.. code:: ipython3

    !wget -O model.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

Next we will ``tar`` the model and upload it to S3 with the help of
utilities available from Python SDK. We’ll delete the local copies of
the data as it’s not required anymore.

.. code:: ipython3

    !tar -czvf langid.tar.gz model.bin
    model_location = sess.upload_data("langid.tar.gz", bucket=bucket, key_prefix=prefix)
    !rm langid.tar.gz model.bin

Creating SageMaker Inference Endpoint
-------------------------------------

Next we’ll create a SageMaker inference endpoint with the BlazingText
container. This endpoint will be compatible with the pre-trained models
available from FastText and can be used for inference directly without
any modification. The inference endpoint works with content-type of
``application/json``.

.. code:: ipython3

    lang_id = sagemaker.Model(model_data=model_location, image=container, role=role, sagemaker_session=sess)
    lang_id.deploy(initial_instance_count = 1,instance_type = 'ml.m4.xlarge')
    predictor = sagemaker.RealTimePredictor(endpoint=lang_id.endpoint_name, 
                                       sagemaker_session=sess,
                                       serializer=json.dumps,
                                       deserializer=sagemaker.predictor.json_deserializer)

Next we’ll pass few sentences from various languages to the endpoint to
verify that the language identification works as expected.

.. code:: ipython3

    sentences = ["hi which language is this?",
                 "mon nom est Pierre",
                 "Dem Jungen gab ich einen Ball.",
                 "আমি বাড়ি যাবো."]
    payload = {"instances" : sentences}

.. code:: ipython3

    predictions = predictor.predict(payload)
    print(predictions)

FastText expects the class label to be prefixed by ``__label__`` and
that’s why when we are performing inference with pre-trained model
provided by FastText, we can see that the output label is prefixed with
``__label__``. With a little preprocessing, we can strip the
``__label__`` prefix from the response.

.. code:: ipython3

    import copy
    predictions_copy = copy.deepcopy(predictions) # Copying predictions object because we want to change the labels in-place
    for output in predictions_copy:
        output['label'] = output['label'][0][9:].upper() #__label__ has length of 9
    
    print(predictions_copy)

Stop / Close the Endpoint (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we should delete the endpoint before we close the notebook if
we don’t need to keep the endpoint running for serving realtime
predictions.

.. code:: ipython3

    sess.delete_endpoint(predictor.endpoint)

Similarly, we can host any pre-trained `FastText word2vec
model <https://fasttext.cc/docs/en/pretrained-vectors.html>`__ using
SageMaker BlazingText hosting.
