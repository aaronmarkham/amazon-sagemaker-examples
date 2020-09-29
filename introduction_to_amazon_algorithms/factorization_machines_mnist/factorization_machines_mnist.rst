An Introduction to Factorization Machines with MNIST
====================================================

**Making a Binary Prediction of Whether a Handwritten Digit is a 0**

1.  `Introduction <#Introduction>`__
2.  `Prerequisites and Preprocessing <#Prequisites-and-Preprocessing>`__
3.  `Permissions and environment
    variables <#Permissions-and-environment-variables>`__
4.  `Data ingestion <#Data-ingestion>`__
5.  `Data inspection <#Data-inspection>`__
6.  `Data conversion <#Data-conversion>`__
7.  `Training the FM model <#Training-the-FM-model>`__
8.  `Set up hosting for the model <#Set-up-hosting-for-the-model>`__
9.  `Import model into hosting <#Import-model-into-hosting>`__
10. `Create endpoint configuration <#Create-endpoint-configuration>`__
11. `Create endpoint <#Create-endpoint>`__
12. `Validate the model for use <#Validate-the-model-for-use>`__

Introduction
------------

Welcome to our example introducing Amazon SageMaker’s Factorization
Machines Algorithm! Today, we’re analyzing the
`MNIST <https://en.wikipedia.org/wiki/MNIST_database>`__ dataset which
consists of images of handwritten digits, from zero to nine. We’ll use
the individual pixel values from each 28 x 28 grayscale image to predict
a yes or no label of whether the digit is a 0 or some other digit (1, 2,
3, … 9).

The method that we’ll use is a factorization machine binary classifier.
A factorization machine is a general-purpose supervised learning
algorithm that you can use for both classification and regression tasks.
It is an extension of a linear model that is designed to parsimoniously
capture interactions between features in high dimensional sparse
datasets. For example, in a click prediction system, the factorization
machine model can capture click rate patterns observed when ads from a
certain ad-category are placed on pages from a certain page-category.
Factorization machines are a good choice for tasks dealing with high
dimensional sparse datasets, such as click prediction and item
recommendation.

Amazon SageMaker’s Factorization Machine algorithm provides a robust,
highly scalable implementation of this algorithm, which has become
extremely popular in ad click prediction and recommender systems. The
main purpose of this notebook is to quickly show the basics of
implementing Amazon SageMaker Factorization Machines, even if the use
case of predicting a digit from an image is not where factorization
machines shine.

To get started, we need to set up the environment with a few
prerequisite steps, for permissions, configurations, and so on.

Prequisites and Preprocessing
-----------------------------

Permissions and environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*This notebook was created and tested on an ml.m4.xlarge notebook
instance.*

Let’s start by specifying:

-  The S3 bucket and prefix that you want to use for training and model
   data. This should be within the same region as the Notebook Instance,
   training, and hosting.
-  The IAM role arn used to give training and hosting access to your
   data. See the documentation for how to create these. Note, if more
   than one role is required for notebook instances, training, and/or
   hosting, please replace the boto regexp with a the appropriate full
   IAM role arn string(s).

.. code:: ipython3

    import re
    import boto3
    import sagemaker
    from sagemaker import get_execution_role
    import json
    role = get_execution_role()
    
    print(role)
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(bucket)
    
    prefix = 'sagemaker/DEMO-fm-mnist'


Data ingestion
~~~~~~~~~~~~~~

Next, we read the dataset from an online URL into memory, for
preprocessing prior to training. This processing could be done *in situ*
by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc.,
assuming the dataset is present in the appropriate location. Then, the
next step would be to transfer the data to S3 for use in training. For
small datasets, such as this one, reading into memory isn’t onerous,
though it would be for larger datasets.

.. code:: ipython3

    %%time
    import pickle, gzip, numpy, urllib.request, json
    
    # Load the dataset
    urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Data inspection
~~~~~~~~~~~~~~~

Once the dataset is imported, it’s typical as part of the machine
learning process to inspect the data, understand the distributions, and
determine what type(s) of preprocessing might be needed. You can perform
those tasks right here in the notebook. As an example, let’s go ahead
and look at one of the digits that is part of the dataset.

.. code:: ipython3

    %matplotlib inline
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (2,10)
    
    
    def show_digit(img, caption='', subplot=None):
        if subplot==None:
            _,(subplot)=plt.subplots(1,1)
        imgr=img.reshape((28,28))
        subplot.axis('off')
        subplot.imshow(imgr, cmap='gray')
        plt.title(caption)
    
    show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))

Data conversion
~~~~~~~~~~~~~~~

Since algorithms have particular input and output requirements,
converting the dataset is also part of the process that a data scientist
goes through prior to initiating training. In this particular case, the
Amazon SageMaker implementation of Factorization Machines takes
recordIO-wrapped protobuf, where the data we have today is a pickle-ized
numpy array on disk.

Most of the conversion effort is handled by the Amazon SageMaker Python
SDK, imported as ``sagemaker`` below.

*Notice, despite the fact that most use cases for factorization machines
will utilize spare input, we are writing our data out as dense tensors.
This will be fine since the MNIST dataset is not particularly large or
high dimensional.*

.. code:: ipython3

    import io
    import numpy as np
    import sagemaker.amazon.common as smac
    
    vectors = np.array([t.tolist() for t in train_set[0]]).astype('float32')
    labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1.0, 0.0).astype('float32')
    
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, vectors, labels)
    buf.seek(0)

Upload training data
--------------------

Now that we’ve created our recordIO-wrapped protobuf, we’ll need to
upload it to S3, so that Amazon SageMaker training can use it.

.. code:: ipython3

    import boto3
    import os
    
    key = 'recordio-pb-data'
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
    s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
    print('uploaded training data location: {}'.format(s3_train_data))

Let’s also setup an output S3 location for the model artifact that will
be output as the result of training with the algorithm.

.. code:: ipython3

    output_location = 's3://{}/{}/output'.format(bucket, prefix)
    print('training artifacts will be uploaded to: {}'.format(output_location))

Training the factorization machine model
----------------------------------------

Once we have the data preprocessed and available in the correct format
for training, the next step is to actually train the model using the
data. Since this data is relatively small, it isn’t meant to show off
the performance of the Amazon SageMaker’s Factorization Machines in
training, although we have tested it on multi-terabyte datasets.

Again, we’ll use the Amazon SageMaker Python SDK to kick off training
and monitor status until it is completed. In this example that takes
between 7 and 11 minutes. Despite the dataset being small, provisioning
hardware and loading the algorithm container take time upfront.

First, let’s specify our containers. Since we want this notebook to run
in all 4 of Amazon SageMaker’s regions, we’ll create a small lookup.
More details on algorithm containers can be found in `AWS
documentation <https://docs-aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html>`__.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    container = get_image_uri(boto3.Session().region_name, 'factorization-machines')

Next we’ll kick off the base estimator, making sure to pass in the
necessary hyperparameters. Notice: - ``feature_dim`` is set to 784,
which is the number of pixels in each 28 x 28 image. -
``predictor_type`` is set to ‘binary_classifier’ since we are trying to
predict whether the image is or is not a 0. - ``mini_batch_size`` is set
to 200. This value can be tuned for relatively minor improvements in fit
and speed, but selecting a reasonable value relative to the dataset is
appropriate in most cases. - ``num_factors`` is set to 10. As mentioned
initially, factorization machines find a lower dimensional
representation of the interactions for all features. Making this value
smaller provides a more parsimonious model, closer to a linear model,
but may sacrifice information about interactions. Making it larger
provides a higher-dimensional representation of feature interactions,
but adds computational complexity and can lead to overfitting. In a
practical application, time should be invested to tune this parameter to
the appropriate value.

.. code:: ipython3

    import boto3
    import sagemaker
    
    sess = sagemaker.Session()
    
    fm = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)
    fm.set_hyperparameters(feature_dim=784,
                          predictor_type='binary_classifier',
                          mini_batch_size=200,
                          num_factors=10)
    
    fm.fit({'train': s3_train_data})

Set up hosting for the model
----------------------------

Now that we’ve trained our model, we can deploy it behind an Amazon
SageMaker real-time hosted endpoint. This will allow out to make
predictions (or inference) from the model dyanamically.

*Note, Amazon SageMaker allows you the flexibility of importing models
trained elsewhere, as well as the choice of not importing models if the
target of model creation is AWS Lambda, AWS Greengrass, Amazon Redshift,
Amazon Athena, or other deployment target.*

.. code:: ipython3

    fm_predictor = fm.deploy(initial_instance_count=1,
                             instance_type='ml.m4.xlarge')

Validate the model for use
--------------------------

Finally, we can now validate the model for use. We can pass HTTP POST
requests to the endpoint to get back predictions. To make this easier,
we’ll again use the Amazon SageMaker Python SDK and specify how to
serialize requests and deserialize responses that are specific to the
algorithm.

Since factorization machines are so frequently used with sparse data,
making inference requests with a CSV format (as is done in other
algorithm examples) can be massively inefficient. Rather than waste
space and time generating all of those zeros, to pad the row to the
correct dimensionality, JSON can be used more efficiently. Since we
trained the model using dense data, this is a bit of a moot point, as
we’ll have to pass all the 0s in anyway.

Nevertheless, we’ll write our own small function to serialize our
inference request in the JSON format that Amazon SageMaker Factorization
Machines expects.

.. code:: ipython3

    import json
    from sagemaker.predictor import json_deserializer
    
    def fm_serializer(data):
        js = {'instances': []}
        for row in data:
            js['instances'].append({'features': row.tolist()})
        return json.dumps(js)
    
    fm_predictor.content_type = 'application/json'
    fm_predictor.serializer = fm_serializer
    fm_predictor.deserializer = json_deserializer

Now let’s try getting a prediction for a single record.

.. code:: ipython3

    result = fm_predictor.predict(train_set[0][30:31])
    print(result)

OK, a single prediction works. We see that for one record our endpoint
returned some JSON which contains ``predictions``, including the
``score`` and ``predicted_label``. In this case, ``score`` will be a
continuous value between [0, 1] representing the probability we think
the digit is a 0 or not. ``predicted_label`` will take a value of either
``0`` or ``1`` where (somewhat counterintuitively) ``1`` denotes that we
predict the image is a 0, while ``0`` denotes that we are predicting the
image is not of a 0.

Let’s do a whole batch of images and evaluate our predictive accuracy.

.. code:: ipython3

    import numpy as np
    
    predictions = []
    for array in np.array_split(test_set[0], 100):
        result = fm_predictor.predict(array)
        predictions += [r['predicted_label'] for r in result['predictions']]
    
    predictions = np.array(predictions)

.. code:: ipython3

    import pandas as pd
    
    pd.crosstab(np.where(test_set[1] == 0, 1, 0), predictions, rownames=['actuals'], colnames=['predictions'])

As we can see from the confusion matrix above, we predict 951 images of
the digit 0 correctly (confusingly this is class 1). Meanwhile we
predict 165 images as the digit 0 when in actuality they aren’t, and we
miss predicting 29 images of the digit 0 that we should have.

*Note: Due to some differences in parameter initialization, your results
may differ from those listed above, but should remain reasonably
consistent.*

(Optional) Delete the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you’re ready to be done with this notebook, please run the
delete_endpoint line in the cell below. This will remove the hosted
endpoint you created and avoid any charges from a stray instance being
left on.

.. code:: ipython3

    import sagemaker
    
    sagemaker.Session().delete_endpoint(fm_predictor.endpoint)
