Highly Performant TensorFlow Batch Inference on Image Data Using the SageMaker CLI
==================================================================================

In this notebook, we’ll show how to use SageMaker batch transform to get
inferences on a large datasets. To do this, we’ll use a TensorFlow
Serving model to do batch inference on a large dataset of images. We’ll
show how to use the new pre-processing and post-processing feature of
the TensorFlow Serving container on Amazon SageMaker so that your
TensorFlow model can make inferences directly on data in S3, and save
post-processed inferences to S3.

The dataset we’ll be using is the `“Challenge
2018/2019” <https://github.com/cvdfoundation/open-images-dataset#download-the-open-images-challenge-28182019-test-set>`__\ ”
subset of the `Open Images V5
Dataset <https://storage.googleapis.com/openimages/web/index.html>`__.
This subset consists of 100,00 images in .jpg format, for a total of
10GB. For demonstration, the
`model <https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model>`__
we’ll be using is an image classification model based on the ResNet-50
architecture that has been trained on the ImageNet dataset, and which
has been exported as a TensorFlow SavedModel.

We will use this model to predict the class that each model belongs to.
We’ll write a pre- and post-processing script and package the script
with our TensorFlow SavedModel, and demonstrate how to get inferences on
large datasets with SageMaker batch transform quickly, efficiently, and
at scale, on GPU-accelerated instances.

Setup
-----

We’ll begin with some necessary imports, and get an Amazon SageMaker
session to help perform certain tasks, as well as an IAM role with the
necessary permissions.

.. code:: ipython3

    import numpy as np
    import os
    import sagemaker
    from sagemaker import get_execution_role
    
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
    
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    prefix = 'sagemaker/DEMO-tf-batch-inference-jpeg-images-python-sdk'
    uri_suffix = 'amazonaws.com'
    account_id = 520713654638
    account_id_cn = {
        'cn-north-1': 422961961927,
        'cn-northwest-1': 423003514399
    }
    if region in ['cn-north-1', 'cn-northwest-1']:
        uri_suffix = 'amazonaws.com.cn'
        account_id = account_id_cn[region]
    print('Region: {}'.format(region))
    print('S3 URI: s3://{}/{}'.format(bucket, prefix))
    print('Role:   {}'.format(role))

Inspecting the SavedModel
-------------------------

In order to make inferences, we’ll have to preprocess our image data in
S3 to match the serving signature of our TensorFlow SavedModel
(https://www.tensorflow.org/guide/saved_model), which we can inspect
using the saved_model_cli
(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/saved_model_cli.py).
This is the serving signature of the ResNet-50 v2 (NCHW, JPEG)
(https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model)
model:

.. code:: ipython3

    !aws s3 cp s3://sagemaker-sample-data-{region}/batch-transform/open-images/model/resnet_v2_fp32_savedmodel_NCHW_jpg.tar.gz .
    !tar -zxf resnet_v2_fp32_savedmodel_NCHW_jpg.tar.gz
    !saved_model_cli show --dir resnet_v2_fp32_savedmodel_NCHW_jpg/1538687370/ --all

The SageMaker TensorFlow Serving Container uses the model’s SignatureDef
named serving_default , which is declared when the TensorFlow SavedModel
is exported. This SignatureDef says that the model accepts a string of
arbitrary length as input, and responds with classes and their
probabilities. With our image classification model, the input string
will be a base-64 encoded string representing a JPEG image, which our
SavedModel will decode.

Writing a pre- and post-processing script
-----------------------------------------

We will package up our SavedModel with a Python script named
``inference.py``, which will pre-process input data going from S3 to our
TensorFlow Serving model, and post-process output data before it is
saved back to S3:

.. code:: ipython3

    !pygmentize code/inference.py

The input_handler intercepts inference requests, base-64 encodes the
request body, and formats the request body to conform to TensorFlow
Serving’s REST API (https://www.tensorflow.org/tfx/serving/api_rest).
The return value of the input_handler function is used as the request
body in the TensorFlow Serving request.

Binary data must use key “b64”, according to the TFS REST API
(https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values),
and since our serving signature’s input tensor has the suffix “\_bytes”,
the encoded image data under key “b64” will be passed to the
“image_bytes” tensor. Some serving signatures may accept a tensor of
floats or integers instead of a base-64 encoded string, but for binary
data (including image data), it is recommended that your SavedModel
accept a base-64 encoded string for binary data, since JSON
representations of binary data can be large.

Each incoming request originally contains a serialized JPEG image in its
request body, and after passing through the input_handler, the request
body contains the following, which our TensorFlow Serving accepts for
inference:

``{"instances": [{"b64":"[base-64 encoded JPEG image]"}]}``

The first field in the return value of ``output_handler`` is what
SageMaker Batch Transform will save to S3 as this example’s prediction.
In this case, our ``output_handler`` passes the content on to S3
unmodified.

Pre- and post-processing functions let you perform inference with
TensorFlow Serving on any data format, not just images. To learn more
about the ``input_handler`` and ``output_handler``, consult the
SageMaker TensorFlow Serving Container README
(https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/README.md).

Packaging a Model
-----------------

After writing a pre- and post-processing script, you’ll need to package
your TensorFlow SavedModel along with your script into a
``model.tar.gz`` file, which we’ll upload to S3 for the SageMaker
TensorFlow Serving Container to use. Let’s package the SavedModel with
the ``inference.py`` script and examine the expected format of the
``model.tar.gz`` file:

.. code:: ipython3

    !tar -cvzf model.tar.gz code --directory=resnet_v2_fp32_savedmodel_NCHW_jpg 1538687370

``1538687370`` refers to the model version number of the SavedModel, and
this directory contains our SavedModel artifacts. The code directory
contains our pre- and post-processing script, which must be named
``inference.py``. I can also include an optional ``requirements.txt``
file, which is used to install dependencies with ``pip`` from the Python
Package Index before the Transform Job starts, but we don’t need any
additional dependencies in this case, so we don’t include a requirements
file.

We will use this ``model.tar.gz`` when we create a SageMaker Model,
which we will use to run Transform Jobs. To learn more about packaging a
model, you can consult the SageMaker TensorFlow Serving Container
`README <https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/README.md>`__.

Run a Batch Transform job
-------------------------

Next, we’ll run a Batch Transform job using our data processing script
and GPU-based Amazon SageMaker Model. More specifically, we’ll perform
inference on a cluster of two instances, though we can choose more or
fewer. The objects in the S3 path will be distributed between the
instances.

Before we create a Transform Job, let’s inspect some of our input data.
Here’s an example, the first image in our dataset:

The data in the input path consists of 100,000 JPEG images of varying
sizes and shapes. Here is a subset:

.. code:: ipython3

    !aws s3 ls s3://sagemaker-sample-data-{region}/batch-transform/open-images/jpg/000 --human-readable

Creating a Model and Running a Transform Job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code below creates a SageMaker Model entity that will be used for
Batch inference, and runs a Transform Job using that Model. The Model
contains a reference to the TFS container, and the ``model.tar.gz``
containing our TensorFlow SavedModel and the pre- and post-processing
``inference.py`` script.

After we create a SageMaker Model, we use it to run batch predictions
using Batch Transform. We specify the input S3 data, content type of the
input data, the output S3 data, and instance type and count.

Performance
~~~~~~~~~~~

For improved performance, we specify two additional parameters
``max_concurrent_transforms`` and ``max_payload``, which control the
maximum number of parallel requests that can be sent to each instance in
a transform job at a time, and the maximum size of each request body.

When performing inference on entire S3 objects that cannot be split by
newline characters, such as images, it is recommended that you set
``max_payload`` to be slightly larger than the largest S3 object in your
dataset, and that you experiment with the ``max_concurrent_transforms``
parameter in powers of two to find a value that maximizes throughput for
your model. For example, we’ve set ``max_concurrent_transforms`` to 64
after experimenting with powers of two, and we set ``max_payload`` to 1,
since the largest object in our S3 input is less than one megabyte.

.. code:: bash

    %%bash -s "$bucket" "$prefix" "$role" "$region" "$uri_suffix" "$account_id"
    # For convenience, we pass in bucket, prefix, role, and region set in first Python set-up cell
    
    BUCKET=$1
    PREFIX=$2
    ROLE_ARN=$3
    REGION=$4
    URI_SUFFIX=$5
    ACCOUNT_ID=$6
    
    timestamp() {
      date +%Y-%m-%d-%H-%M-%S
    }
    
    # Creating the SageMaker Model: 
    MODEL_NAME="image-classification-tfs-$(timestamp)"
    MODEL_DATA_URL="s3://$BUCKET/$PREFIX/model/model.tar.gz"
    
    aws s3 cp model.tar.gz $MODEL_DATA_URL
    
    # This image is maintained at https://github.com/aws/sagemaker-tensorflow-serving-container
    TFS_VERSION="1.13"
    PROCESSOR_TYPE="gpu"
    IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.$URI_SUFFIX/sagemaker-tensorflow-serving:$TFS_VERSION-$PROCESSOR_TYPE"
    
    aws sagemaker create-model \
        --model-name $MODEL_NAME \
        --primary-container Image=$IMAGE,ModelDataUrl=$MODEL_DATA_URL \
        --execution-role-arn $ROLE_ARN
    
    # Creating the Transform Job:
    TRANSFORM_JOB_NAME="tfs-image-classification-job-$(timestamp)"
    
    # Specify where to get input data and where to get output data:
    TRANSFORM_S3_INPUT="s3://sagemaker-sample-data-$REGION/batch-transform/open-images/jpg"
    TRANSFORM_S3_OUTPUT="s3://$BUCKET/$PREFIX/output"
    
    # Our inference script validates the Content-Type, so we set it to "application/x-image".
    # SageMaker Model containers can use the content type field to transform multiple different data formats.
    # The Data Source tells Batch to get all objects under the S3 prefix.
    TRANSFORM_INPUT_DATA_SOURCE={S3DataSource={S3DataType="S3Prefix",S3Uri=$TRANSFORM_S3_INPUT}}
    CONTENT_TYPE="application/x-image"
    
    # Specify resources used to transform the job
    INSTANCE_TYPE="ml.p3.2xlarge"
    INSTANCE_COUNT=2
    
    # Performance parameters. MaxPayloadInMB specifies how large each request body can be.
    # Our images happen to be less than 1MB, so we set MaxPayloadInMB to 1MB.
    # MaxConcurrentTransforms configures the number of concurrent requests made to the container at once.
    # The ideal number depends on the payload size, instance type, and model, so some experimentation
    # may be beneficial.
    MAX_PAYLOAD_IN_MB=1
    MAX_CONCURRENT_TRANSFORMS=64
    
    aws sagemaker create-transform-job \
        --model-name $MODEL_NAME \
        --transform-input DataSource=$TRANSFORM_INPUT_DATA_SOURCE,ContentType=$CONTENT_TYPE \
        --transform-output S3OutputPath=$TRANSFORM_S3_OUTPUT \
        --transform-resources InstanceType=$INSTANCE_TYPE,InstanceCount=$INSTANCE_COUNT \
        --max-payload-in-mb $MAX_PAYLOAD_IN_MB \
        --max-concurrent-transforms $MAX_CONCURRENT_TRANSFORMS \
        --transform-job-name $TRANSFORM_JOB_NAME \
    
    
    echo "Model name: $MODEL_NAME"
    echo "Transform job name: $TRANSFORM_JOB_NAME"
    echo "Transform job input path: $TRANSFORM_S3_INPUT"
    echo "Transform job output path: $TRANSFORM_S3_OUTPUT"
    
    # Wait for the transform job to finish.
    aws sagemaker wait transform-job-completed-or-stopped \
      --transform-job-name $TRANSFORM_JOB_NAME
      
    # Examine the output.
    aws s3 ls $TRANSFORM_S3_OUTPUT/000 --human-readable
    
    # Copy an output example locally.
    aws s3 cp $TRANSFORM_S3_OUTPUT/00000b4dcff7f799.jpg.out .

We see that after our transform job finishes, we find one S3 object in
the output path for each object in the input path. This object contains
the inferences from our model for that object, and has the same name as
the corresponding input object, but with ``.out`` appended to it.

Inspecting one of the output objects, we find the prediction from our
TensorFlow Serving model. This is from the example image displayed
above:

.. code:: ipython3

    !cat 00000b4dcff7f799.jpg.out

Conclusion
----------

SageMaker batch transform can transform large datasets quickly and
scalably. We used the SageMaker TensorFlow Serving Container to
demonstrate how to quickly get inferences on a hundred thousand images
using GPU-accelerated instances.

The Amazon SageMaker TFS container supports CSV and JSON data out of the
box. The pre- and post-processing feature of the container lets you run
transform jobs on data of any format. The same container can be used for
real-time inference as well using an Amazon SageMaker hosted model
endpoint.
