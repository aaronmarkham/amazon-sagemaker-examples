Train faster, more flexible models with Amazon SageMaker Linear Learner
=======================================================================

Today Amazon SageMaker is launching several additional features to the
built-in linear learner algorithm. Amazon SageMaker algorithms are
designed to scale effortlessly to massive datasets and take advantage of
the latest hardware optimizations for unparalleled speed. The Amazon
SageMaker linear learner algorithm encompasses both linear regression
and binary classification algorithms. These algorithms are used
extensively in banking, fraud/risk management, insurance, and
healthcare. The new features of linear learner are designed to speed up
training and help you customize models for different use cases. Examples
include classification with unbalanced classes, where one of your
outcomes happens far less frequently than another. Or specialized loss
functions for regression, where it’s more important to penalize certain
model errors more than others.

In this blog post we’ll cover three things: 1. Early stopping and saving
the best model 1. New ways to customize linear learner models,
including: \* Hinge loss (support vector machines) \* Quantile loss \*
Huber loss \* Epsilon-insensitive loss \* Class weights options 1. Then
we’ll walk you through a hands-on example of using class weights to
boost performance in binary classification

Early Stopping
--------------

Linear learner trains models using Stochastic Gradient Descent (SGD) or
variants of SGD like Adam. Training requires multiple passes over the
data, called *epochs*, in which the data are loaded into memory in
chunks called *batches*, sometimes called *minibatches*. How do we know
how many epochs to run? Ideally, we’d like to continue training until
convergence - that is, until we no longer see any additional benefits.
Running additional epochs after the model has converged is a waste of
time and money, but guessing the right number of epochs is difficult to
do before submitting a training job. If we train for too few epochs, our
model will be less accurate than it should be, but if we train for too
many epochs, we’ll waste resources and potentially harm model accuracy
by overfitting. To remove the guesswork and optimize model training,
linear learner has added two new features: automatic early stopping and
saving the best model.

Early stopping works in two basic regimes: with or without a validation
set. Often we split our data into training, validation, and testing data
sets. Training is for optimizing the loss, validation is for tuning
hyperparameters, and testing is for producing an honest estimate of how
the model will perform on unseen data in the future. If you provide
linear learner with a validation data set, training will stop early when
validation loss stops improving. If no validation set is available,
training will stop early when training loss stops improving.

Early Stopping with a validation data set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One big benefit of having a validation data set is that we can tell if
and when we start overfitting to the training data. Overfitting is when
the model gives predictions that are too closely tailored to the
training data, so that generalization performance (performance on future
unseen data) will be poor. The following plot on the right shows a
typical progression during training with a validation data set. Until
epoch 5, the model has been learning from the training set and doing
better and better on the validation set. But in epochs 7-10, we see that
the model has begun to overfit on the training set, which shows up as
worse performance on the validation set. Regardless of whether the model
continues to improve (overfit) on the training data, we want to stop
training after the model starts to overfit. And we want to restore the
best model from just before the overfitting started. These two features
are now turned on by default in linear learner.

The default parameter values for early stopping are shown in the
following code. To tweak the behavior of early stopping, try changing
the values. To turn off early stopping entirely, choose a patience value
larger than the number of epochs you want to run.

::

   early_stopping_patience=3,
   early_stopping_tolerance=0.001,

The parameter early_stoping_patience defines how many epochs to wait
before ending training if no improvement is made. It’s useful to have a
little patience when deciding to stop early, since the training curve
can be bumpy. Performance may get worse for one or two epochs before
continuing to improve. By default, linear learner will stop early if
performance has degraded for three epochs in a row.

The parameter early_stopping_tolerance defines the size of an
improvement that’s considered significant. If the ratio of the
improvement in loss divided by the previous best loss is smaller than
this value, early stopping will consider the improvement to be zero.

Early stopping without a validation data set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When training with a training set only, we have no way to detect
overfitting. But we still want to stop training once the model has
converged and improvement has levelled off. In the left panel of the
following figure, that happens around epoch 25.



Early stopping and calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may already be familiar with the linear learner automated threshold
tuning for binary classification models. Threshold tuning and early
stopping work together seamlessly by default in linear learner.

When a binary classification model outputs a probability (e.g., logistic
regression) or a raw score (SVM), we convert that to a binary prediction
by applying a threshold, for example:

::

   predicted_label = 1 if raw_prediction > 0.5 else 0

We might want to tune the threshold (0.5 in the example) based on the
metric we care about most, such as accuracy or recall. Linear learner
does this tuning automatically using the
‘binary_classifier_model_selection_criteria’ parameter. When threshold
tuning and early stopping are both turned on (the default), then
training stops early based on the metric you request. For example, if
you provide a validation data set and request a logistic regression
model with threshold tuning based on accuracy, then training will stop
when the model with auto-thresholding reaches optimal performance on the
validation data. If there is no validation set and auto-thresholding is
turned off, then training will stop when the best value of the loss
function on the training data is reached.

New loss functions
------------------

The loss function is our definition of the cost of making an error in
prediction. When we train a model, we push the model weights in the
direction that minimizes loss, given the known labels in the training
set. The most common and well-known loss function is squared loss, which
is minimized when we train a standard linear regression model. Another
common loss function is the one used in logistic regression, variously
known as logistic loss, cross-entropy loss, or binomial likelihood.
Ideally, the loss function we train on should be a close match to the
business problem we’re trying to solve. Having the flexibility to choose
different loss functions at training time allows us to customize models
to different use cases. In this section, we’ll discuss when to use which
loss function, and introduce several new loss functions that have been
added to linear learner.



Squared loss
~~~~~~~~~~~~

::

   predictor_type='regressor',
   loss='squared_loss',

.. math:: \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^{N} (w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i)^2

We’ll use the following notation in all of the loss functions we
discuss:

:math:`w_0` is the bias that the model learns

:math:`\mathbf{w}` is the vector of feature weights that the model
learns

:math:`y_i` and :math:`\mathbf{x_i}` are the label and feature vector,
respectively, from example :math:`i` of the training data

:math:`N` is the total number of training examples

Squared loss is a first choice for most regression problems. It has the
nice property of producing an estimate of the mean of the label given
the features. As seen in the plot above, squared loss implies that we
pay a very high cost for very wrong predictions. This can cause problems
if our training data include some extreme outliers. A model trained on
squared loss will be very sensitive to outliers. Squared loss is
sometimes known as mean squared error (MSE), ordinary least squares
(OLS), or :math:`\text{L}_2` loss. Read more about `squared
loss <https://en.wikipedia.org/wiki/Least_squares>`__ on wikipedia.

Absolute loss
~~~~~~~~~~~~~

::

   predictor_type='regressor',
   loss='absolute_loss',

.. math:: \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^{N} |w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i|

| Absolute loss is less common than squared loss, but can be very
  useful. The main difference between the two is that training a model
  on absolute loss will produces estimates of the median of the label
  given the features. Squared loss estimates the mean, and absolute loss
  estimates the median. Whether you want to estimate the mean or median
  will depend on your use case. Let’s look at a few examples: \* If an
  error of -2 costs you $2 and an error of +50 costs you $50, then
  absolute loss models your costs better than squared loss.
| \* If an error of -2 costs you $2, while an error of +50 is simply
  unacceptably large, then it’s important that your errors are generally
  small, and so squared loss is probably the right fit.
| \* If it’s important that your predictions are too high as often as
  they’re too low, then you want to estimate the median with absolute
  loss.
| \* If outliers in your training data are having too much influence on
  the model, try switching from squared to absolute loss. Large errors
  get a large amount of attention from absolute loss, but with squared
  loss, large errors get squared and become huge errors attracting a
  huge amount of attention. If the error is due to an outlier, it might
  not deserve a huge amount of attention.

Absolute loss is sometimes also known as :math:`\text{L}_1` loss or
least absolute error. Read more about `absolute
loss <https://en.wikipedia.org/wiki/Least_absolute_deviations>`__ on
wikipedia.

Quantile loss
~~~~~~~~~~~~~

::

   predictor_type='regressor',
   loss='quantile_loss',
   quantile=0.9,

.. math::  \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N q(y_i - w_o - \mathbf{x_i}^\intercal \mathbf{w})^\text{+} + (1-q)(w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i)^\text{+} 

.. math::  \text{where the parameter } q \text{ is the quantile you want to predict}

Quantile loss lets us predict an upper or lower bound for the label,
given the features. To make predictions that are larger than the true
label 90% of the time, train quantile loss with the 0.9 quantile. An
example would be predicting electricity demand where we want to build
near peak demand since building to the average would result in
brown-outs and upset customers. Read more about `quantile
loss <https://en.wikipedia.org/wiki/Quantile_regression>`__ on
wikipedia.

Huber loss
~~~~~~~~~~

::

   predictor_type='regressor',
   loss='huber_loss',
   huber_delta=0.5,

.. math::  \text{Let the error be } e_i = w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i \text{.  Then Huber loss solves:}

.. math::  \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N I(|e_i| < \delta) \frac{e_i^2}{2} + I(|e_i| >= \delta) |e_i|\delta - \frac{\delta^2}{2} 

.. math::  \text{where } I(a) = 1 \text{ if } a \text{ is true, else } 0 

Huber loss is an interesting hybrid of :math:`\text{L}_1` and
:math:`\text{L}_2` losses. Huber loss counts small errors on a squared
scale and large errors on an absolute scale. In the plot above, we see
that Huber loss looks like squared loss when the error is near 0 and
absolute loss beyond that. Huber loss is useful when we want to train
with squared loss, but want to avoid squared loss’s sensitivity to
outliers. Huber loss gives less importance to outliers by not squaring
the larger errors. Read more about `Huber
loss <https://en.wikipedia.org/wiki/Huber_loss>`__ on wikipedia.

Epsilon-insensitive loss
~~~~~~~~~~~~~~~~~~~~~~~~

::

   predictor_type='regressor',
   loss='eps_insensitive_squared_loss',
   loss_insensitivity=0.25,

For epsilon-insensitive squared loss, we minimize

.. math::  \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N max(0, (w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i)^2 - \epsilon^2) 

And for epsilon-insensitive absolute loss, we minimize

.. math::  \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N max(0, |w_0 + \mathbf{x_i}^\intercal \mathbf{w} - y_i| - \epsilon) 

Epsilon-insensitive loss is useful when errors don’t matter to you as
long as they’re below some threshold. Set the threshold that makes sense
for your use case as epsilon. Epsilon-insensitive loss will allow the
model to pay no cost for making errors smaller than epsilon.

Logistic regression
~~~~~~~~~~~~~~~~~~~

::

   predictor_type='binary_classifier',
   loss='logistic',
   binary_classifier_model_selection_criteria='recall_at_target_precision',
   target_precision=0.9,

Each of the losses we’ve discussed is for regression problems, where the
labels are floating point numbers. The last two losses we’ll cover,
logistic regression and support vector machines, are for binary
classification problems where the labels are one of two classes. Linear
learner expects the class labels to be 0 or 1. This may require some
preprocessing, for example if your labels are coded as -1 and +1, or as
blue and yellow. Logistic regression produces a predicted probability
for each data point:

.. math::  p_i = \sigma(w_0 + \mathbf{x_i}^\intercal \mathbf{w}) 

The loss function minimized in training a logistic regression model is
the log likelihood of a binomial distribution. It assigns the highest
cost to predictions that are confident and wrong, for example a
prediction of 0.99 when the true label was 0, or a prediction of 0.002
when the true label was positive. The loss function is:

.. math::  \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N y_i  \text{log}(p) - (1 - y_i) \text{log}(1 - p) 

.. math::  \text{where } \sigma(x) = \frac{\text{exp}(x)}{1 + \text{exp}(x)}  

Read more about `logistic
regression <https://en.wikipedia.org/wiki/Logistic_regression>`__ on
wikipedia.

Hinge loss (support vector machine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   predictor_type='binary_classifier',
   loss='hinge_loss',
   margin=1.0,
   binary_classifier_model_selection_criteria='recall_at_target_precision',
   target_precision=0.9,

Another popular option for binary classification problems is the hinge
loss, also known as a Support Vector Machine (SVM) or Support Vector
Classifier (SVC) with a linear kernel. It places a high cost on any
points that are misclassified or nearly misclassified. To tune the
meaning of “nearly”, adjust the margin parameter:

It’s difficult to say in advance whether logistic regression or SVM will
be the right model for a binary classification problem, though logistic
regression is generally a more popular choice then SVM. If it’s
important to provide probabilities of the predicted class labels, then
logistic regression will be the right choice. If all that matters is
better accuracy, precision, or recall, then either model may be
appropriate. One advantage of logistic regression is that it produces
the probability of an example having a positive label. That can be
useful, for example in an ad serving system where the predicted click
probability is used as an input to a bidding mechanism. Hinge loss does
not produce class probabilities.

Whichever model you choose, you’re likely to benefit from linear
learner’s options for tuning the threshold that separates positive from
negative predictions

.. math:: \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^{N} y_i(\frac{m+1}{2} - w_0 - \mathbf{x_i}^\text{T}\mathbf{w})^\text{+} + (1-y_i)\frac{m-1}{2} + w_o + \mathbf{x_i}^\text{T}\mathbf{w})^\text{+}

.. math:: \text{where  } a^\text{+} = \text{max}(0, a)

Note that the hinge loss we use is a reparameterization of the usual
hinge loss: typically hinge loss expects the binary label to be in {-1,
1}, whereas ours expects the binary labels to be in {0, 1}. This
reparameterization allows LinearLearner to accept the same data format
for binary classification regardless of the training loss. Read more
about `hinge loss <https://en.wikipedia.org/wiki/Hinge_loss>`__ on
wikipedia.

Class weights
-------------

In some binary classification problems, we may find that our training
data is highly unbalanced. For example, in credit card fraud detection,
we’re likely to have many more examples of non-fraudulent transactions
than fraudulent. In these cases, balancing the class weights may improve
model performance.

Suppose we have 98% negative and 2% positive examples. To balance the
total weight of each class, we can set the positive class weight to be
49. Now the average weight from the positive class is 0.98 :math:`\cdot`
1 = 0.98, and the average weight from the negative class is 0.02
:math:`\cdot` 49 = 0.98. The negative class weight multiplier is always
1.

To incorporate the positive class weight in training, we multiply the
loss by the positive weight whenever we see a positive class label. For
logistic regression, the weighted loss is:

Weighted logistic regression:

.. math::  \text{argmin}_{w_0, \mathbf{w}} \sum_{i=1}^N p y_i  \text{log}(\sigma(w_0 + \mathbf{x_i}^\intercal \mathbf{w})) - (1 - y_i) \text{log}(1 - \sigma(w_0 + \mathbf{x_i}^\intercal \mathbf{w})) 

.. math::  \text{where } p \text{ is the weight for the positive class.} 

The only difference between the weighted and unweighted logistic
regression loss functions is the presense of the class weight, :math:`p`
on the left-hand term in the loss. Class weights in the hinge loss (SVM)
classifier are applied in the same way.

To apply class weights when training a model with linear learner, supply
the weight for the positive class as a training parameter:

::

   positive_example_weight_mult=200,

Or to ask linear learner to calculate the positive class weight for you:

::

   positive_example_weight_mult='balanced',

Hands-on example: Detecting credit card fraud
---------------------------------------------

In this section, we’ll look at a credit card fraud detection dataset.
The data set (Dal Pozzolo et al. 2015) was downloaded from
`Kaggle <https://www.kaggle.com/mlg-ulb/creditcardfraud/data>`__. We
have features and labels for over a quarter million credit card
transactions, each of which is labeled as fraudulent or not fraudulent.
We’d like to train a model based on the features of these transactions
so that we can predict risky or fraudulent transactions in the future.
This is a binary classification problem.

We’ll walk through training linear learner with various settings and
deploying an inference endpoint. We’ll evaluate the quality of our
models by hitting that endpoint with observations from the test set. We
can take the real-time predictions returned by the endpoint and evaluate
them against the ground-truth labels in our test set.

Next, we’ll apply the linear learner threshold tuning functionality to
get better precision without sacrificing recall. Then, we’ll push the
precision even higher using the linear learner new class weights
feature. Because fraud can be extremely costly, we would prefer to have
high recall, even if this means more false positives. This is especially
true if we are building a first line of defense, flagging potentially
fraudulent transactions for further review before taking actions that
affect customers.

First we’ll do some preprocessing on this data set: we’ll shuffle the
examples and split them into train and test sets. To run this under
notebook under your own AWS account, you’ll need to change the Amazon S3
locations. First download the raw data from
`Kaggle <https://www.kaggle.com/mlg-ulb/creditcardfraud/data>`__ and
upload to your SageMaker notebook instance (or wherever you’re running
this notebook). Only 0.17% of the data have positive labels, making this
a challenging classification problem.

.. code:: ipython3

    import boto3
    import io
    import matplotlib.pyplot as plt
    import numpy as np 
    import os
    import pandas as pd 
    
    import sagemaker
    import sagemaker.amazon.common as smac
    from sagemaker import get_execution_role
    from sagemaker.predictor import csv_serializer, json_deserializer

.. code:: ipython3

    # Set data locations
    bucket = '<your_s3_bucket_here>'                        # replace this with your own bucket
    prefix = 'sagemaker/DEMO-linear-learner-loss-weights'   # replace this with your own prefix
    s3_train_key = '{}/train/recordio-pb-data'.format(prefix)
    s3_train_path = os.path.join('s3://', bucket, s3_train_key)
    local_raw_data = 'creditcard.csv.zip'
    role = get_execution_role()

.. code:: ipython3

    # Confirm access to s3 bucket
    for obj in boto3.resource('s3').Bucket(bucket).objects.all():
        print(obj.key)

.. code:: ipython3

    # Read the data, shuffle, and split into train and test sets, separating the labels (last column) from the features
    raw_data = pd.read_csv(local_raw_data).as_matrix()
    np.random.seed(0)
    np.random.shuffle(raw_data)
    train_size = int(raw_data.shape[0] * 0.7)
    train_features  = raw_data[:train_size, :-1]
    train_labels = raw_data[:train_size, -1]
    test_features = raw_data[train_size:, :-1]
    test_labels = raw_data[train_size:, -1]
    
    # Convert the processed training data to protobuf and write to S3 for linear learner
    vectors = np.array([t.tolist() for t in train_features]).astype('float32')
    labels = np.array([t.tolist() for t in train_labels]).astype('float32')
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, vectors, labels)
    buf.seek(0)
    boto3.resource('s3').Bucket(bucket).Object(s3_train_key).upload_fileobj(buf)

We’ll wrap the model training setup in a convenience function that takes
in the S3 location of the training data, the model hyperparameters that
define our training job, and the S3 output path for model artifacts.
Inside the function, we’ll hardcode the algorithm container, the number
and type of EC2 instances to train on, and the input and output data
formats.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    
    def predictor_from_hyperparams(s3_train_data, hyperparams, output_path):
        """
        Create an Estimator from the given hyperparams, fit to training data, and return a deployed predictor
        """
        # specify algorithm containers and instantiate an Estimator with given hyperparams
        container = get_image_uri(boto3.Session().region_name, 'linear-learner')
    
        linear = sagemaker.estimator.Estimator(container,
            role,
            train_instance_count=1,
            train_instance_type='ml.m4.xlarge',
            output_path=output_path,
            sagemaker_session=sagemaker.Session())
        linear.set_hyperparameters(**hyperparams)
        # train model
        linear.fit({'train': s3_train_data})
        # deploy a predictor
        linear_predictor = linear.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
        linear_predictor.content_type = 'text/csv'
        linear_predictor.serializer = csv_serializer
        linear_predictor.deserializer = json_deserializer
        return linear_predictor

And add another convenience function for setting up a hosting endpoint,
making predictions, and evaluating the model. To make predictions, we
need to set up a model hosting endpoint. Then we feed test features to
the endpoint and receive predicted test labels. To evaluate the models
we create in this exercise, we’ll capture predicted test labels and
compare them to actuals using some common binary classification metrics.

.. code:: ipython3

    def evaluate(linear_predictor, test_features, test_labels, model_name, verbose=True):
        """
        Evaluate a model on a test set given the prediction endpoint.  Return binary classification metrics.
        """
        # split the test data set into 100 batches and evaluate using prediction endpoint
        prediction_batches = [linear_predictor.predict(batch)['predictions'] for batch in np.array_split(test_features, 100)]
        # parse raw predictions json to exctract predicted label
        test_preds = np.concatenate([np.array([x['predicted_label'] for x in batch]) for batch in prediction_batches])
        
        # calculate true positives, false positives, true negatives, false negatives
        tp = np.logical_and(test_labels, test_preds).sum()
        fp = np.logical_and(1-test_labels, test_preds).sum()
        tn = np.logical_and(1-test_labels, 1-test_preds).sum()
        fn = np.logical_and(test_labels, 1-test_preds).sum()
        
        # calculate binary classification metrics
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        f1 = 2 * precision * recall / (precision + recall)
        
        if verbose:
            print(pd.crosstab(test_labels, test_preds, rownames=['actuals'], colnames=['predictions']))
            print("\n{:<11} {:.3f}".format('Recall:', recall))
            print("{:<11} {:.3f}".format('Precision:', precision))
            print("{:<11} {:.3f}".format('Accuracy:', accuracy))
            print("{:<11} {:.3f}".format('F1:', f1))
            
        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy, 
                 'F1': f1, 'Model': model_name}

And finally we’ll add a convenience function to delete prediction
endpoints after we’re done with them:

.. code:: ipython3

    def delete_endpoint(predictor):
            try:
                boto3.client('sagemaker').delete_endpoint(EndpointName=predictor.endpoint)
                print('Deleted {}'.format(predictor.endpoint))
            except:
                print('Already deleted: {}'.format(predictor.endpoint))

Let’s begin by training a binary classifier model with the linear
learner default settings. Note that we’re setting the number of epochs
to 40, which is much higher than the default of 10 epochs. With early
stopping, we don’t have to worry about setting the number of epochs too
high. Linear learner will stop training automatically after the model
has converged.

.. code:: ipython3

    # Training a binary classifier with default settings: logistic regression
    defaults_hyperparams = {
        'feature_dim': 30,
        'predictor_type': 'binary_classifier',
        'epochs': 40
    }
    defaults_output_path = 's3://{}/{}/defaults/output'.format(bucket, prefix)
    defaults_predictor = predictor_from_hyperparams(s3_train_path, defaults_hyperparams, defaults_output_path)

And now we’ll produce a model with a threshold tuned for the best
possible precision with recall fixed at 90%:

.. code:: ipython3

    # Training a binary classifier with automated threshold tuning
    autothresh_hyperparams = {
        'feature_dim': 30,
        'predictor_type': 'binary_classifier',
        'binary_classifier_model_selection_criteria': 'precision_at_target_recall', 
        'target_recall': 0.9,
        'epochs': 40
    }
    autothresh_output_path = 's3://{}/{}/autothresh/output'.format(bucket, prefix)
    autothresh_predictor = predictor_from_hyperparams(s3_train_path, autothresh_hyperparams, autothresh_output_path)


Improving recall with class weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we’ll improve on these results using a new feature added to linear
learner: class weights for binary classification. We introduced this
feature in the *Class Weights* section, and now we’ll look into its
application to the credit card fraud dataset by training a new model
with balanced class weights:

.. code:: ipython3

    # Training a binary classifier with class weights and automated threshold tuning
    class_weights_hyperparams = {
        'feature_dim': 30,
        'predictor_type': 'binary_classifier',
        'binary_classifier_model_selection_criteria': 'precision_at_target_recall', 
        'target_recall': 0.9,
        'positive_example_weight_mult': 'balanced',
        'epochs': 40
    }
    class_weights_output_path = 's3://{}/{}/class_weights/output'.format(bucket, prefix)
    class_weights_predictor = predictor_from_hyperparams(s3_train_path, class_weights_hyperparams, class_weights_output_path)


The first training examples used the default loss function for binary
classification, logistic loss. Now let’s train a model with hinge loss.
This is also called a support vector machine (SVM) classifier with a
linear kernel. Threshold tuning is supported for all binary classifier
models in linear learner.

.. code:: ipython3

    # Training a binary classifier with hinge loss and automated threshold tuning
    svm_hyperparams = {
        'feature_dim': 30,
        'predictor_type': 'binary_classifier',
        'loss': 'hinge_loss',
        'binary_classifier_model_selection_criteria': 'precision_at_target_recall', 
        'target_recall': 0.9,
        'epochs': 40
    }
    svm_output_path = 's3://{}/{}/svm/output'.format(bucket, prefix)
    svm_predictor = predictor_from_hyperparams(s3_train_path, svm_hyperparams, svm_output_path)

And finally, let’s see what happens with balancing the class weights for
the SVM model:

.. code:: ipython3

    # Training a binary classifier with hinge loss, balanced class weights, and automated threshold tuning
    svm_balanced_hyperparams = {
        'feature_dim': 30,
        'predictor_type': 'binary_classifier',
        'loss': 'hinge_loss',
        'binary_classifier_model_selection_criteria': 'precision_at_target_recall', 
        'target_recall': 0.9,
        'positive_example_weight_mult': 'balanced',
        'epochs': 40
    }
    svm_balanced_output_path = 's3://{}/{}/svm_balanced/output'.format(bucket, prefix)
    svm_balanced_predictor = predictor_from_hyperparams(s3_train_path, svm_balanced_hyperparams, svm_balanced_output_path)

Now we’ll make use of the prediction endpoint we’ve set up for each
model by sending them features from the test set and evaluating their
predictions with standard binary classification metrics.

.. code:: ipython3

    # Evaluate the trained models
    predictors = {'Logistic': defaults_predictor, 'Logistic with auto threshold': autothresh_predictor, 
                  'Logistic with class weights': class_weights_predictor, 'Hinge with auto threshold': svm_predictor, 
                  'Hinge with class weights': svm_balanced_predictor}
    metrics = {key: evaluate(predictor, test_features, test_labels, key, False) for key, predictor in predictors.items()}
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    display(pd.DataFrame(list(metrics.values())).loc[:, ['Model', 'Recall', 'Precision', 'Accuracy', 'F1']])

The results are in! With threshold tuning, we can accurately predict
85-90% of the fraudulent transactions in the test set (due to randomness
in training, recall will vary between 0.85-0.9 across multiple runs).
But in addition to those true positives, we’ll have a high number of
false positives: 90-95% of the transactions we predict to be fraudulent
are in fact not fraudulent (precision varies between 0.05-0.1). This
model would work well as a first line of defense, flagging potentially
fraudulent transactions for further review. If we instead want a model
that gives very few false alarms, at the cost of catching far fewer of
the fraudulent transactions, then we should optimize for higher
precision:

::

   binary_classifier_model_selection_criteria='recall_at_target_precision', 
   target_precision=0.9,

And what about the results of using our new feature, class weights for
binary classification? Training with class weights has made a huge
improvement to this model’s performance! The precision is roughly
doubled, while recall is still held constant at 85-90%.

Balancing class weights improved the performance of our SVM predictor,
but it still does not match the corresponding logistic regression model
for this dataset. Comparing all of the models we’ve fit so far, logistic
regression with class weights and tuned thresholds did the best.

Note on target vs. observed recall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It’s worth taking some time to look more closely at these results. If we
asked linear learner for a model calibrated to a target recall of 0.9,
then why didn’t we get exactly 90% recall on the test set? The reason is
the difference between training, validation, and testing. Linear learner
calibrates thresholds for binary classification on the validation data
set when one is provided, or else on the training set. Since we did not
provide a validation data set, the threshold were calculated on the
training data. Since the training, validation, and test data sets don’t
match exactly, the target recall we request is only an approximation. In
this case, the threshold that produced 90% recall on the training data
happened to produce only 85-90% recall on the test data (due to some
randomness in training, the results will vary from one run to the next).
The variation of recall in the test set versus the training set is
dependent on the number of positive points. In this example, although we
have over 280,000 examples in the entire dataset, we only have 337
positive examples, hence the large difference. The accuracy of this
approximation can be improved by providing a large validation data set
to get a more accurate threshold, and then evaluating on a large test
set to get a more accurate benchmark of the model and its threshold. For
even more fine-grained control, we can set the number of calibration
samples to a higher number. It’s default value is already quite high at
10 million samples:

::

   num_calibration_samples=10000000,

Clean Up
^^^^^^^^

Finally we’ll clean up by deleting the prediction endpoints we set up:

.. code:: ipython3

    for predictor in [defaults_predictor, autothresh_predictor, class_weights_predictor, 
                      svm_predictor, svm_balanced_predictor]:
        delete_endpoint(predictor)

We’ve just shown how to use the linear learner new early stopping
feature, new loss functions, and new class weights feature to improve
credit card fraud prediction. Class weights can help you optimize recall
or precision for all types of fraud detection, as well as other
classification problems with rare events, like ad click prediction or
mechanical failure prediction. Try using class weights in your binary
classification problem, or try one of the new loss functions for your
regression problems: use quantile prediction to put confidence intervals
around your predictions by learning 5% and 95% quantiles. For more
information about new loss functions and class weights, see the linear
learner
`documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html>`__.

References
''''''''''

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
Bontempi. Calibrating Probability with Undersampling for Unbalanced
Classification. In Symposium on Computational Intelligence and Data
Mining (CIDM), IEEE, 2015. See link to full license text on
`Kaggle <https://www.kaggle.com/mlg-ulb/creditcardfraud>`__.

