Fairness Linear Learner in SageMaker
====================================

1. `Introduction <#Introduction>`__

   1. `Fairness definition <#fairness>`__

2. `Prerequisites and Data <#pre_and_data>`__

   1. `Initialize SageMaker <#initsagemaker>`__
   2. `Download data <#download_data>`__
   3. `Loading the data: Adult Dataset <#load_data>`__
   4. `Data inspection <#inspect_data>`__
   5. `Data encoding <#encode_data>`__
   6. `Data conversion and upload of the training data <#upload_data>`__

3. `Training the standard linear model <#train_linear_model>`__

   1. `Accuracy and Fairness of the model <#performance_linear_model>`__

4. `Changing the data to impose fairness <#impose_fairness>`__

   1. `Train the model with the fair data <#train_fair_model>`__
   2. `Accuracy and Fairness of the fair
      model <#performance_fair_model>`__
   3. `Sanity check: performance on the training
      set <#performance_fair_model_train>`__

5. `Distribution of the outputs <#distrib>`__

Introduction 
-------------

There have recently been concerns about bias in machine learning
algorithms as a result of mimicking existing human prejudices. Nowadays,
several Machine Learning methods have strong social implications, for
example they are used to predict bank loans, insurance rates or
advertising. Unfortunately, an algorithm that learns from historical
data will naturally inherit the past biases. In this notebook, we
present how to overcome this problem by using SageMaker and Fair
Algorithms in the context of Linear Learners.

We will start by introducing some of the concepts and math behind
fairness, then we will get ourselves setup to use these concepts in
SageMaker, download data, train a model, and finally apply our fairness
concepts to adjust our model predictions appropriately.

Fairness definition 
~~~~~~~~~~~~~~~~~~~~

We introduce here a classic measure of Fairness. Giving a dataset, we
can consider a binary sensitive feature (for example, gender between
male and female). We show here definition of Equal
Opportunity\ :math:`^{[1]}` among these two groups :math:`A` (female)
and :math:`B` (male) and a trained model :math:`f`:

.. math:: \mathbb{P}_{(x,y)} \big[ f(x)>0 \, \big| \, x \text{ in group } A, y = 1 \big] = \mathbb{P}_{(x,y)} \big[ f(x)>0 \, \big| \, x \text{ in group } B, y = 1 \big],

where :math:`\mathbb{P}_{(x,y)}` is the probability with respect to all
the possible examples :math:`(x,y)`.

Practically, we are imposing the same True Positive Rate (TPR) among the
two groups. Starting from this definition, we can estimate the
Difference in Equal Opportunity (DEO) of the model :math:`f` as:

.. math:: DEO(f) = \Big| \mathbb{P}_{(x,y)}\big[ f(x)>0 \, \big| \, x \text{ in group } A, y = 1 \big] -  \mathbb{P}_{(x,y)} \big[ f(x)>0 \, \big| \, x \text{ in group } B, y = 1 \big] \Big|.

A low value of this measure means a fair model :math:`f` with respect to
the sensitive feature.

Now, we introduce a method to empirically evalute Accuracy, TPR and DEO
from a list of predictions. The input of this method are: dataset, list
of predictions obtained by using a machine learning model and indices of
the exmaple in group :math:`A` and group :math:`B`.

:math:`^{[1]}`\ Moritz Hardt, Eric Price, and Nati Srebro. “`Equality of
opportunity in supervised
learning <http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf>`__”.
Advances in neural information processing systems (2016).

.. code:: ipython3

    from sklearn.metrics import confusion_matrix
    
    def deo_from_list(dataset, predictions, groupA_idxs, groupB_idxs):
        tnA, fpA, fnA, tpA  = confusion_matrix(np.where(dataset[groupA_idxs][:,-1] == 1, 1, 0),
                               predictions[groupA_idxs]).ravel()
        tnB, fpB, fnB, tpB  = confusion_matrix(np.where(dataset[groupB_idxs][:,-1] == 1, 1, 0),
                                               predictions[groupB_idxs]).ravel()
    
        print('Examples in group A: %d' % len(groupA_idxs))
        print('Examples in group B: %d' % len(groupB_idxs))
    
        print('TPR group A: %f' % (float(tpA) / (tpA + fnA)))
        print('TPR group B: %f' % (float(tpB) / (tpB + fnB)))
        print('TPR all dataset: %f' % (float(tpA + tpB) / (tpA + fnA + tpB + fnB)))
        print('Accuracy all dataset: %f' % (float(tpA + tnA + tpB + tnB) /
              (tpA + fpA + fnA + tnA + tpB + fpB + fnB + tnB)))
        
        return(np.abs(float(tpA) / (tpA + fnA) - float(tpB) / (tpB + fnB)))

It is usefull to have a method to evaluate Accuracy, TPR and DEO
directly from a model. In this case the method requries as input: our
trained model :math:`f`, dataset and index of the sensitive feature.

.. code:: ipython3

    def deo_from_model(model, dataset, sensitive_feature, groupA_value=None):
        if groupA_value == None:
            groupA_value = np.max(dataset[:, sensitive_feature])
            
        groupA_idxs = [idx for idx, val in enumerate(dataset)
                       if val[sensitive_feature] == groupA_value]
        groupB_idxs = [idx for idx, val in enumerate(dataset)
                       if val[sensitive_feature] != groupA_value]
        
        predictions = []
        for array in np.array_split(dataset[:,:-1], 100):
            result = model.predict(array)
            predictions += [r['predicted_label'] for r in result['predictions']]
        
        predictions = np.array(predictions)
        
        return deo_from_list(dataset, predictions, groupA_idxs, groupB_idxs)

Prerequisites and Data 
-----------------------

Initialize SageMaker 
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from sagemaker import Session
    bucket = Session().default_bucket() #'fairness-test2'
    prefix = 'sagemaker/DEMO-linear-adult'
    
    # Define IAM role
    from sagemaker import get_execution_role
    import pandas as pd
    import numpy as np
    import urllib
    import os
    import sklearn.preprocessing as preprocessing
    import seaborn as sns
    
    role = get_execution_role()

Download data 
~~~~~~~~~~~~~~

Data Source:
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

Let’s **download** the data and save it in the local folder with the
name adult.data and adult.test from UCI repository\ :math:`^{[2]}`.

:math:`^{[2]}`\ Dua Dheeru, and Efi Karra Taniskidou. “`UCI Machine
Learning Repository <http://archive.ics.uci.edu/ml>`__”. Irvine, CA:
University of California, School of Information and Computer Science
(2017).

.. code:: ipython3

    if not os.path.isfile('adult.data'):
        urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                                  'adult.data')
        print('adult.data saved!')
    else:
        print('adult.data already here.')
    
    if not os.path.isfile('adult.test'):
        urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                                  'adult.test')
        print('adult.test saved!')
    else:
        print('adult.test already here.')

Loading the data: Adult Dataset 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From UCI repository, this database contains 14 features concerning
demographic characteristics of :math:`45222` instances (:math:`32561`
for training and :math:`12661` for test). The task is to predict if a
person has an income per year that is more (or less) than
:math:`50000\,\$`.

Here the list of the features and their possible values: - (1) **Age**:
continuous. - (2) **Workclass**: Private, Self-emp-not-inc,
Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay,
Never-worked. - (3) **Fnlwgt**: continuous (the number of people the
census takers believe that observation represents). - (4) **Education**:
Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm,
Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate,
5th-6th, Preschool. - (5) **Education-num**: continuous. - (6)
**Marital-status**: Married-civ-spouse, Divorced, Never-married,
Separated, Widowed, Married-spouse-absent, Married-AF-spouse. - (7)
**Occupation**: Tech-support, Craft-repair, Other-service, Sales,
Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct,
Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
Protective-serv, Armed-Forces. - (8) **Relationship**: Wife, Own-child,
Husband, Not-in-family, Other-relative, Unmarried. - (9) **Ethnic
group**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. -
(10) **Gender**: Female, Male. - (11) **Capital-gain**: continuous. -
(12) **Capital-loss**: continuous. - (13) **Hours-per-week**:
continuous. - (14) **Native-country**: United-States, Cambodia, England,
Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan,
Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland,
Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic,
Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru,
Hong, Holand-Netherlands.

And finally our binary prediction task: - (15) **Target**: <=50, >50.

.. code:: ipython3

    %%time
    original_data = pd.read_csv(
        "adult.data",
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Ethnic group", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
    original_test = pd.read_csv(
        "adult.test",
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Ethnic group", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

Data inspection 
~~~~~~~~~~~~~~~~

Plotting histograms of the distribution of the different features is a
good way to visualize the data. We plot both the whole dataset
distributions (left) and the distrubtions in the case of positive labels
only (right).

.. code:: ipython3

    import matplotlib.pyplot as plt
    from math import ceil
    
    positive_idxs = [idx for idx, val in enumerate(original_data['Target']) if val == ">50K"]
    
    fig = plt.figure(figsize=(20,100))
    cols = 2
    rows = ceil(float(original_data.shape[1]) / cols) * 2
    for i, column in enumerate(original_data.columns):
        ax = fig.add_subplot(rows, cols, 2 * i + 1)
        ax.set_title(column)
        if original_data.dtypes[column] == np.object:
            original_data[column][:].value_counts(sort=True).plot(kind="bar", axes=ax)
        else:
            original_data[column][:].hist(axes=ax)
            plt.xticks(rotation="vertical")
            
        ax = fig.add_subplot(rows, cols, 2 * i + 2)
        ax.set_title(column + " (only positive examples)")
        if original_data.dtypes[column] == np.object:
            original_data[column][positive_idxs].value_counts(sort=True).plot(kind="bar", axes=ax)
        else:
            original_data[column][positive_idxs].hist(axes=ax)
            plt.xticks(rotation="vertical")
    
    plt.subplots_adjust(hspace=0.7, wspace=0.2)

Data encoding 
~~~~~~~~~~~~~~

We apply a preprocessing encoder for the categorial features.

.. code:: ipython3

    # Encode the categorical features as numbers
    def number_encode_features(df):
        result = df.copy()
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = preprocessing.LabelEncoder()
                #  print('Column:', column, result[column])
                result[column] = encoders[column].fit_transform(result[column].fillna('None'))
        return result, encoders
    
    # Calculate the correlation and plot it
    encoded_data, _ = number_encode_features(original_data)
    training_data_matrix = np.array(encoded_data.values, dtype=float)
    encoded_data, _ = number_encode_features(original_test)
    test_data_matrix = np.array(encoded_data.fillna(0).values, dtype=float)
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
    training_data_matrix = scaler.fit_transform(training_data_matrix)
    test_data_matrix = scaler.transform(test_data_matrix)

Data conversion and upload of the training data 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since algorithms have particular input and output requirements,
converting the dataset is also part of the process that a data scientist
goes through prior to initiating training. In this particular case, the
Amazon SageMaker implementation of Linear Learner takes recordIO-wrapped
protobuf, where the data we have today is a pickle-ized numpy array on
disk.

We also need to upload it to S3, so that Amazon SageMaker training can
use it.

.. code:: ipython3

    import io
    import numpy as np
    import sagemaker.amazon.common as smac
    import boto3
    import os
    
    vectors = np.array([t.tolist() for t in training_data_matrix[:,:-1]]).astype('float32')
    labels = np.where(np.array([t.tolist() for t in training_data_matrix[:,-1]]) == 1, 1, 0).astype('float32')
    
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, vectors, labels)
    buf.seek(0)
    
    key = 'recordio-pb-data'
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
    s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
    print('uploaded training data location: {}'.format(s3_train_data))

Let’s also setup an output S3 location for the model artifact that will
be output as the result of training with the algorithm.

.. code:: ipython3

    output_location = 's3://{}/{}/output'.format(bucket, prefix)
    print('training artifacts will be uploaded to: {}'.format(output_location))

Standard linear model 
----------------------

Once we have the data preprocessed and available in the correct format
for training, the next step is to actually train the model using the
data. More details on algorithm containers can be found in `AWS
documentation <https://docs-aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html>`__.

.. code:: ipython3

    from sagemaker.amazon.amazon_estimator import get_image_uri
    import sagemaker
    
    container = get_image_uri(boto3.Session().region_name, 'linear-learner', "latest")
    
    sess = sagemaker.Session()
    linear = sagemaker.estimator.Estimator(container,
                                           role, 
                                           train_instance_count=1, 
                                           train_instance_type='ml.c4.xlarge',
                                           output_path=output_location,
                                           sagemaker_session=sess)
    linear.set_hyperparameters(feature_dim=14,
                               predictor_type='binary_classifier',
                               mini_batch_size=200)
    
    linear.fit({'train': s3_train_data})

Accuracy and Fairness of the model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we’ve trained our model, we can deploy it behind an Amazon
SageMaker real-time hosted endpoint. This will allow out to make
predictions (or inference) from the model dyanamically.

.. code:: ipython3

    from sagemaker.predictor import csv_serializer, json_deserializer
    linear_predictor = linear.deploy(initial_instance_count=1,
                                     instance_type='ml.m4.xlarge')
    
    
    linear_predictor.content_type = 'text/csv'
    linear_predictor.serializer = csv_serializer
    linear_predictor.deserializer = json_deserializer

Prediction for the test data.

.. code:: ipython3

    import numpy as np
    
    predictions = []
    distance_from_hyperplane = []
    for array in np.array_split(test_data_matrix[:,:-1], 100):
        result = linear_predictor.predict(array)
        predictions += [r['predicted_label'] for r in result['predictions']]
        distance_from_hyperplane += [r['score'] for r in result['predictions']]
    
    predictions_test = np.array(predictions)
    distance_from_hyperplane_test = np.array(distance_from_hyperplane)    
    
    import pandas as pd
    
    pd.crosstab(np.where(test_data_matrix[:,-1] == 1, 1, 0), predictions_test,
                rownames=['actuals'], colnames=['predictions'])

We selcted as sensitive feature the gender, dividing the data between
“female” and “male”. Let’s check if our model is fair (in the sense of
Equal Opportunity). In the following the performance concerning
Accuracy, True Positive Rate (among the two groups and the whole
dataset) and DEO.

.. code:: ipython3

    sensitive_feature_gender = 9  # Gender
    groupA_value = 0.0 # Female
    
    deo = deo_from_model(linear_predictor, test_data_matrix,
                         sensitive_feature_gender, groupA_value=groupA_value)
    print('DEO: %f' % deo)

A DEO of about :math:`0.24` is quite high, this means that there is a
large gap between the True Positive Rate of the group A (female) and the
group B (male) and cosequently there is not Equal Opportunity.

In other words, we can say that for every :math:`100` correctly
predicted positive examples in one group, the disadvantage group will
have only :math:`100 - 24 = 76` of them. Consequently, it is clear that
the distribution of the error is unfair, and our goal now is to solve
this unfairness maintaining a comparable prediction accuracy.

Changing the data to impose fairness 
-------------------------------------

In the case of linear model, there is a simple but powerful way to
impose the fairness constraint\ :math:`^{[3]}`, i.e. :math:`DEO(f) = 0`.
Firstly, we have to introduce the following vector :math:`u`:

.. math::  u = \frac{1}{n(+, A)} \sum_{x \in X(+, A)} x - \frac{1}{n(+, B)} \sum_{x \in X(+, B)} x,

where :math:`X(+, A)` and :math:`X(+, B)` are the sets of positively
labeled examples in group :math:`A` and :math:`B` and :math:`n(+, A)`
and :math:`n(+, B)` their cardinalities. The vector :math:`u` represents
the unfair model, and our goal is to impose a serach space of the models
that is orthogonal to it. In the implementation we consider, without
loss of generality, that the binary sensitive feature is :math:`0` for
the gorup A and :math:`+1` for group B.

In our linear case, we can impose this orthogonal constraint,
i.e. impose the fairness, by applying a preprocessing of the original
data as following:

.. math::  \hat{x}_j = x_j - x_i \frac{u_j}{u_i} \,\,\,\, j \in \{1, \dots, i-1, i+1, \dots, d\}.

 Where :math:`i` is the index of the sensitive feature and
:math:`\hat{x}_j` is the new value for the feature :math:`j^{th}`. It is
important to note that - as consequence of this method - the new data
representation has a :math:`1` feature less compared the original one
(in our case the sensitive feature).

:math:`^{[3]}`\ Michele Donini, et al. “`Empirical Risk Minimization
under Fairness Constraints <https://arxiv.org/abs/1802.08626>`__.” arXiv
preprint arXiv:1802.08626 (2018).

.. code:: ipython3

    class UncorrelationMethod:
        def __init__(self, sensitive_feature, groupA_value=0.0):
            self.val0 = groupA_value
            self.sensitive_feature = sensitive_feature
            self.u = None
    
        def new_representation(self, examples):
            if self.u is None:
                print('You have to fit the model first...')
                return examples
            new_examples = np.array([ex if ex[self.sensitive_feature] == self.val0 # case x_i = 0, Group A
                                     else ex + self.u for ex in examples]) # case x_i = 1, Group B
            new_examples = np.delete(new_examples, self.sensitive_feature, 1)
            return new_examples
        
        def fit(self, dataset):
            tmp = [ex for idx, ex in enumerate(dataset)
                   if dataset[idx, -1] == 1 and ex[self.sensitive_feature] == self.val0]
            average_A_1 = np.mean(tmp, 0)
            n_A_1 = len(tmp)
            tmp = [ex for idx, ex in enumerate(dataset)
                   if dataset[idx, -1] == 1 and ex[self.sensitive_feature] != self.val0]
            average_not_A_1 = np.mean(tmp, 0)
            n_not_A_1 = len(tmp)
            N_1 = len([ex for idx, ex in enumerate(dataset) if dataset[idx, -1] == 1])
            self.u = average_A_1[:-1] - average_not_A_1[:-1]
            # Our hypothesis of values 0 (A) and +1 (B) for the sensitive feature among the two groups
            # has the following consequence:
            self.u[self.sensitive_feature] = -1.0

At this point we are ready to apply this algorithm to our data.

.. code:: ipython3

    uncorr_data = UncorrelationMethod(sensitive_feature_gender, 0.0)
    uncorr_data.fit(training_data_matrix) 
    new_training_data_matrix = np.hstack([uncorr_data.new_representation(training_data_matrix[:, :-1]),
                                         training_data_matrix[:, -1:-2:-1]])
    new_test_data_matrix = np.hstack([uncorr_data.new_representation(test_data_matrix[:, :-1]),
                                      test_data_matrix[:, -1:-2:-1]])

Train the model with the fair data 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we have simply to repeat the training by using this new dataset. It
is important to note that the new dataset has one feature less than the
original one (:math:`13` instead of :math:`14`).

.. code:: ipython3

    vectors = np.array([t.tolist() for t in new_training_data_matrix[:,:-1]]).astype('float32')
    labels = np.where(np.array([t.tolist() for t in new_training_data_matrix[:,-1]]) == 1, 1, 0).astype('float32')
    
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, vectors, labels)
    buf.seek(0)
    
    key = 'recordio-pb-data'
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
    s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
    print('uploaded training data location: {}'.format(s3_train_data))
    
    output_location = 's3://{}/{}/output'.format(bucket, prefix)
    print('training artifacts will be uploaded to: {}'.format(output_location))
    
    linearf = sagemaker.estimator.Estimator(container,
                                           role, 
                                           train_instance_count=1, 
                                           train_instance_type='ml.c4.xlarge',
                                           output_path=output_location,
                                           sagemaker_session=sess)
    linearf.set_hyperparameters(feature_dim=13,
                               predictor_type='binary_classifier',
                               mini_batch_size=200)
    
    path_fair_model = linearf.fit({'train': s3_train_data})
    
    linear_predictorf = linearf.deploy(initial_instance_count=1,
                                     instance_type='ml.m4.xlarge')
    linear_predictorf.content_type = 'text/csv'
    linear_predictorf.serializer = csv_serializer
    linear_predictorf.deserializer = json_deserializer

Now, we can calculate the predictions using our fair linear model.

.. code:: ipython3

    predictions = []
    distance_from_hyperplane = []
    for array in np.array_split(new_test_data_matrix[:,:-1], 100):
        result = linear_predictorf.predict(array)
        predictions += [r['predicted_label'] for r in result['predictions']]
        distance_from_hyperplane += [r['score'] for r in result['predictions']]
    
    distance_from_hyperplane_test_fair = np.array(distance_from_hyperplane)    
    predictions_test_fair = np.array(predictions)
    pd.crosstab(np.where(new_test_data_matrix[:,-1] == 1, 1, 0), predictions_test_fair,
                rownames=['actuals'], colnames=['predictions'])

Accuracy and Fairness of the fair model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s see the performance concerning accuracy and fairness for our new
model. We selcted as sensitive feature the Gender, dividing the data
between “female” and “male”.

.. code:: ipython3

    groupA_idxs = [idx for idx, val in enumerate(test_data_matrix)
                   if val[sensitive_feature_gender] == groupA_value]
    groupB_idxs = [idx for idx, val in enumerate(test_data_matrix)
                   if val[sensitive_feature_gender] != groupA_value]
    deo = deo_from_list(new_test_data_matrix, predictions_test_fair, groupA_idxs, groupB_idxs)
    print('DEO: %f' % deo)

The comparison with respect to the original model is the following: -
Concerning the accuracy the difference is not significative. In fact,
from an original :math:`0.8235` of accuracy, we obtain :math:`0.8086`
(decrease of about :math:`1.8 \%`). - Concerning the DEO, the original
model has a level of unfairness of :math:`0.2404` and our fair model of
:math:`0.0612`, with a decrese of more than :math:`75 \%`.

Sanity check: performance on the training set 
----------------------------------------------

Let’s see the performance of our method on the training set in order to
see if we apply the correct constraint and we do not overfit the data.

.. code:: ipython3

    predictions = []
    distance_from_hyperplane = []
    for array in np.array_split(new_training_data_matrix[:,:-1], 100):
        result = linear_predictorf.predict(array)
        predictions += [r['predicted_label'] for r in result['predictions']]
        distance_from_hyperplane += [r['score'] for r in result['predictions']]
    
    distance_from_hyperplane_train_fair = np.array(distance_from_hyperplane)    
    predictions_train_fair = np.array(predictions)
    pd.crosstab(np.where(new_training_data_matrix[:,-1] == 1, 1, 0),
                predictions_train_fair, rownames=['actuals'], colnames=['predictions'])
    
    groupA_idxs = [idx for idx, val in enumerate(training_data_matrix)
                   if val[sensitive_feature_gender] == groupA_value]
    groupB_idxs = [idx for idx, val in enumerate(training_data_matrix)
                   if val[sensitive_feature_gender] != groupA_value]
    
    deo = deo_from_list(new_training_data_matrix, predictions_train_fair, groupA_idxs, groupB_idxs)
    print('DEO: %f' % deo)

The value of the DEO is 0.0097. This confirm that our method is able to
implement the fairness constraint in the linear case.

Distribution of the outputs 
----------------------------

Now we plot the values of :math:`\langle w,x \rangle - b` for all the
examples :math:`x` with :math:`y=1`, where :math:`w` is the trained
model, for both the original model and our fair one. The value
:math:`\langle w,x \rangle - b` can be considered as the distance of
:math:`x` from the hyperplane that divides our feature space between
positive and negative examples. When this value is positive means that
our classifier predicts a positive label for the example :math:`x`.
Consequently, the area of the histogram is a visualization of the True
Positive Rate (the difference between the blue and orange areas is an
approximation of the DEO). For this reason, similar blue and orange
areas measn a more fair model (with respect to the senstive feature
“gender”).

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    sensitive_feature = sensitive_feature_gender
    
    SMALL_SIZE = 12
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16
    bins = 30
    
    X = training_data_matrix[:, :-1]
    y = training_data_matrix[:, -1]
    Xte = test_data_matrix[:, :-1]
    yte = test_data_matrix[:, -1]
    ypos = np.max(y)
    yneg = np.min(y)
    idx_group_A1 = [idx for idx, v in enumerate(Xte) 
                    if v[sensitive_feature] == groupA_value and yte[idx] == ypos]
    idx_group_B1 = [idx for idx, v in enumerate(Xte) 
                    if v[sensitive_feature] != groupA_value and yte[idx] == ypos]
    
    
    titles = ['Adult Dataset - TPR Area - Linear',
             'Adult Dataset - TPR Area - Fair Linear']
    for i,distance_from_hyperplane in enumerate([distance_from_hyperplane_test,
                                               distance_from_hyperplane_test_fair]):
        distance_from_hyperplane = distance_from_hyperplane - 0.5
        xmin = np.min([np.min(distance_from_hyperplane[idx_group_A1]),
                       np.min(distance_from_hyperplane[idx_group_B1])])
        xmax = np.max([np.max(distance_from_hyperplane[idx_group_A1]),
                       np.max(distance_from_hyperplane[idx_group_B1])])
        fig, ax = plt.subplots(figsize=(8, 6), dpi=90)
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        #plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
        pdf, bins, patches = ax.hist(distance_from_hyperplane[idx_group_A1], bins=bins,
                                     density=True, stacked=True, label='A=Female, Y=1', alpha=1.0)
        ax.hist(distance_from_hyperplane[idx_group_B1], bins=bins, density=True,
                stacked=True, label='B=Male, Y=1', alpha=0.5)
        ax.legend(loc='upper left')
        ax.set_xlim(left=0.0, right=xmax)
        ax.set_ylim(0, 3)
        plt.title(titles[i])
    
    plt.show()

(Optional) Delete the Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you’re ready to be done with this notebook, please run the
delete_endpoint line in the cell below. This will remove the hosted
endpoint you created and avoid any charges from a stray instance being
left on.

.. code:: ipython3

    import sagemaker
    
    sagemaker.Session().delete_endpoint(linear_predictor.endpoint)
    sagemaker.Session().delete_endpoint(linear_predictorf.endpoint)

Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"). You may
not use this file except in compliance with the License. A copy of the
License is located at http://aws.amazon.com/apache2.0/ or in the
"license" file accompanying this file. This file is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
