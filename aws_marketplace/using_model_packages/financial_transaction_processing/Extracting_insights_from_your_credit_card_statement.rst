Extracting insights from your credit card statements
====================================================

Manually identifying additional tax deductibles from your family’s
credit card statements is quite tedious as the data in the statement is
often unstructured. In this notebook, you will learn how to classify,
categorize, and convert your transaction data from unstructured to
structured format. You will learn how to use a pre-trained machine
learning model to identify candidate merchant names and locations and
make the process of matching candidates against a dictionary, efficient.
Once transaction data is converted into a structured format, you will be
able to use it for multiple purposes such as: 1. To identify precise
amounts you paid for items that are tax deductible. 2. If you have
enough data, to train a model that learns your expenditure patterns and
identifies fraudulent activity. 3. To understand your expenditure
patterns.

Overview:
^^^^^^^^^

In `Step 1 <#Step-1:-Perform-preliminary-analysis-on-the-dataset>`__ of
this notebook, you will load a sample transaction file into a data-frame
and perform preliminary analysis such as identifying the transaction
date and extracting subscription fees from your statement. Other
interesting features in the transaction log are city name and merchant
name, which are often multi-term entities. Identifying merchant name as
well as city requires huge lookups in the dictionary as considerable
number of permutations and combinations need to be searched if you take
brute force approach.

Let us consider following transaction log entry.

::

   3/25/15,6.43,PURCHASE AUTHORIZED ON 03/23 XXXX CONCH TOUR TRAIN XXX XXXXX 1034 G KEY WEST FL XXXXXXXX

The transaction entry contains a multi-termed merchant name entity
(conch tour train) and a multi-termed city (KEY WEST) that you want to
extract. To do so confidently, you have to identify all unigrams,
bi-grams, tri-grams from the transaction log and match them against a
dictionary. If you ignore the anonymized data and all other words
occurring before the date 03/23, you get 8 unique words [CONCH TOUR
TRAIN 1034 G KEY WEST FL] leading to following 21 unique ngrams(For this
use-case, we will stick to unigram/bigrams/trigrams).

::

   {'TRAIN 1034', 'CONCH', 'TOUR', '1034 G', 'G KEY WEST', 'CONCH TOUR TRAIN', 'G KEY', 'KEY WEST', 'CONCH TOUR', 'WEST', 'FL', 'KEY WEST FL', 'KEY', '1034 G KEY', 'TOUR TRAIN 1034', 'TRAIN', 'TOUR TRAIN', '1034', 'TRAIN 1034 G', 'G', 'WEST FL'}

To identify the merchant name from this transaction log entry, you would
need to do a maximum of **21 X (size of merchant info dictionary)**
lookups. Similarly, to identify the city name from this transaction log
entry, you would need to do a maximum of **21 X (size of city name
dictionary)** lookups. Given that each transaction log entry will have a
variable number of words, the merchant/location computation becomes a
computationally expensive task.

You can reduce the time required for doing such lookups with an ML model
that identifies potential candidate merchant/city names. E.g. the
`Transaction Data Parsing
(NER) <https://aws.amazon.com/marketplace/pp/prodview-sqnwjvzzqntn2>`__
ML model returned the following output for “PURCHASE AUTHORIZED ON 03/23
CONCH TOUR TRAIN 1034 G KEY WEST FL” as input.

::

   [{'key': 'CONCH', 'type': 'NE_MERCHANT', 'start_pos': 30, 'end_pos': 35},
    {'key': 'TOUR', 'type': 'NE_MERCHANT', 'start_pos': 36, 'end_pos': 40},
    {'key': 'TRAIN', 'type': 'NE_MERCHANT', 'start_pos': 41, 'end_pos': 46},
    {'key': 'KEY', 'type': 'NE_STORE_LOCATION', 'start_pos': 55, 'end_pos': 58},
    {'key': 'WEST', 'type': 'NE_STORE_LOCATION', 'start_pos': 59, 'end_pos': 63},
    {'key': 'FL', 'type': 'NE_STORE_LOCATION', 'start_pos': 64, 'end_pos': 66}]

Given that we now have 3 candidate words indicating merchant name, the
name-space (a total of 6 unique unigrams/bigrams/trigrams) for doing
dictionary lookups is much smaller.

In `Step
2 <#Step-2:-Use-an-ML-model-to-identify-potential-merchants-and-locations-for-each-transaction>`__,
you will perform a prediction on an ML Model to identify candidate
merchant and location names from each transaction log entry. In `Step
3 <#Step-3:-Identify-merchant-name-from-transaction-log>`__, you will
identify a precise merchant name by doing a lookup on candidate merchant
names and in `Step
4 <#Step-4:-Identify-state-and-city-for-each-transaction-log-entry>`__,
you will identify city and state information by doing lookups on
candidate city names. Finally your will do cleanup in `Step
5 <#Step-5:-Next-steps-and-cleanup>`__.

Contents:
^^^^^^^^^

-  `Pre-requisites <#Pre-requisites>`__
-  `Step 1: Perform preliminary analysis and data extraction on the
   dataset <#Step-1:-Perform-preliminary-analysis-on-the-dataset>`__

   -  `Step 1.1: Load and View the
      dataset <#Step-1.1-Load-and-View-the-dataset>`__
   -  `Step 1.2 Identify Transaction
      date <#Step-1.2-Identify-Transaction-date>`__
   -  `Step 1.3 Identify
      subscriptions <#Step-1.3-Identify-subscriptions>`__

-  `Step 2: Use an ML model to identify potential merchants and
   locations for each
   transaction <#Step-2:-Use-an-ML-model-to-identify-potential-merchants-and-locations-for-each-transaction>`__

   -  `Step 2.1: Deploy the model <#Step-2.1:-Deploy-the-model>`__
   -  `Step 2.2: Populate potential merchants and locations in
      dataframe <#Step-2.2:-Populate-candidate-merchants-and-locations-in-dataframe>`__

-  `Step 3: Identify merchant name from transaction
   log <#Step-3:-Identify-merchant-name-from-transaction-log>`__

   -  `Step 3.1: Identify merchant
      name <#Step-3.1:-Identify-merchant-name>`__
   -  `Step 3.2: Visualize expenses <#Step-3.2:-Visualize-expenses>`__

-  `Step 4: Identify state and city for each transaction log
   entry <#Step-4:-Identify-state-and-city-for-each-transaction-log-entry>`__

   -  `Step 4.1: Populate state in which transaction took
      place <#Step-4.1:-Populate-state-in-which-transaction-took-place>`__
   -  `Step 4.2: Populate city and
      country <#Step-4.2:-Populate-city-and-country>`__

-  `Step 5: Next steps and cleanup <#Step-5:-Next-steps-and-cleanup>`__

Usage instructions
^^^^^^^^^^^^^^^^^^

You can run this notebook one cell at a time (By using Shift+Enter for
running a cell).

**Pre-requisites**

This sample notebook requires subscription to `Transaction Data Parsing
(NER) <https://aws.amazon.com/marketplace/pp/prodview-sqnwjvzzqntn2>`__,
a pre-trained machine learning model package from AWS Marketplace. If
your AWS account has not been subscribed to this listing, here is the
process you can follow: 1. Open the
`listing <https://aws.amazon.com/marketplace/pp/prodview-sqnwjvzzqntn2>`__
from AWS Marketplace 1. Read the **Highlights** section and then
**product overview** section of the listing. 1. View **usage
information** and then **additional resources.** 1. Note the supported
instance types. 1. Next, click on **Continue to subscribe.** 1. Review
**End user license agreement, support terms**, as well as **pricing
information.** 1. **“Accept Offer”** button needs to be clicked if your
organization agrees with EULA, pricing information as well as support
terms. If **Continue to configuration** button is active, it means your
account already has a subscription to this listing. Once you click on
**Continue to configuration** button and then choose region, you will
see that a Product Arn will appear. This is the model package ARN that
you need to specify while creating a deployable model. However, for this
notebook, the Model Package ARN has been specified in
**src/model_package_arns.py** file and you do not need to specify the
same explicitly.

2. This notebook requires the IAM role associated with this notebook to
   have **comprehend:DetectEntities** IAM permission.

Step 1: Perform preliminary analysis on the dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    #In this section, we will import necessary libraries and define variables such as S3 bucket, etc.
    import json
    import re
    import datetime
    import calendar
    
    import boto3
    import sagemaker as sage
    from sagemaker import get_execution_role
    from sagemaker import ModelPackage
    from src.model_package_arns import ModelPackageArnProvider
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    import nltk
    from nltk.corpus import wordnet
    nltk.download('wordnet')
    
    
    role = get_execution_role()
    sagemaker_session = sage.Session()
    comprehend = boto3.client('comprehend')

.. code:: ipython3

    #Lets define a utility function thats accepts text and returns trigrams,bigrams,and unigrams.
    def get_grams(text):
        
        potential_product_names=[]
        
        #Identify trigrams
        trigrams = [text for text in zip(text.split(" ")[:-1], text.split(" ")[1:],text.split(" ")[2:])]
        for trigram in trigrams:
            potential_product_names.append(' '.join(trigram))
        
        #Identify bigrams    
        bigrams = [text for text in zip(text.split(" ")[:-1], text.split(" ")[1:])]
        for bigram in bigrams:
            potential_product_names.append(' '.join(bigram))
        #Identify unigrams
        potential_product_names=potential_product_names+ text.split(" ")
        
        return set(potential_product_names)
    
    text='CONCH TOUR TRAIN 1034 G KEY WEST FL'
    
    print('Number of unigrams/bigrams/tri-grams:',len(get_grams(text)))
    print('unigrams/bigrams/tri-grams found: ',get_grams(text))

Step 1.1 Load and View the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During inspection of the dataset, you will see that the transaction
description consists of following parts: \* Transaction type \* Date of
transaction \* Merchant name \* Transaction/Vendor Location

.. code:: ipython3

    df = pd.read_csv('data/raw/sample-transaction-data.csv', index_col=None)
    df.head()

We can see that **description** column contains anonymized data
(sequences of character ‘X’), let us remove the anonymous text and
special characters from the description.

.. code:: ipython3

    #The following method accepts a text and performs following tasks:
    #1. Removes anonymized words(In this dataset, anonymized values are sequences of letter X).
    #2. Removes all special characters.
    def clean_text(text):
        text=text.strip()
        
        #Remove special characters
        text = re.sub('[^A-Za-z0-9. /]+','', text)
        
        #Remove anonymized values
        text = re.sub('(^X+)|( X+ )|(X+$)',' ', text)
        return text.strip()
    
    #Let's test the function
    text="XXX RECURRING TRANSFER TO CHIKXXKI K XXXXXXXX SAVINGS REF XXXXXXXX XXXXXXXXX"
    print(clean_text(text))

.. code:: ipython3

    #Let us clean values from the description column
    df['description']=df['description'].apply(lambda x:clean_text(x))
    df.tail()

Step 1.2 Identify Transaction date
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The date available on the ledger is the transaction posted date. Let us
rename the ‘date’ column to reflect the same.

.. code:: ipython3

    df.rename(columns={"date": "transaction_posted_date"},inplace=True)

Let us extract the transaction date available in the description field.
Based on preliminary examination, it is clear that this information is
available in mm/dd format.

.. code:: ipython3

    #extract_date function extracts month and day from a text that contains date in mm/dd format
    def extract_date(text):
        DATE_EXTRACTION_REGEX='ON ([\d]?\d)/([\d]?\d)'
        return re.findall(DATE_EXTRACTION_REGEX,text)
    
    #extract_date('PURCHASE AUTHORIZED ON 03/23 XXXX CONCH TOUR TRAIN XXX XXXXX 1034 G KEY WEST FL XXXXXXXX')

.. code:: ipython3

    #x1 = datetime.datetime(2020, 12, 31)
    #x2 = datetime.datetime(2021, 10, 1)
    #abs((x1-x2).days)

.. code:: ipython3

    #This function extracts the date on which transaction occured. 
    def set_transaction_date(row):
        
        posted_date=datetime.datetime.strptime(row['transaction_posted_date'], '%m/%d/%y')
        
        result=extract_date(row['description'])
        if result:
            month=int(result[0][0])
            day= int(result[0][1])
            
            row['transaction_month']=month
            row['transaction_date']=day
            transaction_date=datetime.datetime(int(posted_date.year), month, day)
            
            #Logic to carry forward the year. 
            # Here we assume that the transaction gets posted in less than 20 days from the actual transaction date.
            if(abs((posted_date-transaction_date).days))<20:
                row['transaction_year']=posted_date.year
            else:
                row['transaction_year']=(posted_date.year -1)
        return row

.. code:: ipython3

    df = df.apply(lambda row:set_transaction_date(row),axis=1)

.. code:: ipython3

    df['transaction_month'].isna().sum()

Step 1.3 Identify subscriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on preliminary analysis of the data, we can see that susbcription
log entries contain word “RECURRING”. We will write a rule based on this
to identify subscriptions.

.. code:: ipython3

    #This method accepts a row and identifies subscriptions
    def identify_subscriptions(row):
        if 'RECURRING' in row['description']:
            row['subscription']='True'
            row['state_name']='N/A'
            row['state_code']='N/A'
            row['country_code']='N/A'
            row['city_name']='N/A'
        else:
            row['subscription']='False'
        return row

.. code:: ipython3

    df=df.apply(lambda row:identify_subscriptions(row),axis=1)

.. code:: ipython3

    df['subscription'].value_counts()

Let us take a look at the subscription fees paid.

.. code:: ipython3

    df[df['subscription']=='True'][['description','amount']]

.. code:: ipython3

    #Print the total subscription fees paid.
    df[df['subscription']=='True']['amount'].sum()

Step 2: Use an ML model to identify potential merchants and locations for each transaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transaction description log is machine generated and does not follow
grammar. A rule-based part-of-speech tagger might not yield best
results, which is why we will take a different approach here. We will
feed this information to a Machine learning model specifically developed
for extracting the merchant and location information from a transaction
log. For more information, see the **Product overview** of the
`Transaction Data Parsing
(NER) <https://aws.amazon.com/marketplace/pp/prodview-sqnwjvzzqntn2?qid=1580859301012&sr=0-2&ref_=srh_res_product_title>`__
machine learning model.

Step 2.1: Deploy the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Get model_package_arn
    modelpackage_arn = ModelPackageArnProvider.get_transactional_NER_model_package_arn(sagemaker_session.boto_region_name)
    
    # Define predictor wrapper class
    def ner_detection_predict_wrapper(endpoint, session):
        return sage.RealTimePredictor(endpoint, session, content_type='application/json')
    
    # Create a deployable model for the transaction data parsing model package.
    ner_model = ModelPackage(role=role,
                             model_package_arn=modelpackage_arn,
                             sagemaker_session=sagemaker_session,
                             predictor_cls=ner_detection_predict_wrapper)
    
    # Deploy the model
    ner_predictor = ner_model.deploy(initial_instance_count=1, 
                                     instance_type='ml.m5.xlarge',
                                     endpoint_name='transaction-processing')


**Note**: For ease of demonstration, this notebook deploys an endpoint.
However, instead of deploying an Amazon SageMaker endpoint, you can also
run a batch transform job to perform inference on an ML model.

.. code:: ipython3

    payload = {'instance': 'PURCHASE AUTHORIZED ON 03/23  CONCH TOUR TRAIN  1034 G KEY WEST FL'}
    json_val=json.loads(ner_predictor.predict(json.dumps(payload)).decode('utf-8'))['ner']
    json_val

Step 2.2: Populate candidate merchants and locations in dataframe
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    #This function populates 'prediction' column with prediction performed on description of each transaction log.
    def identify_merchant_and_location(row):
        
        payload = {'instance': row['description']}
        prediction=json.loads(ner_predictor.predict(json.dumps(payload)).decode('utf-8'))['ner']
        
        #delete start_pos and end_pos as we do not require them.
        for value in prediction:
            del value['start_pos']
            del value['end_pos']
        
        row['prediction']=prediction
        return row

.. code:: ipython3

    df = df.apply(lambda row:identify_merchant_and_location(row),axis=1)

.. code:: ipython3

    df.head()

Since the prediction has been saved in the dataframe itself, you do not
need the endpoint anymore. Let us delete the endpoint as well as the
model.

.. code:: ipython3

    ner_predictor.delete_endpoint(delete_endpoint_config=True)
    ner_predictor.delete_model()

Step 3: Identify merchant name from transaction log
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the purpose of this experiment, you will be doing lookups on a
manually curated list of businesses. However, for real-world finance
data processing, you would look into a dataset such as commercial
version of `7+ Million Company
Dataset <https://www.peopledatalabs.com/company-dataset>`__ or products
from `AWS Data
Exchange <https://aws.amazon.com/marketplace/search/results?page=1&filters=FulfillmentOptionType&FulfillmentOptionType=AWSDataExchange&ref_=header_nav_dm_aws_data_exchange>`__
such as `Canada corporate
registrations <https://aws.amazon.com/marketplace/pp/prodview-4u57ozcd5b56e?ref_=srh_res_product_title>`__,
`UK registered
companies <https://aws.amazon.com/marketplace/pp/prodview-sydh5kttmyiag?ref_=srh_res_product_title#overview>`__.

.. code:: ipython3

    #Let us load a sample list of businesses into a dataframe for lookup.
    merchants=pd.read_csv('data/config/businesses.csv')
    merchants.head()

Step 3.1: Identify merchant name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    #This function accepts name and returns merchant type as well as sub_type
    def get_business_type(name):
        if(len(name)>2):
            results = merchants[merchants['name'].str.contains(name.lower())]
            if len(results)>0:
                merchant_type=results.iloc[0]['merchant_type']
                merchant_sub_type=results.iloc[0]['sub_type']
                return [merchant_type,merchant_sub_type]
        return []
    #print(get_business_type('CONCH TOUR TRAIN'))
    
    #This function populates name, type, and sub_type for each row.
    def populate_merchant_info(row):
        
        #We will populate three columns in the dataframe - name, type and subtype.
        #Based on preliminary analysis, if the purchase was made online, description contains a short domain name.
        #Let us identify all online trasactions.
        for name in row['description'].split(" "):
            if (('.' in name) & (len(name)>2)):
                row['vendor_website']=name.lower()
                row['merchant_name']=name.lower()
    
                #Since we are not interested in the state in which website was hosted, lets mark it as N/A
                row['state_code']='N/A'
                row['state_name']='N/A'
                row['city_name']='N/A'
                row['country_code']='N/A'
                business_type=get_business_type(name)
                if len(business_type) >0:
                    row['merchant_type']=business_type[0]
                    row['merchant_sub_type']=business_type[1]
                return row
        
        #Note that the ML model returned all possible candidates for the business name. 
        #Given that business names could be multi-termed entities, we need to do a lookup for all ngrams generated from 
        #candidate merchants - for this experiment, we will stick to trigrams,bigrams, and unigrams.
        #print(row['prediction'])
        
        row['vendor_website']='N/A'
        names=[]
        
        for result in row['prediction']:
            if result['type'] == 'NE_MERCHANT':
                names.append(result['key'])
    
        if len(names) >=1:
            ngrams=get_grams(' '.join(names))
            for ngram in ngrams:
                business_type=get_business_type(ngram)
                if len(business_type) >0:
                    row['merchant_name']=ngram.lower()
                    row['merchant_type']=business_type[0]
                    row['merchant_sub_type']=business_type[1]
                    return row
            
            #If direct lookup of the business name was not successful, then let us use Amazon Comprehend to
            #identify the name of the business.
            for ngram in ngrams:
                result=comprehend.detect_entities(Text=' '.join(ngram),LanguageCode='en')
                if len(result ['Entities']) >0 and result['Entities'][0]['Score']>0.7 and result['Entities'][0]['Type'] == 'ORGANIZATION':
                    row['merchant_name']=result['Entities'][0]['Text']
                    business_type=get_business_type(row['merchant_name'])
                    if len(business_type) >0:
                        row['merchant_type']=business_type[0]
                        row['merchant_sub_type']=business_type[1]
                        return row
        return row

.. code:: ipython3

    %%time
    df=df.apply(lambda row:populate_merchant_info(row),axis=1)

.. code:: ipython3

    df['vendor_website'].value_counts()

.. code:: ipython3

    print((df.isna().sum()/df.shape[0])*100)

We can see that ~49% expenses in the data are either recurring charges
or are happening online.

.. code:: ipython3

    df['merchant_type']=df['merchant_type'].fillna('Unknown')
    df['merchant_sub_type']=df['merchant_sub_type'].fillna('Unknown')
    df['merchant_name']=df['merchant_name'].fillna('Unknown')

.. code:: ipython3

    print('Merchant name not available for',df[df['merchant_name']=='Unknown']['description'].count(),'transactions')
    print('Total amount spent in unknown transactions is',df[df['merchant_name']=='Unknown']['amount'].sum())

Let us take a look at these records.

.. code:: ipython3

    df[df['merchant_name']=='Unknown']['description']

Step 3.2: Visualize expenses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lets plot amount of money spent with a aspecific merchant.

.. code:: ipython3

    df.groupby(['merchant_name']).sum()['amount'].sort_values().plot.bar(subplots=True, figsize=(18, 4))

Lets plot amount of money spent based on merchant type.

.. code:: ipython3

    df.groupby(['merchant_type']).sum()['amount'].sort_values().plot.pie( figsize=(7, 7),legend=True)

Lets plot amount of money spent based on merchant sub-type.

.. code:: ipython3

    df.groupby(['merchant_sub_type']).sum()['amount'].sort_values().plot.bar(subplots=True, legend=True,figsize=(15, 4))

Lets plot a graph that shows amount of money spent each month on a
specific expense-sub-type.

.. code:: ipython3

    months= df['transaction_month'].unique()
    
    fig, axes = plt.subplots(nrows=6, ncols=2,figsize=(14,24))
    for i,month in enumerate(months):    
        row=int(i/2)
        col=i%2
        df[df['transaction_month']==month].groupby(['merchant_sub_type']).sum()['amount'].plot.barh(title=calendar.month_name[month],ax=axes[row][col], legend=True)
    fig.tight_layout()
    fig.show()

Step 4: Identify state and city for each transaction log entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have identified merchant information, let us populate
location information. We will use two resources for this lookup: 1.
`wordnet <https://wordnet.princeton.edu/>`__ database. 2.
`geonames <https://www.geonames.org/>`__ dataset.

Step 4.1: Populate state in which transaction took place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Wordnet <https://wordnet.princeton.edu/>`__ is a lexical database for
english language. In Wordnet, a synset is a distinct concept that is
interlinked with other synsets based on lexical, conceptual, and
semantic relationships. This is an important characteristic of a synset
we will use to lookup the state information.

Let us see how state/country synsets look like:

.. code:: ipython3

    state_synset = wordnet.synsets("State",'n')[0]
    print('State:',state_synset.definition())
    country_synset = wordnet.synsets("Country",'n')[0]
    print('Country:',country_synset.definition())

These are the right word synsets! We will use these synsets to identify
state-codes from candidate location information.

.. code:: ipython3

    #This function populates state_code as well as state_name for each transaction log entry.
    def populate_state(row):
        #Populate state information for non-web/non-subscription transactions
        if (( row['vendor_website'] =='N/A') & (row['subscription']  == 'False')):
    
            #Since state code is towards the end in transaction log, we will iterate prediction in reverse order.
            for result in reversed(row['prediction']):
                if result['type'] == 'NE_STORE_LOCATION':
                    synsets = wordnet.synsets(result['key'])
                    for synset in synsets:
                        #Adjust threshold incase correct state codes are not getting populated.
                        if synset.path_similarity(state_synset) and synset.path_similarity(state_synset)> 0.3:
                            row['state_name']=synset.lemmas()[0].name().strip()
                            row['state_code']=result['key'].strip()
                            return row   
        return row

.. code:: ipython3

    df=df.apply(lambda row:populate_state(row),axis=1)

.. code:: ipython3

    df['state_code'].value_counts()

Step 4.2: Populate city and country
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, Let us download the dictionary of cities in the world that have
population greater than 500 people from
http://download.geonames.org/export/dump/.

.. code:: bash

    %%bash
    wget -O  data/config/cities500.zip 'http://download.geonames.org/export/dump/cities500.zip' -nv 
    unzip -q data/config/cities500.zip -d ./data/config/

Next, we load the data into a dataframe for easier lookup.

.. code:: ipython3

    location_df = pd.read_csv('data/config/cities500.txt', header=None,names=['geonameid','name','countrycode','potential_state_code'],usecols=[0,1,8,10], encoding='utf-8', sep='\t')

.. code:: ipython3

    location_df['name']=location_df['name'].str.lower()

.. code:: ipython3

    location_df[(location_df['name'].str.contains('san antonio'))]

We can see that city name is not unique. We will need to couple
city_name with state_code to uniquely identify the city in which
purchase was made.

.. code:: ipython3

    #This method identifies city from the transaction log description.
    def populate_city(row):
        #Populate state information for non-web/non-subscription transactions
        if (( row['vendor_website'] =='N/A') & (row['subscription']  == 'False')):
            locations=[]
            for result in row['prediction']:
                if result['type'] == 'NE_STORE_LOCATION':
                    locations.append(result['key'])
            
            ngrams=get_grams(' '.join(locations))
            #print(ngrams)
            
            if row['state_code'] =='N/A':
                #Description does not contain statecode, identify the city only if a perfect match is found.
                for ngram in ngrams:    
                    results=location_df[(location_df['name']==ngram.lower())]
                    
                    if len(results)==1:
                        row['city_name']=ngram.strip()
                        row['country_code']=results['countrycode']
                        row['state_code'] =results['potential_state_code']
                        #print(':found->' +ngram)
                        return row
                    elif len(results)>1:
                        print('No statecode available: Multiple candidates found. Aborting : '+results)
            else:
                #Description contains statecode, use the same to uniquely identify the city.
                for ngram in ngrams:
                    results=location_df[(location_df['name'] ==ngram.lower()) &(location_df['potential_state_code'] == row['state_code'])]
                    if len(results)==1:
                        row['city_name']=ngram.strip()
                        row['country_code']=results['countrycode']
                        #print('found->' +ngram)
                        return row
                    elif len(results)>1:
                        print(row['state_code']+'multiple candidates found. Aborting : '+results)
    
            print('City not found in Transaction: :',row['description'],': State identified: '+row['state_name'])
        return row

.. code:: ipython3

    %%time
    df=df.apply(lambda row:populate_city(row),axis=1)

City name was not populated for those records for which city information
is not available in the transaction log.

Next, let us visualize the expenditure by city

.. code:: ipython3

    months= df['transaction_month'].unique()
    
    fig, axes = plt.subplots(nrows=4, ncols=3,figsize=(14,8))
    for i,month in enumerate(months):    
        row=int(i/3)
        col=i%3
        df[df['transaction_month']==month].groupby(['city_name']).sum()['amount'].plot.barh(title=calendar.month_name[month],ax=axes[row][col], legend=True)
        axes[row][col].yaxis.set_label_text("")
    
    fig.tight_layout(pad=3.0)
    fig.show()


Step 5: Next steps and cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that transaction data is available in a structured format, you can
use it for multiple purposes such as: 1. To identify amounts you paid
for items/services that are tax deductible. 2. If you have enough data,
train a model that learns your expenditure patterns and identifies
fraudulent activity. 3. Identify expenditure patterns from the data.

Here are some other models from AWS Marketlace you could potentially
explore to do more with ML on your financial data: 1. `Mphasis
DeepInsights Card Fraud
Analyzer <https://aws.amazon.com/marketplace/pp/prodview-cgigha6wcty26?qid=1584052648768&sr=0-2&ref_=srh_res_product_title>`__
to identify fraudulent activity. 2. `Credit Default
Prediction <https://aws.amazon.com/marketplace/pp/prodview-ivuqcwb5yrrh2?qid=1584052502210&sr=0-1&ref_=srh_res_product_title>`__
to help support your loan process. 3. `Loan Approval
Prediction <https://aws.amazon.com/marketplace/pp/prodview-wjoa4tqle6ism?qid=1584052983476&sr=0-5&ref_=brs_res_product_title>`__
to help support loan approval process. 4. `DeepInsights Branch Location
Predictor <https://aws.amazon.com/marketplace/pp/prodview-b4drdxcomdyvg?qid=1584053422977&sr=0-11&ref_=brs_res_product_title>`__
to help identify potential location for a new branch.

Finally, if the AWS Marketplace subscription was created just for an
experiment and you would like to unsubscribe, here are the steps that
can be followed. Before you cancel the subscription, ensure that you do
not have any `deployable
model <https://console.aws.amazon.com/sagemaker/home#/models>`__ created
from the model package or using the algorithm. Note - You can find this
information by looking at the container name associated with the model.

**Steps to unsubscribe from the product on AWS Marketplace:**

Navigate to Machine Learning tab on Your `Software subscriptions
page <https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=lbr_tab_ml>`__.
Locate the listing that you would need to cancel, and click Cancel
Subscription.
