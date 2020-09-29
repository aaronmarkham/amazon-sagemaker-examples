Copying data from Redshift to S3 and back
=========================================

--------------

--------------

Contents
--------

1. `Introduction <#Introduction>`__
2. `Reading from Redshift <#Reading-from-Redshift>`__
3. `Upload to S3 <#Upload-to-S3>`__
4. `Writing back to Redshift <#Writing-back-to-Redshift>`__

Introduction
------------

In this notebook we illustrate how to copy data from Redshift to S3 and
vice-versa.

Prerequisites
~~~~~~~~~~~~~

In order to successfully run this notebook, you’ll first need to: 1.
Have a Redshift cluster within the same VPC. 1. Preload that cluster
with data from the `iris data
set <https://archive.ics.uci.edu/ml/datasets/iris>`__ in a table named
public.irisdata. 1. Update the credential file
(``redshift_creds_template.json.nogit``) file with the appropriate
information.

Also, note that this Notebook instance needs to resolve to a private IP
when connecting to the Redshift instance. There are two ways to resolve
the Redshift DNS name to a private IP: 1. The Redshift cluster is not
publicly accessible so by default it will resolve to private IP. 1. The
Redshift cluster is publicly accessible and has an EIP associated with
it but when accessed from within a VPC, it should resolve to private IP
of the Redshift cluster. This is possible by setting following two VPC
attributes to yes: DNS resolution and DNS hostnames. For instructions on
setting that up, see Redshift public docs on `Managing Clusters in an
Amazon Virtual Private Cloud
(VPC) <https://docs.aws.amazon.com/redshift/latest/mgmt/managing-clusters-vpc.html>`__.

Notebook Setup
~~~~~~~~~~~~~~

Let’s start by installing ``psycopg2``, a PostgreSQL database adapter
for the Python, adding a few imports and specifying a few configs.

.. code:: ipython3

    !conda install -y -c anaconda psycopg2

.. code:: ipython3

    import os
    import boto3
    import pandas as pd
    import json
    import psycopg2
    import sqlalchemy as sa
    
    region = boto3.Session().region_name
    
    bucket='<your_s3_bucket_name_here>' # put your s3 bucket name here, and create s3 bucket
    prefix = 'sagemaker/DEMO-redshift'
    # customize to your bucket where you have stored the data
    
    credfile = 'redshift_creds_template.json.nogit'

Reading from Redshift
---------------------

We store the information needed to connect to Redshift in a credentials
file. See the file ``redshift_creds_template.json.nogit`` for an
example.

.. code:: ipython3

    # Read credentials to a dictionary
    with open(credfile) as fh:
        creds = json.loads(fh.read())
    
    # Sample query for testing
    query = 'select * from public.irisdata;'

We create a connection to redshift using our credentials, and use this
to query Redshift and store the result in a pandas DataFrame, which we
then save.

.. code:: ipython3

    print("Reading from Redshift...")
    
    def get_conn(creds): 
        conn = psycopg2.connect(dbname=creds['db_name'], 
                                user=creds['username'], 
                                password=creds['password'],
                                port=creds['port_num'],
                                host=creds['host_name'])
        return conn
    
    def get_df(creds, query):
        with get_conn(creds) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result_set = cur.fetchall()
                colnames = [desc.name for desc in cur.description]
                df = pd.DataFrame.from_records(result_set, columns=colnames)
        return df
    
    df = get_df(creds, query)
    
    print("Saving file")
    localFile = 'iris.csv'
    df.to_csv(localFile, index=False)
    
    print("Done")

Upload to S3
------------

.. code:: ipython3

    print("Writing to S3...")
    
    fObj = open(localFile, 'rb')
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, localFile)).upload_fileobj(fObj)
    print("Done")

Writing back to Redshift
------------------------

We now demonstrate the reverse process of copying data from S3 to
Redshift. We copy back the same data but in an actual application the
data would be the output of an algorithm on Sagemaker.

.. code:: ipython3

    print("Reading from S3...")
    # key unchanged for demo purposes - change key to read from output data
    key = os.path.join(prefix, localFile)
    
    s3 = boto3.resource('s3')
    outfile = 'iris2.csv'
    s3.Bucket(bucket).download_file(key, outfile)
    df2 = pd.read_csv(outfile)
    print("Done")

.. code:: ipython3

    print("Writing to Redshift...")
    
    connection_str = 'postgresql+psycopg2://' + \
                      creds['username'] + ':' + \
                      creds['password'] + '@' + \
                      creds['host_name'] + ':' + \
                      creds['port_num'] + '/' + \
                      creds['db_name'];
                        
    df2.to_sql('irisdata_v2', connection_str, schema='public', index=False)
    print("Done")

We read the copied data in Redshift - success!

.. code:: ipython3

    pd.options.display.max_rows = 2
    conn = get_conn(creds)
    query = 'select * from irisdata3'
    df = pd.read_sql_query(query, conn)
    df
