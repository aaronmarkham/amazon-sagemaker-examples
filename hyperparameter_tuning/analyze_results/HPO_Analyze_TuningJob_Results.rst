Analyze Results of a Hyperparameter Tuning job
==============================================

Once you have completed a tuning job, (or even while the job is still
running) you can use this notebook to analyze the results to understand
how each hyperparameter effects the quality of the model.

--------------

Set up the environment
----------------------

To start the analysis, you must pick the name of the hyperparameter
tuning job.

.. code:: ipython3

    import boto3
    import sagemaker
    import os
    
    region = boto3.Session().region_name
    sage_client = boto3.Session().client('sagemaker')
    
    tuning_job_name = 'YOUR-HYPERPARAMETER-TUNING-JOB-NAME'

Track hyperparameter tuning job progress
----------------------------------------

After you launch a tuning job, you can see its progress by calling
describe_tuning_job API. The output from describe-tuning-job is a JSON
object that contains information about the current state of the tuning
job. You can call list_training_jobs_for_tuning_job to see a detailed
list of the training jobs that the tuning job launched.

.. code:: ipython3

    # run this cell to check current status of hyperparameter tuning job
    tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)
    
    status = tuning_job_result['HyperParameterTuningJobStatus']
    if status != 'Completed':
        print('Reminder: the tuning job has not been completed.')
        
    job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
    print("%d training jobs have completed" % job_count)
        
    is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
    objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']

.. code:: ipython3

    from pprint import pprint
    if tuning_job_result.get('BestTrainingJob',None):
        print("Best model found so far:")
        pprint(tuning_job_result['BestTrainingJob'])
    else:
        print("No training jobs have reported results yet.")

Fetch all results as DataFrame
------------------------------

We can list hyperparameters and objective metrics of all training jobs
and pick up the training job with the best objective metric.

.. code:: ipython3

    import pandas as pd
    
    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
    
    full_df = tuner.dataframe()
    
    if len(full_df) > 0:
        df = full_df[full_df['FinalObjectiveValue'] > -float('inf')]
        if len(df) > 0:
            df = df.sort_values('FinalObjectiveValue', ascending=is_minimize)
            print("Number of training jobs with valid objective: %d" % len(df))
            print({"lowest":min(df['FinalObjectiveValue']),"highest": max(df['FinalObjectiveValue'])})
            pd.set_option('display.max_colwidth', -1)  # Don't truncate TrainingJobName        
        else:
            print("No training jobs have reported valid results yet.")
            
    df

See TuningJob results vs time
-----------------------------

Next we will show how the objective metric changes over time, as the
tuning job progresses. For Bayesian strategy, you should expect to see a
general trend towards better results, but this progress will not be
steady as the algorithm needs to balance *exploration* of new areas of
parameter space against *exploitation* of known good areas. This can
give you a sense of whether or not the number of training jobs is
sufficient for the complexity of your search space.

.. code:: ipython3

    import bokeh
    import bokeh.io
    bokeh.io.output_notebook()
    from bokeh.plotting import figure, show
    from bokeh.models import HoverTool
    
    class HoverHelper():
    
        def __init__(self, tuning_analytics):
            self.tuner = tuning_analytics
    
        def hovertool(self):
            tooltips = [
                ("FinalObjectiveValue", "@FinalObjectiveValue"),
                ("TrainingJobName", "@TrainingJobName"),
            ]
            for k in self.tuner.tuning_ranges.keys():
                tooltips.append( (k, "@{%s}" % k) )
    
            ht = HoverTool(tooltips=tooltips)
            return ht
    
        def tools(self, standard_tools='pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset'):
            return [self.hovertool(), standard_tools]
    
    hover = HoverHelper(tuner)
    
    p = figure(plot_width=900, plot_height=400, tools=hover.tools(), x_axis_type='datetime')
    p.circle(source=df, x='TrainingStartTime', y='FinalObjectiveValue')
    show(p)

Analyze the correlation between objective metric and individual hyperparameters
-------------------------------------------------------------------------------

Now you have finished a tuning job, you may want to know the correlation
between your objective metric and individual hyperparameters you’ve
selected to tune. Having that insight will help you decide whether it
makes sense to adjust search ranges for certain hyperparameters and
start another tuning job. For example, if you see a positive trend
between objective metric and a numerical hyperparameter, you probably
want to set a higher tuning range for that hyperparameter in your next
tuning job.

The following cell draws a graph for each hyperparameter to show its
correlation with your objective metric.

.. code:: ipython3

    ranges = tuner.tuning_ranges
    figures = []
    for hp_name, hp_range in ranges.items():
        categorical_args = {}
        if hp_range.get('Values'):
            # This is marked as categorical.  Check if all options are actually numbers.
            def is_num(x):
                try:
                    float(x)
                    return 1
                except:
                    return 0           
            vals = hp_range['Values']
            if sum([is_num(x) for x in vals]) == len(vals):
                # Bokeh has issues plotting a "categorical" range that's actually numeric, so plot as numeric
                print("Hyperparameter %s is tuned as categorical, but all values are numeric" % hp_name)
            else:
                # Set up extra options for plotting categoricals.  A bit tricky when they're actually numbers.
                categorical_args['x_range'] = vals
    
        # Now plot it
        p = figure(plot_width=500, plot_height=500, 
                   title="Objective vs %s" % hp_name,
                   tools=hover.tools(),
                   x_axis_label=hp_name, y_axis_label=objective_name,
                   **categorical_args)
        p.circle(source=df, x=hp_name, y='FinalObjectiveValue')
        figures.append(p)
    show(bokeh.layouts.Column(*figures))
