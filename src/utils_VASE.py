
#import conda.cli.python_api as Conda
#import re
from sklearn.model_selection import GridSearchCV
import mlflow
import warnings
import tempfile
import os
from datetime import datetime
import pandas as pd
import os
from urllib.parse import urlparse
from os.path import exists

def is_local(url):
    url_parsed = urlparse(url)
    if url_parsed.scheme not in ('http', 'https', 'azureml'): # Possibly a local file
        return True
    return False

#def conda_reqs_to_dict():
#    output = Conda.run_command(Conda.Commands.LIST)
#    # output is a tuple, also containing the exitcode
#    data = output[0].split("\n")
#
#    # skip header row, andfinal line termination
#    words = [re.findall("[^\s]+",x) for x in data[3:-1]]
#    a= pd.DataFrame(words)
#    a[a['3']!='pypi']
#    return (a[0]+'=='+a[1]).to_list()
#    #return [w[0]+"=="+w[1] for w in words]


def log_run(gridsearch: GridSearchCV, experiment_name: str, model_name: str, run_index: int, conda_env, tags={}):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        conda_env (str): A dictionary that describes the conda environment (MLFlow Format)
        tags (dict): Dictionary of extra data and tags (usually features)
    """
    
    cv_results = gridsearch.cv_results_
    with mlflow.start_run(run_name=str(run_index)) as run:  

        mlflow.log_param("folds", gridsearch.cv)

        print("Logging parameters")
        params = list(gridsearch.param_grid.keys())
        for param in params:
            mlflow.log_param(param, cv_results["param_%s" % param][run_index])

        print("Logging metrics")
        for score_name in [score for score in cv_results if "mean_test" in score]:
            mlflow.log_metric(score_name, cv_results[score_name][run_index])
            mlflow.log_metric(score_name.replace("mean","std"), cv_results[score_name.replace("mean","std")][run_index])
        print("Logging model")        
        mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name, conda_env=conda_env)

        print("Logging CV results matrix")
        tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(tempdir)
        timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
        filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
        csv = os.path.join(tempdir, filename)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.DataFrame(cv_results).to_csv(csv, index=False)
        
        mlflow.log_artifact(csv, "cv_results")                
        mlflow.set_tags(tags) 

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        print(mlflow.get_artifact_uri())
        print("runID: %s" % run_id)
        print( "experimentID: %s" % experiment_id )

        mlflow.end_run()

def log_results(gridsearch: GridSearchCV, experiment_name, model_name, deploy_url,conda_env, tags={}, log_only_best=False):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        tags (dict): Dictionary of extra tags
        log_only_best (bool): Whether to log only the best model in the gridsearch or all the other models as well
    """
    #conda_env = {
    #    'name': 'mlflow-env',
    #    'channels': ['conda-forge'],
     #   'dependencies': conda_reqs_to_dict()
     #   }

    if (is_local(deploy_url)) & (~os.path.exists(deploy_url)):
           os.makedirs(deploy_url)
           print(f"Creating folder to log experiments at {deploy_url}")

    mlflow.set_tracking_uri(deploy_url)



    best = gridsearch.best_index_

    mlflow.set_experiment(experiment_name)


    if(log_only_best):
        log_run(gridsearch, experiment_name, model_name, best, conda_env, tags)
    else:
        for i in range(len(gridsearch.cv_results_['params'])):
            log_run(gridsearch, experiment_name, model_name, i, conda_env, tags)
