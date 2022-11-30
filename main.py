#%%
# Import some of the sklearn modules you are likely to use.
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)

from src.transformations_VASE import downscale
from datetime import datetime
import tempfile
import os 
import warnings

#mlflow.set_tracking_uri("""azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/a5c59ac8-7ad8-42c0-8220-8bda77ee09af/resourceGroups/Assignment3/providers/Microsoft.MachineLearningServices/workspaces/Assignment3_workspace""")
#mlflow.set_experiment("VASE - RF best performing")

class WindPowOrkney(mlflow.pyfunc.PythonModel):
    # Estimator of wind power production in Orkney:
    # It predicts the power generation in Orkeny given a wind speed and direction parameters

    def __init__(self, component=False, max_depth=5, max_leaf_nodes=25, n_estimators=100):
        from sklearn.pipeline import Pipeline
        from src.transformations_VASE import wind_transformer
        from sklearn.ensemble import RandomForestRegressor

        self.component= component
        self.max_depth= max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.n_estimators= n_estimators
        col_trans=ColumnTransformer([
            ('wind_tr', wind_transformer(component=self.component), ['Speed','Direction']),
            ('scaler', StandardScaler(), ['Speed'])
            ])
        
        preprocessor = Pipeline([('col_tr',col_trans)])

        RF_p= make_pipeline(
            StandardScaler(),
            (RandomForestRegressor(max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes,n_estimators= self.n_estimators))
            )
        self.pipeline = make_pipeline(preprocessor, RF_p)

    def fit(self,x,y):
        self.pipeline.fit(x,y)       
        return self

    def predict(self, context, samples):

        pred = self.pipeline.predict(samples)
        tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(tempdir)
        timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
        filename = "call_%s.csv" % ( timestamp)
        csv = os.path.join(tempdir, filename)
        samples['prediction']=pred
        samples['run_ts']=datetime.now()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.DataFrame(samples).to_csv(csv, index=False)

        mlflow.log_artifact(csv, "results")                

        return pred

if __name__ == "__main__":
    data_path= "dataset.json"
    timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
    mlflow.log_param("dataset_time", timestamp)

    df = pd.read_json(data_path, orient="split")
        
    df = downscale(df)

    X = df[["Speed","Direction"]]
    y = df["Total"]
    x_train, x_test, y_train, y_test=train_test_split(
        X, y, test_size=0.2, shuffle=False) 

    rf_params = {
        'pipeline-1__col_tr__wind_tr__component' :  False, 
        'pipeline-2__randomforestregressor__max_depth': 5,
        'pipeline-2__randomforestregressor__max_leaf_nodes': 25,
        'pipeline-2__randomforestregressor__n_estimators': 100
    }

    mod = WindPowOrkney(
    component = rf_params["pipeline-1__col_tr__wind_tr__component"],
    max_depth = rf_params['pipeline-2__randomforestregressor__max_depth'],
    max_leaf_nodes = rf_params['pipeline-2__randomforestregressor__max_leaf_nodes'],
    n_estimators = rf_params['pipeline-2__randomforestregressor__n_estimators'] ).fit(x_train,y_train)
    
    r = mod.predict(None, x_test)

    mlflow.log_metric("MAE", mean_absolute_error(y_test, r))    
    mlflow.log_metric("RMSE", mean_squared_error(y_test, r, squared=True))
    mlflow.log_metric("r2", r2_score(y_test, r))
    tags_RF = {"engineering": "Azure Platform",
        "release.candidate": "RF_Best",
        "release.version": "0.1.0"}

    mlflow.log_params(rf_params)
    mlflow.set_tags(tags_RF)
    #mlflow.sklearn.log_model(mod, "model", conda_env='conda.yaml')
    mlflow.pyfunc.save_model(f"model_{timestamp}", python_model=mod, conda_env='conda.yaml')
    mlflow.end_run()