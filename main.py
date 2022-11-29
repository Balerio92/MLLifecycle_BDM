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

from transformations_VASE import downscale
from datetime import datetime
import tempfile
import os 
import warnings
from utils_VASE import log_results

class WindPowOrkney(mlflow.pyfunc.PythonModel):
    # Estimator of wind power production in Orkney:
    # It predicts the power generation in Orkeny given a wind speed and direction parameters

    def __init__(self, component=True, gamma='auto', kernel='rbf'):
        from sklearn.pipeline import Pipeline
        from transformations_VASE import wind_transformer
        from sklearn.svm import SVR
        
        self.component= component
        self.gamma= gamma
        self.kernel = kernel
        col_trans=ColumnTransformer([
            ('wind_tr', wind_transformer(component=self.component), ['Speed','Direction']),
            ('scaler', StandardScaler(), ['Speed'])
            ])
        
        preprocessor = Pipeline([('col_tr',col_trans)])

        SVR_p= make_pipeline(
            StandardScaler(),
            (SVR(gamma=self.gamma, kernel=self.kernel))
            )
        self.pipeline = make_pipeline(preprocessor, SVR_p)

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
    data_path= r"dataset.json"
    
    mlflow.log_param("dataset_time", datetime.now())

    df = pd.read_json(data_path, orient="split")
        
    df = downscale(df)

    X = df[["Speed","Direction"]]
    y = df["Total"]
    x_train, x_test, y_train, y_test=train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=0) 

    svr_params = {
    'pipeline-1__col_tr__wind_tr__component' : True,   
    'pipeline-2__svr__gamma':'auto',
    'pipeline-2__svr__kernel':'rbf'
    }
    mod = WindPowOrkney(svr_params["pipeline-1__col_tr__wind_tr__component"],svr_params['pipeline-2__svr__gamma'],
    svr_params['pipeline-2__svr__kernel'] ).fit(x_train,y_train)
    r = mod.predict(None, x_test)

    mlflow.log_metric("MAE", mean_absolute_error(y_test, r))    
    mlflow.log_metric("RMSE", mean_squared_error(y_test, r, squared=True))
    mlflow.log_metric("r2", r2_score(y_test, r))
    tags_SVR = {"engineering": "Azure Platform",
        "release.candidate": "SVR_Best",
        "release.version": "0.1.0"}

    mlflow.log_params(svr_params)
    mlflow.set_tags(tags_SVR)

    mlflow.pyfunc.save_model("SVR_model", python_model=mod, conda_env='conda.yaml')
