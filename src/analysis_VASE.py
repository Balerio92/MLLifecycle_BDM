#%%
# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import mlflow
import mlflow.pyfunc
import pandas as pd


from transformations_VASE import wind_transformer
from transformations_VASE import downscale

from utils_VASE import log_results

from plotting_VASE import (plot_search_results, plot_powe_time)
#%%

from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append("..")
#%%
data_path= r"../dataset.json"
    
df = pd.read_json(data_path, orient="split")
    
df = downscale(df)


col_trans=ColumnTransformer([
    ('wind_tr', wind_transformer(), ['Speed','Direction']),
    ('scaler', StandardScaler(), ['Speed'])
    ]
)
preprocessor = Pipeline([('col_tr',col_trans)])

SVR_p= make_pipeline(
    StandardScaler(),
    (SVR()),
)
random_forest = make_pipeline(
    StandardScaler(),
    RandomForestRegressor()

)
pipeline_SVR= make_pipeline(preprocessor, SVR_p)
pipeline_RF= make_pipeline(preprocessor, random_forest)

svr_params = {
    'pipeline-1__col_tr__wind_tr__component' : [True,False],   
    'pipeline-2__svr__gamma':[1, 0.1, 0.01,'auto'],
    'pipeline-2__svr__kernel':['poly', 'rbf', 'sigmoid']
}
rf_params = {
    'pipeline-1__col_tr__wind_tr__component' :  [True,False], 
    'pipeline-2__randomforestregressor__max_depth': [5,10,50],
    'pipeline-2__randomforestregressor__max_leaf_nodes': [25,50,100],
    'pipeline-2__randomforestregressor__n_estimators': [10,50,100]
}

X = df[["Speed","Direction"]]
y = df["Total"]
x_train, x_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=0) 
my_cv = TimeSeriesSplit(n_splits=5)
scoring = {"RMSE": "neg_root_mean_squared_error", "R2": 'r2','max_error':'max_error' }

#%%
grid_SVR = GridSearchCV(pipeline_SVR, svr_params,
                         scoring=scoring,
                         return_train_score=True,
                         refit = 'RMSE',
                         cv=my_cv, verbose = 1)
grid_SVR.fit(x_train,y_train)
svr_pred=grid_SVR.predict(x_test)

grid_RF= GridSearchCV(pipeline_RF, rf_params,
                         scoring=scoring,
                         refit = 'RMSE',
                         return_train_score=True,
                         cv=my_cv, verbose = 1)
grid_RF.fit(x_train,y_train)
rf_pred=grid_RF.predict(x_test)

# %%
#deploy_url = """azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/a5c59ac8-7ad8-42c0-8220-8bda77ee09af/resourceGroups/Assignment3/providers/Microsoft.MachineLearningServices/workspaces/Assignment3_workspace"""

deploy_url = 'mlruns'
condapath = '../conda.yaml'
tags_RF = {"engineering": "Azure Platform",
        "release.candidate": "RF1",
        "release.version": "0.0.1"}

log_results(grid_RF,"VASE - RandomForestGridSearch Final","RF",deploy_url,conda_env= condapath,  tags=tags_RF)

tags_SVR = {"engineering": "Azure Platform",
        "release.candidate": "SVR1",
        "release.version": "0.0.1"}
log_results(grid_SVR,"VASE - SVR Grid Search Final","SVR",deploy_url, conda_env= condapath, tags=tags_SVR )

# %%
plot_search_results(grid_RF)
plot_search_results(grid_SVR)
       
# %%
plot_powe_time(x_train,x_test,y_train,y_test, svr_pred)

plot_powe_time(x_train,x_test,y_train,y_test, rf_pred)

# %%

# Load mlflow res
RF_df=mlflow.search_runs(experiment_names=[ "VASE - RandomForestGridSearch Final" ])

SVR_df=mlflow.search_runs(experiment_names=[ "VASE - SVR Grid Search Final" ])
perf_df = RF_df.append(SVR_df)

# %%
perf_df.sort_values(['metrics.std_test_RMSE'], inplace= True)

red = perf_df.iloc[:20]
# %%
tags_SVR = {"engineering": "Azure Platform",
        "release.candidate": "SVR_Best",
        "release.version": "0.1.0"}
log_results(grid_SVR,"VASE - SVR Best Grid","SVR", deploy_url,conda_env= condapath, tags=tags_SVR, log_only_best=True )
