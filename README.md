# MLflow project for wind power production in Orkney

This Mlflow project was produced as part of the course Big Data Management Technical at ITU. 

### Requirements 
To run the project a python installation and the mlflow package are required. 
The mlflow package can be installed through the command:
 `pip install mlflow`

### Model build
To build the mlflow project run the command:
`mlflow run .`

### Deploy the model 
To deploy the built model run the command:
`mlflow models serve -m model_timestamp`