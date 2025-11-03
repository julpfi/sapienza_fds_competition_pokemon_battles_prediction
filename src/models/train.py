
from . import models as models
from . import tune 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.base import BaseEstimator


def train(X:pd.DataFrame, y:pd.Series, model_type:str="logistic", grid_search:bool=True
          )-> BaseEstimator:
    '''
    Description: 
        Trains the selected model type given the feature set of the training data and the assocaited outcomes.
        If selected, executes gridsearch and fits models on best estiamtors.
        For logistic regression, a pipeline is set up to implement necessary standardization
    Param: 
        X: feature data set of training data
        y: Outcome variable for each record 
        model_type (str): Declares what model to fit 
        grid_search (bool): Determines if basemodel is return or if grid search is executed 
                            which fits the model for the best parameters  
    Retunrs: 
        Base estimator: Fitted model (basemodel or best-fit from gird search; 
                                    for logistic that is packed inside the pipeline)
    '''

    # Create a pipeline adds scaling to the model 
    # NOTE: Custom standardization and Gridsearch might cause porblem 
    #       => Use pipeline as sklearn takes care of that
    if model_type == "logistic":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', models.get_model(model_type=model_type))
            ])
    elif model_type in  ["random_forest", "xgboost"]:
        model = models.get_model(model_type=model_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


    if grid_search: 
        if model_type == "logistic":
            # Create para_grid that is handeled in pipeline step
            # We must prefix parameters with 'model__' (the name of our step)
            param_grid = {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__penalty": ["l1", "l2"], 
                "model__solver": ["liblinear", "saga"],
                "model__max_iter": [5000]
            }

        elif model_type == "random_forest":
            # Random forest does not need scaling => no pipeline needed => direct param grid
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            }
        elif model_type == "xgboost": 
            # XGboost does not need scaling => no pipeline needed => direct param grid
            param_grid = {
                'n_estimators': [100, 200, 500, 1000],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.5, 1],
                'reg_lambda': [0.1, 1.0, 5.0, 10.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Performs gridsearch and fits model with best parameterization
        model = tune.perform_grid_search(model, X, y, param_grid, model_type=model_type)

    # No grid search => fit baseline model 
    else:
        print("Training without hyperparameter tuning\n")
        model.fit(X, y)

    print(f"Completed training\n")
    return model