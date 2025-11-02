
from . import models as models
from tune import perform_grid_search
from data.standardize_data import standardize_data


def train(X, y, model_type:str="logistic", grid_search:bool=True, **kwargs):
    # Standardize data for logistic regression 
    if model_type == "logistic": 
        X = standardize_data(X, train=True)

    # Create baseline model
    model = models.get_model(model_type=model_type)

    # Perform grid search or fit baseline model 
    if grid_search: 
        # Declare param grids used in grid search 
        if model_type == "logistic":
            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"], 
                "solver": ["liblinear", "saga"]
            }
        elif model_type == "random_forest":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
        model = perform_grid_search(model, X, y, param_grid)
    else:
        print("Training without hyperparameter tuning\n")
        model.fit(X, y)

    print(f"Completed training\n")
  
    return model





# Archived as gridsearch implements a k-fold CV already (not stratified but that is ok as data is balanced) 
"""
import numpy as np
from src.utils.model_evaluation import evaluate_classification
from sklearn.model_selection import StratifiedKFold


def cross_validate_model(X, y, model_type, n_splits=5, random_state=42, **model_kwargs):
    # Could also use simple KFold but after research StratifiedKFold works the same and is just more robust
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for (train_x, val_x) in folds.split(X, y):
        X_train, X_test = X.iloc[train_x], X.iloc[val_x]
        y_train, y_test = y.iloc[train_x], y.iloc[val_x]
        
        model = models.get_model(model_type=model_type, **model_kwargs)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = evaluate_classification(y_test, pred)
        results.append(metrics["accuracy"])

    accuracy_all = np.mean(results)
    print("Average CV accuracy:" , round(accuracy_all, 4))
    return accuracy_all

"""
