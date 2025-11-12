

import pandas as pd
from sklearn.base import BaseEstimator

from . import models as models
from . import tune 
from . import custom_voting_model
from . import voting_model
from utils.user_model_selection import get_user_model_selection_voting, get_user_model_selection_custom_voting


def train(X:pd.DataFrame, y:pd.Series, model_type:str="logistic", grid_search:bool=True)-> BaseEstimator:
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
        Base estimator: Fitted model (Best fit form GridSearch/RandomSearchCV)
    '''

    # Train is the main entry point for training the models 
    #   => For the two voting classifiers we call the differing methods from here
    if model_type == "custom_voting": 
        model_names = get_user_model_selection_custom_voting()
        print("Selected models for Custom Voting:", model_names)
        return custom_voting_model.train_voting(X=X, y=y, model_names=model_names,  grid_search=grid_search)
    elif model_type == "voting": 
        model_names = get_user_model_selection_voting()
        print("Selected models for Voting:", model_names)
        return voting_model.train_voting(X=X, y=y, model_names=model_names)

    model = models.get_model(model_type=model_type)

    if grid_search: 
        # Gets defined param grid assocaited to model
        param_grid = models.get_param_grid(model_type=model_type)

        # Performs gridsearch and fits model with best parameterization
        model, _ = tune.perform_grid_search(model, X, y, param_grid, model_type=model_type)
        
        # Prints feature importance for tree based classifiers (random forest and xgb)
        if hasattr(model, 'named_steps'):
            feature_selection = model.named_steps.get('feature_selection')
            classifier = model.named_steps.get('model')
            current_features = X.columns 
            
            if feature_selection is not None:
                selected_mask = feature_selection.get_support()
                current_features = X.columns[selected_mask] 
                print(f"\n--- Selected Features for {model_type} ---")
                print(current_features.tolist())

            if model_type in ["random_forest", "xgboost"]:
                importance = classifier.feature_importances_
                feature_importance_df = pd.DataFrame({'Feature': current_features, 'Importance': importance})
                feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
                print(f"\n--- Feature Importances for {model_type} ---")
                print(feature_importance_df.to_string())
                #print("\n TAIL:", feature_importance_df.tail(20))


    # No grid search => fit baseline model 
    else:
        print("Training without hyperparameter tuning\n")
        model.fit(X, y)
    

    print(f"Completed training\n")
    return model