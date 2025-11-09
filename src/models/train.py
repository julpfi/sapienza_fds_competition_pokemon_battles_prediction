
from . import models as models
from . import tune 
import pandas as pd
from sklearn.base import BaseEstimator
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
        Base estimator: Fitted model (basemodel or best-fit from gird search; 
                                    for logistic that is packed inside the pipeline)
    '''

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
        
        # RFE PART 
        # Check if the model is a pipeline and contains RFE
        if hasattr(model, 'named_steps'):
            rfe = model.named_steps.get('feature_selection')  # Adjust the name if different
            if rfe is not None:
                # Get the support mask and feature names
                selected_features = rfe.support_
                feature_names = X.columns
                best_features = feature_names[selected_features]
                print("Best found features after RFE:", best_features.tolist())


    # No grid search => fit baseline model 
    else:
        print("Training without hyperparameter tuning\n")
        model.fit(X, y)
    
    print(f"Completed training\n")
    return model