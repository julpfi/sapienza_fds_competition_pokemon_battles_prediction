from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils.config import SEED
from xgboost import XGBClassifier

def get_logistic_regression():
    # creates param dict and sets default values 
    model_params = dict(random_state=SEED, max_iter=1000) 
    return LogisticRegression(**model_params)

def get_random_forest():
    # creates param dict and sets default values 
    model_params = dict(random_state=SEED, n_estimators=100) 
    return RandomForestClassifier(**model_params)

def get_xgboost():
    # creates param dict and sets default values 
    model_params = dict(random_state=SEED)
    return XGBClassifier(**model_params)

def get_model(model_type: str = "logistic"):
    '''
    Description
        Wrapper function that creates a logistic regression or random forest model
        We can pass params that are used in the model creation; if no params are passed the necessary default values are used
    Params: 
        model_type: str: selects the type of model (options: "logistic", "random_forest")
    '''
    if model_type == "logistic":
        return get_logistic_regression()
    elif model_type == "random_forest":
        return get_random_forest()
    elif model_type == "xgboost": 
        return get_xgboost()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    