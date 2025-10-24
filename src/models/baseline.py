from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression(**kwargs):
    # creates param dict and sets default values 
    model_params = dict(random_state=42, max_iter=1000) 
    # updates params -> overwrites defaults if new value is passed
    model_params.update(kwargs) 
    return LogisticRegression(**model_params)

def get_random_forest(**kwargs):
    # creates param dict and sets default values 
    model_params = dict(random_state=42, n_estimators=100) 
    # updates params -> overwrites defaults if new value is passed
    model_params.update(kwargs) 
    return RandomForestClassifier(**model_params)

def get_model(model_type: str = "logistic", **kwargs):
    '''
    Description: 
        Wrapper function that creates a logistic regression or random forest model
        We can pass params that are used in the model creation; if no params are passed the necessary default values are used
    Params: 
        model_type: str: selects the type of model (options: "logistic", "random_forest")
    '''
    if model_type == "logistic":
        return get_logistic_regression(**kwargs)
    elif model_type == "random_forest":
        return get_random_forest(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    