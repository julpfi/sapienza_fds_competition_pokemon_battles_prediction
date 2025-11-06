from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import SEED

# --------------- MODEL HELPERS ---------------

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
    model_params = dict(random_state=SEED, objective='binary:logistic')
    return XGBClassifier(**model_params)

def get_knn():
    # creates param dict and sets default values 
    model_params = dict() # No random state for KNN
    return KNeighborsClassifier(**model_params)

# --------------- BASE MODEL  ---------------

def get_base_model(model_type: str = "logistic"):
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
    elif model_type == "knn": 
        return get_knn()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    


# --------------- GET MODEL AS PIPELINE ---------------

def get_model(model_type:str): 
    '''
    #TODO: Description 
    '''
    if model_type in ["logistic", "knn"]:
        # Create a pipeline adds scaling to the model 
        # Custom standardization and Gridsearch might cause porblem => Use pipeline as sklearn takes care of that
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', get_base_model(model_type=model_type))
            ])
        
    elif model_type in ["random_forest", "xgboost"]:
        # Wrapping in simple pipeline for consistency
        pipeline = Pipeline([
            ('model', get_base_model(model_type=model_type))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return pipeline


# --------------- CONFIG PARAM GRID ---------------

def get_param_grid(model_type:str):
    '''
    #TODO: Description 
        # Create para_grid that is handeled in pipeline step
        # We must prefix parameters with 'model__' (the name of our step)
    '''
    if model_type == "logistic":
        # Pipeline needed for scaling => must use 'model__' prefix
        return  [
            # 1. lbfgs (L2 only)
            {
                "model__solver": ["lbfgs"],
                "model__penalty": ["l2"],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__max_iter": [1000]
            },
            
            # 2. liblinear (L1 and L2))
            {
                "model__solver": ["liblinear"],
                "model__penalty": ["l1", "l2"],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__max_iter": [1000]
            },
            
            # 3. Saga (L1 and L2)
            {
                "model__solver": ["saga"],
                "model__penalty": ["l1", "l2"],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__max_iter": [5000]
            },
            
            # 4. Saga (both L1 and L2 with elasticnet)
            {
                "model__solver": ["saga"],
                "model__penalty": ["elasticnet"],
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__l1_ratio": [0.25, 0.5, 0.75],
                "model__max_iter": [5000]
            }
        ]

    elif model_type == "random_forest":
        # In pipeline for consistency
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20, 40],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"]
        }
    elif model_type == "xgboost": 
        # In pipeline for consistency
        return {
            'model__n_estimators': [100, 200, 500, 1000],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7, 9],
            'model__subsample': [0.7, 0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'model__gamma': [0, 0.1, 0.5, 1],
            'model__reg_lambda': [0.1, 1.0, 5.0, 10.0],
            'model__reg_alpha': [0, 0.1, 0.5, 1.0]
        }
    elif model_type == "knn": 
        # Pipeline needed for scaling => must use 'model__' prefix
        return {
            'model__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan']
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
