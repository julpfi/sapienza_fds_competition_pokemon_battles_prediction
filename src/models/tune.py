import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator

from utils.config import SEED
import time

def perform_grid_search(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    n_iter: int = 50,
    scoring: str = 'accuracy',
    n_splits: int = 5,
    random_state: int = SEED,
    model_type:str="logistic"
) -> BaseEstimator:


    print("Starting Grid Search")
    start_time = time.time()

    # 1. Define the Cross-Validation strategy
    # Stratified ensures balanced splits; Althpgh not needed (balanced data) might help and does not hurt (recommeneded in several sources)
    kf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    #kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # 2. Initialize Grid Search or RandomizedSearchCV for xgboost 
    if model_type in ["logistic", "random_forest"]:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=kf,
            verbose=1,
            n_jobs=-1,  # How the process is run on local cores
            refit=True, 

        )
    elif model_type == "xgboost":
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter, #Only for RandomizedSearch
            cv=kf,
            scoring=scoring,
            n_jobs=-1,  # How the process is run on local cores
            random_state=SEED,
            verbose=1
        )
        
    else:
         raise ValueError(f"Unknown model type for tuning: {model_type}")

    # 3. Execute Search (Fit)
    # The Grid Search performs K-Fold CV internally for every parameter combination.
    grid_search.fit(X, y)

    end_time = time.time()
    
    # 4. Output Results
    print("\n--- Tuning Results ---")
    print("Time Elapsed:", round(end_time - start_time, 2) , "seconds")
    print(f"Best Hyperparameters found: {grid_search.best_params_}")
    print(f"Best Mean CV Score ({scoring}):", round(grid_search.best_score_, 4))
    
    # The grid_search.best_estimator_ is the best model retrained on the entire X and y.
    return grid_search.best_estimator_

