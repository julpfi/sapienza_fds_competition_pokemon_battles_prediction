import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator
from typing import Dict, Any
from utils.config import SEED

def perform_grid_search(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, List[Any]],
    scoring: str = 'accuracy',
    n_splits: int = 5,
    random_state: int = SEED
) -> BaseEstimator:


    print("Starting Grid Search")
    start_time = time.time()

    # 1. Define the Cross-Validation strategy
    # This defines the OUTER loop (K-Fold strategy used by GridSearchCV)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 2. Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=kf,
        verbose=1,
        n_jobs=-1,  # Use all available cores for parallel processing
        refit=True  # WATCH OUT: Automatically retrains the best model on the WHOLE dataset
    )

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

