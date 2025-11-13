from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import pandas as pd
import time

from .models import get_model, get_param_grid
from models.tune import perform_grid_search
from utils.config import SEED

def train_meta_model(X: pd.DataFrame, y: pd.Series, model_names: list,) -> BaseEstimator:

    print("\n\n--- Training Meta Model ---")
    start_time = time.time()
    
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  

    print("Getting best params for each base learner...")
    base_models = {}
    for name in model_names:
        model = get_model(name)
        param_grid = get_param_grid(name)
        best_model, _ = perform_grid_search(model, X, y, param_grid, model_type=name)
        base_models[name] = best_model
    
    estimators = [(name, base_models[name]) for name in model_names]
    
    meta_model = LogisticRegression(max_iter=1000)
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=inner_cv,
        n_jobs=-1,
        passthrough=False
    )
    
    print("Evaluating full stack with outer cross-validation...")
    cv_scores = cross_val_score(stacking_model, X, y, cv=outer_cv, scoring='accuracy')
    
    print(f"\nMeta model outer CV score:", round(cv_scores.mean(), 4))
    print(f"Time taken:", round((time.time() - start_time), 2))
    
    print("Fitting final meta model on all data...")
    stacking_model.fit(X, y)
    return stacking_model