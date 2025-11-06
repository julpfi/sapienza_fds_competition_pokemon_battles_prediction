from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator
import pandas as pd
import time

from .models import get_model, get_param_grid
from .tune import perform_grid_search
from utils.config import SEED

def train_voting(X: pd.DataFrame, y: pd.Series, model_names: list,) -> BaseEstimator:
    # Trains voting classifier with cv-based weights using grid search tuned base models

    print("\n\n--- Training Weighted Voting Ensemble ---")
    start_time = time.time()

    # Used by perform_grid_search and for per-model scoring
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  
    # Used to evaluate the final ensemble
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  

    # Get base models and best params via perform_grid_search
    print("Getting best params for each weak learner...")
    base_models = {}
    base_scores = {}
    for name in model_names:
        model = get_model(name)
        param_grid = get_param_grid(name)
        best_model, best_params = perform_grid_search(model, X, y, param_grid, model_type=name)
        base_models[name] = best_model

        scores = cross_val_score(best_model, X, y, cv=inner_cv, scoring='accuracy', n_jobs=1)
        base_scores[name] = float(scores.mean())
    
    #  Create soft-voting classifier
    estimators = [(name, base_models[name]) for name in model_names]
    weights = [base_scores[name] for name in model_names]
    
    voting_model = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=1
    )
    
    # Final outer cv to evaluate the ensemble
    cv_scores = cross_val_score(voting_model, X, y, cv=outer_cv, scoring='accuracy')
    
    print(f"\nVoting Ensemble outer CV Score:", round(cv_scores.mean(), 4))
    print(f"Time taken:", round((time.time() - start_time), 2))
    
    # Final fit on full data
    voting_model.fit(X, y)
    return voting_model