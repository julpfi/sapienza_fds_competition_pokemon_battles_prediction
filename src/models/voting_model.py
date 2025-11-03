import pandas as pd
from . import tune
from .models import get_model, get_param_grid
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.config import SEED
from sklearn.pipeline import Pipeline



def evaluate_voting_cv(estimators: list, X: pd.DataFrame, y: pd.Series):
    # Evaluates the Voting classifier using cross-validation
    voting_model_cv = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    cv_scores = cross_val_score(
        estimator=voting_model_cv, 
        X=X, 
        y=y, 
        cv=kf, 
        scoring='accuracy',
        n_jobs=1 
    )
    
    print(f"Mean CV Accuracy: {round(cv_scores.mean(), 4)}")


def train_voting(
    X: pd.DataFrame, 
    y: pd.Series, 
    grid_search: bool = True,
    model_names: list[str] = ['logistic', 'random_forest'] 
) -> VotingClassifier:
    
    estimators = []

    for name in model_names:
        if grid_search:
            print(f"\n------ Voting model: Tune for {name} ------ ")
            # Only the best parameters are needed (voting classifier needs unfitted models)
            _, best_params = tune.perform_grid_search(
                get_model(name), X, y, get_param_grid(name), model_type=name
            )
        else:
            best_params = {}


        model = get_model(name)
        model.set_params(**best_params)

        estimators.append((name, model))

    print("\n------ Evaluating voting model with cross-validation ------ ")

    evaluate_voting_cv(estimators=estimators, X=X, y=y)
    
    print("\n------ Fitting final voting model ------ ")

    voting_model = VotingClassifier(
        estimators=estimators, 
        voting='soft',
        n_jobs=-1
    )
    
    voting_model.fit(X, y)
    print("Voting ensemble training complete")
    return voting_model