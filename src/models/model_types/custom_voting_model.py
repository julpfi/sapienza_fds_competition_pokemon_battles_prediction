import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.config import SEED
from .. import tune
from .models import get_model, get_param_grid


class CustomVotingClassifier(BaseEstimator, ClassifierMixin):    
    '''
    Background of this Custom Implementation: 
        This CustomVotingClassifier was implemented based on the logic of the Sklearn VotingClassifer. The issue is that sklearn 
        might contain a bug (at least we had issue) to implement XGBoost within the VotingClassifier. The resulting error arose inside 
        the sklearn pipeline/model handling and states: "ValueError: The estimator XGBClassifier should be a classifier". Such error
        is also documented on other classifier on Github and Stackoverflow and any suggested fixes did not work. At the time, we 
        focussed on having XGBoost as one of the main weak learners in the voting so we decided on a simple local replica. Later,
        we also implemented the HistGradientBoostingClassifier which as the "LightGBM-sklearn-version" works with the normal voting classifier.
    '''

    def __init__(self, estimators):
        self.estimators = estimators
        self.weights = None
        
    def fit(self, X, y):
        check_X_y(X, y)
        
        self.estimators_ = []
        self.classes_ = np.unique(y)
        
        for name, estimator in self.estimators:
            # Clone and fit each estimator on the full training data
            fitted_estimator = clone(estimator).fit(X, y)
            self.estimators_.append((name, fitted_estimator))
            
        return self

    def predict_proba(self, X):
        check_is_fitted(self) 
        all_probas = [estimator.predict_proba(X) for name, estimator in self.estimators_]
        return np.average(all_probas, axis=0, weights=self.weights)
        

    def predict(self, X):
        # Uses soft voting => accounts for predicted probbility     
        probas = self.predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        
        return self.classes_[predicted_indices]


def get_estimators(X, y, model_names, grid_search=True):
    estimators = []
    for name in model_names:
        if grid_search:
            print(f"\n------ Tuning {name} ------ ")
            _, best_params = tune.perform_grid_search(
                get_model(name), X, y, get_param_grid(name), model_type=name
            )
        else:
            print(f"\n------ Using base {name} model ------ ")
            best_params = {}

        model = get_model(name)
        model.set_params(**best_params)
        estimators.append((name, model))
        
    return estimators


def train_voting(X: pd.DataFrame, y: pd.Series, model_names:list[str], grid_search: bool = True) -> CustomVotingClassifier:
    # Get Weaker Base Models 
    estimators = get_estimators(X, y, model_names=model_names, grid_search=grid_search)
    
    # Get ensemble with custom voting classifier
    voting_model = CustomVotingClassifier(estimators=estimators)
    
    # Evaluate the voting classifier with cv
    print("\n------ Evaluating full ensemble with cross-validation ------ ")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # cross_val_score clones the unfitted model for each fold
    cv_scores = cross_val_score(
        voting_model, 
        X, y, 
        cv=kf, 
        scoring='accuracy',
        n_jobs=1 
    )
    print(f"Mean Ensemble CV Accuracy: {round(cv_scores.mean(), 4)}")

    # Fit final voting model
    print("\n------ Fitting final voting model ------ ")
    voting_model.fit(X, y) 
    
    print("Voting ensemble training complete")
    return voting_model