from src.utils.model_evaluation import evaluate_classification
from sklearn.model_selection import StratifiedKFold
from . import baseline as models
import numpy as np


def cross_validate_model(X, y, model_type, n_splits=5, random_state=42, **model_kwargs):
    # Could also use simple KFold but after research StratifiedKFold works the same and is just more robust
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for (train_x, val_x) in folds.split(X, y):
        X_train, X_test = X.iloc[train_x], X.iloc[val_x]
        y_train, y_test = y.iloc[train_x], y.iloc[val_x]
        
        model = models.get_model(model_type=model_type, **model_kwargs)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = evaluate_classification(y_test, pred)
        results.append(metrics["accuracy"])

    accuracy_all = np.mean(results)
    print("Average CV accuracy:" , round(accuracy_all, 4))
    return accuracy_all



def train_baseline(X, y):
    
    cross_validate_model(X, y, "random_forest")

    model = models.get_model("random_forest")

    return model.fit(X, y)