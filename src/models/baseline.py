from data.feature_engineering import feature_engineering
from sklearn.linear_model import LogisticRegression


def train_baseline_model(X, y):
    model = LogisticRegression(random_state=42, max_iter=1000)
    return model.fit(X, y)