from data.feature_engineering import feature_engineering
import LogisticRegression


def train_baseline_model(X, y):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)