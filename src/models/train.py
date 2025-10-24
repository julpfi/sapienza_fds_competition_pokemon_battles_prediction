from . import baseline as models


def train_baseline(X, y):
    model = models.get_model("random_forest")
    return model.fit(X, y)