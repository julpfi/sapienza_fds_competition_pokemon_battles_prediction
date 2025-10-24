import baseline as models


def train_baseline(X, y):
    model = models.get_model("logistic")
    return model.fit(X, y)