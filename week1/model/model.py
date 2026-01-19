import joblib

class TrainedModel:
    def __init__(self, path):
        self.model = joblib.load(path)

    def predict(self, X):
        return self.model.predict(X)
