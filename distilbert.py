import ktrain
from ktrain import text
from sklearn.metrics import roc_auc_score

MODEL_NAME = "distilbert-base-uncased"


class DistilBertTrain:
    def __init__(self, x_train, y_train):
        self.t = text.Transformer(
            MODEL_NAME, maxlen=100, classes=[True, False]
        )
        trn = self.t.preprocess_train(x_train, y_train)
        model = self.t.get_classifier()
        self.learner = ktrain.get_learner(model, train_data=trn, batch_size=64)

    def fit(self, epochs=2):
        self.learner.fit_onecycle(5e-6, epochs)
        self.predictor = ktrain.get_predictor(
            self.learner.model, preproc=self.t
        )

    def predict(self, x_test):
        y_pred = self.predictor.predict(x_test)
        return y_pred

    def save(self, filename):
        self.predictor.save(filename)

