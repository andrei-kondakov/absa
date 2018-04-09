from time import time

from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(time() - self.start_time)


def get_metrics(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
