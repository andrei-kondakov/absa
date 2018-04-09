from time import time
from keras.callbacks import Callback


class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(time() - self.start_time)
