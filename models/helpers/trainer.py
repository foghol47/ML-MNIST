from tensorflow import keras
import tensorflow as tf
import numpy as np
from utils.log_utils import LogUtils

class Trainer:
    
    def __init__(self, train_dataset, test_dataset, model):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        
    def train(self):
        self.model.summary()
        tensorboard_callback = LogUtils.get_tensorboard_callback()
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20)
        self.model.fit(self.train_dataset, epochs=50, verbose=2, validation_data=self.test_dataset, callbacks=[tensorboard_callback, early_stopping_callback])
        self.model.save('saved_models\\model')

    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred