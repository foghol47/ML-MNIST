import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime


class LogUtils:
    
    @staticmethod    
    def get_tensorboard_callback():
        log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(log_dir)
        file_writer.set_as_default()
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 
        return tensorboard_callback
        