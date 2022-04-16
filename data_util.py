import idx2numpy
import tensorflow as tf
from tensorflow import keras
import numpy as np

class DataUtil:
    
    @staticmethod
    def load_MNIST(normalize=True):
        num_classes = 10
        
        x_test = idx2numpy.convert_from_file('Dataset/t10k-images.idx3-ubyte').astype(np.float32).reshape(10000, 28, 28, 1)
        y_test = idx2numpy.convert_from_file('Dataset/t10k-labels.idx1-ubyte').reshape(10000, 1)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        x_train = idx2numpy.convert_from_file('Dataset/train-images.idx3-ubyte').astype(np.float32).reshape(60000, 28, 28, 1)
        y_train = idx2numpy.convert_from_file('Dataset/train-labels.idx1-ubyte').reshape(60000, 1)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        
        if normalize:
            x_test = x_test / 255
            x_train = x_train / 255
 
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(64).batch(64)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(64).batch(64)
        
        return (train_dataset, test_dataset)
       
    @staticmethod   
    def get_ytest_labels():
        y_test = idx2numpy.convert_from_file('Dataset/t10k-labels.idx1-ubyte')
        return y_test