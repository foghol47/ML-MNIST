import idx2numpy
import tensorflow as tf

class DataUtil:
    
    @staticmethod
    def load_MNIST():
        test_images = idx2numpy.convert_from_file('Dataset/t10k-images.idx3-ubyte')
        test_labels = idx2numpy.convert_from_file('Dataset/t10k-labels.idx1-ubyte')
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        
        train_images = idx2numpy.convert_from_file('Dataset/train-images.idx3-ubyte')
        train_labels = idx2numpy.convert_from_file('Dataset/train-labels.idx1-ubyte')
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        
        return (train_dataset, test_dataset)
        