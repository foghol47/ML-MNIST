from data_util import DataUtil
from trainer import Trainer
import seaborn as sn
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from model import Model
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def plot_cm(cm, filename):
    figure= plt.figure(figsize=(10, 7))
    ax = sn.heatmap(cm, annot=True)
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    plt.savefig(filename)


def main():

    train_dataset, test_dataset = DataUtil.load_MNIST()
    model = Model.create_model()
    trainer = Trainer(train_dataset, test_dataset, model)
    
    y_label = DataUtil.get_ytest_labels()
    y_pred = trainer.predict(test_dataset)
    cm = confusion_matrix(y_label, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    
    plot_cm(cm, "before_train.png")

    trainer.train()
    
    y_pred = trainer.predict(test_dataset)
    cm = confusion_matrix(y_label, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    
    plot_cm(cm, "after_train.png")


if __name__ == '__main__':
    main()
    