import matplotlib.pyplot as plt
import seaborn as sn
import os

class PlotUtil():
    
    @staticmethod
    def plot_cm(cm, filename):
        figure= plt.figure(figsize=(10, 7))
        ax = sn.heatmap(cm, annot=True, fmt='d')
        ax.set_xlabel('predicted label')
        ax.set_ylabel('true label')
        
        path = os.getcwd() + '//pics//'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + filename)