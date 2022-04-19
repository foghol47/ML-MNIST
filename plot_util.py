import matplotlib.pyplot as plt
import seaborn as sn

class PlotUtil():
    
    @staticmethod
    def plot_cm(cm, filename):
        figure= plt.figure(figsize=(10, 7))
        ax = sn.heatmap(cm, annot=True, fmt='d')
        ax.set_xlabel('predicted label')
        ax.set_ylabel('true label')
        plt.savefig(filename)