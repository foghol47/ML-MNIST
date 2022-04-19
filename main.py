from utils.data_util import DataUtil
from models.helpers.trainer import Trainer
from utils.plot_util import PlotUtil
import os
from models.model import Model
from sklearn.metrics import confusion_matrix


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    train_dataset, test_dataset = DataUtil.load_MNIST()
    model = Model.create_model()
    trainer = Trainer(train_dataset, test_dataset, model)
    
    y_label = DataUtil.get_ytest_labels()
    y_pred = trainer.predict(test_dataset)
    cm = confusion_matrix(y_label, y_pred)
    
    PlotUtil.plot_cm(cm, "before_train.png")
    
    trainer.train()
    
    y_pred = trainer.predict(test_dataset)
    cm = confusion_matrix(y_label, y_pred)
    
    PlotUtil.plot_cm(cm, "after_train.png")


if __name__ == '__main__':
    main()
    