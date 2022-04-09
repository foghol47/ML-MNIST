from data_util import DataUtil
from trainer import Trainer
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    train_dataset, test_dataset = DataUtil.load_MNIST()
    
    trainer = Trainer(train_dataset, test_dataset)
    trainer.train()
    

if __name__ == '__main__':
    main()
    