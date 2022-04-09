from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import datetime

class Trainer:
    
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
    def train(self):
        model = Trainer.__create_model()
        model.summary()
        model.fit(self.train_dataset, epochs=7, validation_data=self.test_dataset)
        
        score = model.evaluate(self.test_dataset)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
    
    @staticmethod    
    def __create_model():
        model = keras.Sequential(
            [
            layers.Conv2D(16, (3, 3), input_shape=(28, 28, 1), activation='relu', kernel_initializer='he_uniform'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='relu'),
            layers.Dense(10, activation='softmax')
            ]
        )
        
        model.compile(loss='categorical_crossentropy', 
                optimizer='adam',  
                metrics=['accuracy'])
        
        return model