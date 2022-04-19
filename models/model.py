from tensorflow import keras
from tensorflow.keras import layers

class Model:
    
    @staticmethod
    def create_model():
        model = keras.Sequential(
            [
            layers.Conv2D(16, (4, 4), input_shape=(28, 28, 1), padding='same', activation='relu', kernel_initializer='he_uniform'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(10, activation='relu'),
            layers.Dense(10, activation='softmax')
            ]
        )
        
        model.compile(
                loss='categorical_crossentropy', 
                optimizer='adam',  
                metrics=['accuracy'])
        
        return model