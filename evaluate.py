from tensorflow import keras
import os
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    model = keras.models.load_model('models\\model')
    predicts = []
    for i in range(10):
        image = Image.open('test3\\' + str(i) + '.png').convert('L')
        data = 1 - np.asarray(image).reshape(1,28, 28, 1) / 255
        x_pre = model.predict(data)
        x_pre = np.argmax(x_pre, axis=1)
        predicts.append(int(np.squeeze(x_pre)))

    print('predicted: ' + str(predicts))
    print('excepted:  ' + str(list(range(10))))
    
    
if __name__ == '__main__':
    main()