from tensorflow import keras
import os
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    model_path = '..\\..\\saved_models\\model'
    test_path = '..\\..\\tests\\'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError('model not found. you need train a model and save it')
        
    if not os.path.exists(test_path):
        raise FileNotFoundError('test files not found')
        
    model = keras.models.load_model(model_path)
    predicts = []
    for i in range(10):
        image = Image.open(test_path + 'test3\\' + str(i) + '.png').convert('L')
        data = 1 - np.asarray(image).reshape(1,28, 28, 1) / 255
        x_pre = model.predict(data)
        x_pre = np.argmax(x_pre, axis=1)
        predicts.append(int(np.squeeze(x_pre)))

    print('predicted: ' + str(predicts))
    print('excepted:  ' + str(list(range(10))))
    
    
if __name__ == '__main__':
    main()