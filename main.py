import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import cv2
import os
import numpy as np
import keras

model = Sequential()

def loadImages():
    imageList=[]
    labelList=[]

    rootdir = r'C:\Users\charles\Desktop\20180222\20180222\chair\images'
    list = os.listdir(rootdir)
    for item in list:
        path = os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f = cv2.imread(path)
            f = cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(0)#類別0

    rootdir = r'C:\Users\charles\Desktop\20180222\20180222\table\images'
    list = os.listdir(rootdir)
    for item in list:
        path = os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f = cv2.imread(path)
            f = cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(1)#類別1
    return np.asarray(imageList), keras.utils.to_categorical(labelList, 2)

def Net_model(nb_classes, lr = 0.001,decay=1e-6,momentum=0.9):
    model.add(Convolution2D(filters = 10, kernel_size = (5, 5),
                            padding = 'valid',
                            input_shape = (448, 448, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Convolution2D(filters = 20, kernel_size = (10, 10)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = lr, decay = decay, momentum = momentum, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer = sgd)

    return model

nb_classes = 2
nb_epoch = 30
nb_step = 6
batch_size = 64

x,y = loadImages()

from keras.preprocessing.image import ImageDataGenerator
dataGenerator = ImageDataGenerator()
dataGenerator.fit(x)
data_generator = dataGenerator.flow(x, y, batch_size, True) #generator函數，用來生成批處理數據

model = Net_model(nb_classes = nb_classes, lr = 0.0001) #加載網絡模型

history = model.fit_generator(data_generator, epochs = nb_epoch, steps_per_epoch = nb_step, shuffle = True) #訓練網絡

model.save_weights('C:\\Users\\charles\\Desktop\\trained_model_weights.h5')
print("DONE, model saved in path")
