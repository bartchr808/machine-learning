# All necessary imports
import numpy as np
from keras.models import sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K

seed = 7
numpy.random.seed(seed)


def model(weights = None, drop_rate = 0.0):
    model = Sequential()
    model.add(ZeroPadding2D(2,2), input_shape=(1, 48, 48))
    model.add(Conv2D(320, (3, 3), activation='PReLU'))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(320, (3, 3), activation='PReLU'))
    #model.add(Input(tensor=tf.nn.fractional_max_pool()))

    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(640, (3, 3), activation='PReLU'))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(640, (3, 3), activation='PReLU'))
    #model.add(Input(tensor=tf.nn.fractional_max_pool()))

    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(960, 3, 3, activation='PReLU'))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(960, 3, 3, activation='PReLU'))
    #model.add(Input(tensor=tf.nn.fractional_max_pool()))

    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(1280, 3, 3, activation='PReLU'))
    model.add(ZeroPadding2D(1,1))
    model.add(Conv2D(1280, 3, 3, activation='PReLU'))
    #model.add(Input(tensor=tf.nn.fractional_max_pool()))

    model.add(Flatten())
    model.add(Dense(1600, activation='PReLU'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1600, activation='PReLU'))
    model.add(Dropout(drop_rate))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if weights:
        model.load_weights(weights_path)

    return model

# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
