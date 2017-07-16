# All necessary imports
import numpy as np
import h5py
from APL import APLUnit
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

seed = 7
np.random.seed(seed)

#def frac_max_pool(x):
#    return tf.nn.fractional_max_pool(x,p_ratio)[0]

def model(weights = None, S = 5, p_ratio = [1.0, 2.6, 2.6, 1.0]):

    model = Sequential()

    model.add(Dropout(0.0, input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (5, 5)))
    model.add(APLUnit(S=S))
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (5, 5)))
    model.add(APLUnit(S=S))
    model.add(InputLayer(input_tensor = tf.nn.fractional_max_pool(model.layers[7].output, p_ratio, overlapping=True)[0]))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (4, 4)))
    model.add(APLUnit(S=S))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(4096))
    model.add(APLUnit(S=S))
    model.add(Dense(6, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    if weights:
        model.load_weights(weights)

    return model

# build the model
model = model('../VGG16_regular_ninth_try2_PRIVATE_TEST.h5')

test_datagen = ImageDataGenerator(rescale = 1./255)

def prediction(img):
    #prediction_generator = test_datagen.flow_from_directory(
    #        img,
    #        target_size = (48, 48),
    #        color_mode = 'grayscale')
    prediction_generator = test_datagen.flow(img, [1])
    #return model.predict_proba(img)

    return model.predict_generator(prediction_generator, 1)
