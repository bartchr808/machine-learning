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
from batch_fscore import fbeta_score

seed = 7
np.random.seed(seed)

#def frac_max_pool(x):
#    return tf.nn.fractional_max_pool(x,p_ratio)[0]

def model(weights = None, S = 5, p_ratio = [1.0, 1.41, 1.41, 1.0]):
    model = Sequential()
    model.add(Dropout(0.07, input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(PReLU())
    #model.add(APLUnit(S=S))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(PReLU())
    #model.add(APLUnit(S=S))
    #model.add(Lambda(frac_max_pool))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
#    model.add(InputLayer(input_tensor = tf.nn.fractional_max_pool(model.layers[6].output, p_ratio, overlapping=True)[0]))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU())
#    model.add(APLUnit(S=S))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU())
 #   model.add(APLUnit(S=S))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
#    model.add(InputLayer(input_tensor = tf.nn.fractional_max_pool(model.layers[13].output, p_ratio, overlapping=True)[0]))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3))
    model.add(PReLU())
    #model.add(APLUnit(S=S))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3))
    model.add(PReLU())
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3))
    model.add(PReLU())
    #model.add(APLUnit(S=S))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
#    model.add(InputLayer(input_tensor = tf.nn.fractional_max_pool(model.layers[20].output, p_ratio, overlapping=True)[0]))
    
    model.add(Flatten())
    model.add(Dense(1024))
#    model.add(APLUnit(S=S))
    model.add(PReLU())
    model.add(Dropout(0.67))
    model.add(Dense(1024))
    model.add(PReLU())
    #model.add(APLUnit(S=S))
    model.add(Dropout(0.67))
    model.add(Dense(6, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', fbeta_score])

    if weights:
        model.load_weights(weights)

    return model

# build the model
model = model("VGG16_regular_first_try.h5")

batch_size = 256

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

# Fit the model
train_generator = train_datagen.flow_from_directory(
        '../Training',  # this is the target directory
        target_size = (48, 48),  # all images will be resized to 48x48
        batch_size = batch_size,
        color_mode = 'grayscale')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../PublicTest',
        target_size = (48, 48),
        batch_size = batch_size,
        color_mode = 'grayscale')

model.fit_generator(
        train_generator,
        steps_per_epoch = 28273 // batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps= 3533 // batch_size)
model.save_weights('VGG16_regular_first_try.h5')  # always save your weights after training or during training
