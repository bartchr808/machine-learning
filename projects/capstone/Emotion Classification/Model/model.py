# All necessary imports
import numpy as np
import h5py
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import PReLU
import tensorflow as tf

seed = 7
np.random.seed(seed)


def model(weights = None, drop_rate = .5, p_ratio=[1.0, 1.44, 1.44, 1.0]):
    model = Sequential()
    model.add(ZeroPadding2D((2,2), input_shape=(48, 48, 3)))
    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU())
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3)))
    model.add(PReLU())
    model.add(InputLayer(input_tensor=tf.nn.fractional_max_pool(model.layers[3].output, p_ratio)[0]))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(PReLU())
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3)))
    model.add(PReLU())
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3))
    model.add(PReLU())
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3))
    model.add(PReLU())
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3))
    model.add(PReLU())
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3))
    model.add(PReLU())
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(900, activation=PReLU()))
    model.add(Dropout(drop_rate))
    model.add(Dense(900, activation=PReLU()))
    model.add(Dropout(drop_rate))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if weights:
        model.load_weights(weights)

    return model

# build the model
model = model()

batch_size = 5

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Fit the model
train_generator = train_datagen.flow_from_directory(
        '../Training',  # this is the target directory
        target_size=(48, 48),  # all images will be resized to 48x48
        batch_size=batch_size)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../PublicTest',
        target_size=(48, 48),
        batch_size=batch_size)

model.fit_generator(
        train_generator,
        steps_per_epoch=20,  #28709 // batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=20) #3589 // batch_size)
#model.save_weights('first_try.h5')  # always save your weights after training or during training
