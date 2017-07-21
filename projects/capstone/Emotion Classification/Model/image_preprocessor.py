# Rough script for visualizing how the preprocesed data looks like
# Saves images in according folders

import os
from itertools import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# load and process data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 30,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

def process(filePath, newFilePath, fileName):
    img = load_img(filePath + '/' + fileName)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 48, 48)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 1, 48, 48)

    # the .flow() command below generates batches of randomly transformed images
    d = newFilePath + filePath[-1]
    for batch in datagen.flow(x, batch_size = 1, save_to_dir = d, save_prefix = b + '_' + fileName.rstrip('jpg').rstrip('.'), save_format='jpeg'):
        break
batch = ''.join(list(map(str, range(6))))

# For each subset of the data, save the preprocessed to folder
for b in batch:
    for filename in os.listdir('../Training/' + b):
        process('../Training/' + b, '../Training_P/', filename)

for b in batch:
    for filename in os.listdir('../PublicTest/' + b):
        process('../PublicTest/' + b, '../PublicTest_P/', filename)

for b in batch:
    for filename in os.listdir('../PrivateTest/' + b):
        process('../PrivateTest/' + b, '../PrivateTest_P/', filename)
