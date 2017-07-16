import os
import time
import cv2
import numpy as np
from PIL import Image
from webcam_CNN import prediction
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

vidcap = cv2.VideoCapture(0)

def crop_center(img, cropx, cropy):
    y,x, z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

while (True):
    retval, image = vidcap.read()
    #vidcap.open(0)
    #retval, image = vidcap.retrieve()
    #vidcap.release()
    #cv2.imwrite("images/samples/test.png", image)

    #image_file = Image.open("images/samples/test.png") # open colour image
    #img = image_file.convert('L')

    #print(np.array(image).shape)
    width, height, depth = np.array(image).shape   # Get dimensions

    new_width, new_height = (650, 650) # 1280 × 720

    img = crop_center(image, 650, 650)
    #img.save("images/samples/test.png")

    #i = np.array(image)
    print("----")
    print(np.array(Image.open('./images/samples/test.png')).shape)
    #i = np.resize(img, [1, 48, 48, 1])
    i = rgb2gray(img)
    i = np.resize(i, [1, 48, 48, 1])
    print(i.shape)

    pred_array = prediction(i)[0]
    print(pred_array)

    print("Angry: ", pred_array[0], "\nFear: ", pred_array[1], "\nHappy: ", pred_array[2], "\nSad: ", pred_array[3], "\nSurprise: ", pred_array[4], "\nNeutral: ", pred_array[5])

    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.22, 0.72, 0.15, 0.15]
    ax2 = fig.add_axes([left, bottom, width, height])
    #img = mpimg.imread('images/samples/test.png')

    plt.tick_params(
    axis='both',
    which='both',
    labelleft='off',
    labelbottom='off')

    ax1.tick_params(
    axis='both',
    which='both',
    labelleft='off',
    labelbottom='off')

    ax1.imshow(img)

    Emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    y_pos = np.arange(len(Emotions))
    ax2.bar(y_pos, pred_array, align='center', alpha=0.5)
    ax2.set_xticks(y_pos, Emotions)
    #ax2.plot(pred_array, color='green')

    plt.pause(0.00001)


    """pred_array = prediction("./images")

    print("Angry: ", pred_array[0][0], "\nFear: ", pred_array[0][1], "\nHappy: ", pred_array[0][2], "\nSad: ", pred_array[0][3], "\nSurprise: ", pred_array[0][4], "\nNeutral: ", pred_array[0][5])"""
    #os.remove("images/samples/test.png")
    print("File Removed!")
