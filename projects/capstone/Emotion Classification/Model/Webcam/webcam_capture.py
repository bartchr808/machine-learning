import os
import time
import cv2
from PIL import Image
#from webcam_CNN import prediction
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

vidcap = cv2.VideoCapture(0)

while (True):
    retval, image = vidcap.read()
    #vidcap.open(0)
    #retval, image = vidcap.retrieve()
    #vidcap.release()
    cv2.imwrite("images/samples/test.png", image)

    image_file = Image.open("images/samples/test.png") # open colour image
    img = image_file.convert('LA')

    width, height = img.size   # Get dimensions

    new_width, new_height = (650, 650) # 1280 × 720

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    img = img.crop((left, top, right, bottom))
    img.save("images/samples/test.png")

    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.22, 0.72, 0.15, 0.15]
    ax2 = fig.add_axes([left, bottom, width, height])
    img = mpimg.imread('images/samples/test.png')

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
    ax2.plot(range(6)[::-1], color='green')
    plt.pause(0.00001)


    """pred_array = prediction("./images")

    print("Angry: ", pred_array[0][0], "\nFear: ", pred_array[0][1], "\nHappy: ", pred_array[0][2], "\nSad: ", pred_array[0][3], "\nSurprise: ", pred_array[0][4], "\nNeutral: ", pred_array[0][5])"""
    os.remove("images/samples/test.png")
    print("File Removed!")
