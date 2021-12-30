import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns

print(os.listdir("C:/Users/roshd/IdeaProjects/object/"))

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization, MaxPooling2D, Flatten, Conv2D, Dense
)

SIZE = 256


def readFiles(directoryPath, imagesRF, imagesDT, labels):
    for directory_path_Train in glob.glob(directoryPath):
        label = directory_path_Train.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path_Train, "*.jpg")):
            print(img_path)

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            imgRF = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgDT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imagesRF.append(imgRF)
            imagesDT.append(imgDT)
            labels.append(label)


train_imagesDT = []
train_imagesRF = []
train_labels = []

test_imagesDT = []
test_imagesRF = []
test_labels = []

readFiles('C:/Users/roshd/IdeaProjects/object/Train/*', train_imagesRF, train_imagesDT, train_labels)

readFiles('C:/Users/roshd/IdeaProjects/object/Test/*', test_imagesRF, test_imagesDT, test_labels)
train_imagesDT = np.array(train_imagesDT)
train_imagesRF = np.array(train_imagesRF)
train_labels = np.array(train_labels)

test_imagesDT = np.array(test_imagesDT)
test_imagesRF = np.array(test_imagesRF)
test_labels = np.array(test_labels)

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

root = tk.Tk()

img =  ImageTk.PhotoImage(image=Image.fromarray(test_imagesRF[0]))

canvas = tk.Canvas(root,width=300,height=300)
canvas.pack()
canvas.create_image(20,20, anchor="nw", image=img)

root.mainloop()