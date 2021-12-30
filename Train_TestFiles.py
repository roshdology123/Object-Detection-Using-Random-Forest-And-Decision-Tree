import cv2
import glob
import numpy as np
import os

SIZE = 256


class Loading:
    def LoadImages(TrainingPath, train_imagesDT, train_imagesRF, train_labels):
        print(TrainingPath)
        for directory_path_Train in glob.glob(TrainingPath):
            label = directory_path_Train.split("\\")[-1]
            print(label)
            for img_path in glob.glob(os.path.join(directory_path_Train, "*.jpg")):
                print(img_path)

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE))
                imgRF = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                imgDT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                train_imagesRF.append(imgRF)
                train_imagesDT.append(imgDT)
                train_labels.append(label)

        train_imagesDT = np.array(train_imagesDT)
        train_imagesRF = np.array(train_imagesRF)
        train_labels = np.array(train_labels)
n = randint(0, 100, 1)
img = x_testRF[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0)
input_img_features = VGG_model.predict(input_img)
input_img_features = input_img_features.reshape(input_img_features.shape[0], -1)

prediction_RF = RF_model.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF])
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
from numpy.random import randint

from tkinter import *
from tkinter import messagebox
from numpy.random import randint


app = Tk()
app.title("welcome to Project")
app.geometry("500x500+150+150")

from sklearn import metrics


def DecisionTree():
    messagebox.showinfo("Accuracy for Decision Tree. \n", str(metrics.accuracy_score(test_labels, prediction_DT)))


btn1 = Button(app, text="DecisionTree", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=DecisionTree)
btn1.pack()


def RandomForests():
    messagebox.showinfo("Accuracy for Random Forests.\n", str(metrics.accuracy_score(test_labels, prediction_RF)))


btn2 = Button(app, text="Random Forest", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=RandomForests)
btn2.pack()
def haha():
    app4 = Tk()
    app4.geometry("600x600")
    n = randint(0, len(x_testRF))
    img = x_testRF[n]
    img = PhotoImage(img)


    image = Label (app4,text= print("The prediction for this image is: ", prediction_RF))
    z = Label(app4,image = img).pack()

    print("The actual label for this image is: ", test_labels[n])
    image.pack()
    app4.mainloop()

btn5 = Button (app,text="RF",width= 20,height=3,bg="#a09f9f",fg="white",borderwidth=0,command=haha)
btn5.pack()
app.mainloop()
