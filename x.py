from numpy.random import randint

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

# for directory_path_Train in glob.glob('C:/Users/roshd/IdeaProjects/object/Train/*'):
#     label = directory_path_Train.split("\\")[-1]
#     print(label)
#     for img_path in glob.glob(os.path.join(directory_path_Train, "*.jpg")):
#         print(img_path)
#
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (SIZE, SIZE))
#         imgRF = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         imgDT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         train_imagesRF.append(imgRF)
#         train_imagesDT.append(imgDT)
#         train_labels.append(label)
#
# train_imagesDT = np.array(train_imagesDT)
# train_imagesRF = np.array(train_imagesRF)
# train_labels = np.array(train_labels)
#
# #####################
#
# test_imagesDT = []
# test_imagesRF = []
# test_labels = []
#
# for directory_path in glob.glob("C:/Users/roshd/IdeaProjects/object/Test/*"):
#
#     fruitLabel = directory_path.split("\\")[-1]
#     for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#         print(img_path)
#
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (SIZE, SIZE))
#         imgRF = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         imgDT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         test_imagesRF.append(imgRF)
#         test_imagesDT.append(imgDT)
#         test_labels.append(fruitLabel)
#
# test_imagesDT = np.array(test_imagesDT)
# test_imagesRF = np.array(test_imagesRF)
# test_labels = np.array(test_labels)
#
# ################################
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

################################

x_trainDT, x_trainRF, y_train, x_testDT, x_testRF, y_test = train_imagesDT, train_imagesRF, train_labels_encoded, \
                                                            test_imagesDT, test_imagesRF, test_labels_encoded

x_trainRF, x_testRF = x_trainRF / 255.0, x_testRF / 255.0
x_trainDT, x_testDT = x_trainDT / 255.0, x_testDT / 255.0

###############################

n_samples_train, nx_train, ny_train = x_trainDT.shape
d2_x_dataset_train = x_trainDT.reshape((n_samples_train, nx_train * ny_train))

n_samples_test, nx_test, ny_test = x_testDT.shape
d2_x_dataset_test = x_testDT.reshape((n_samples_test, nx_test * ny_test))
###############################

from keras.utils.np_utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

################################

from sklearn.tree import DecisionTreeClassifier

x_DT_model = DecisionTreeClassifier(criterion='entropy', max_depth=50, random_state=42)
x_DT_model.fit(d2_x_dataset_train, y_train)
prediction_DT = x_DT_model.predict(d2_x_dataset_test)
prediction_DT = le.inverse_transform(prediction_DT)

################################

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from numpy import mean
from numpy import std

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_DT)
print('Confusion matrix\n\n', cm)

# for n in range(0, 12):
#     img = d2_x_dataset_test[n]
#     plt.imshow(x_test[n])
#     plt.show()
#
#     input_img = np.expand_dims(img, axis=0)
#     input_img_features = x_DT_model.predict(input_img)
#
#     prediction_DT = x_DT_model.predict(input_img)[0]
#
#     prediction_DT = le.inverse_transform([prediction_DT])
#     print("I think this image is : ", prediction_DT)
#     print("The actual label for this image is: ", test_labels[n])
#     print("ارجوك متشتمنيش لو انا غلط انا عبيط والله")

################################

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

feature_extractor = VGG_model.predict(x_trainRF)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features

################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std

RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

RF_model.fit(X_for_RF, y_train)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(RF_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
x_test_feature = VGG_model.predict(x_testRF)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)

prediction_RF = RF_model.predict(x_test_features)
prediction_RF = le.inverse_transform(prediction_RF)

################################

from sklearn import metrics

print('Accuracy for Random forest = ', metrics.accuracy_score(test_labels, prediction_RF))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)

print('Confusion matrix\n\n', cm)
################################

from tkinter import *
from tkinter import messagebox
from numpy.random import randint
from PIL import Image, ImageTk
import matplotlib

matplotlib.use("TkAgg")

app = Tk()
app.title("welcome to Project")
app.geometry("500x500+150+150")

from sklearn import metrics


def DecisionTreeAccuracy():
    messagebox.showinfo("Accuracy for Decision Tree. \n", str(metrics.accuracy_score(test_labels, prediction_DT)))


btn1 = Button(app, text="DecisionTree", width=80, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=DecisionTreeAccuracy)
btn1.pack()


def RandomForestAccuracy():
    messagebox.showinfo("Accuracy for Random Forests.\n", str(metrics.accuracy_score(test_labels, prediction_RF)))


btn2 = Button(app, text="Random Forest", width=80, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=RandomForestAccuracy)
btn2.pack()


def RandomForest():
    app4 = Tk()
    app4.geometry("600x600")

    n = randint(0, len(x_testRF))
    img = x_testRF[n]

    input_img = np.expand_dims(img, axis=0)
    input_img_features = VGG_model.predict(input_img)
    input_img_features = input_img_features.reshape(input_img_features.shape[0], -1)

    prediction_RF = RF_model.predict(input_img_features)[0]
    prediction_RF = le.inverse_transform([prediction_RF])
    image = Image.fromarray(test_imagesRF[n])
    img = ImageTk.PhotoImage(image=image, master=app4)
    image = Label(app4, text=("The prediction for this image is: ", prediction_RF), image=img)
    image.config(compound='bottom')
    print("The actual label for this image is: ", test_labels[n])
    image.pack()
    app4.mainloop()


btn5 = Button(app, text="RF", width=80, height=3, bg="#a09f9f", fg="white", borderwidth=0, command=RandomForest)
btn5.pack()


def DecisionTree():
    app5 = Tk()
    app5.geometry("600x600")

    n = randint(0, len(x_testRF))
    img = d2_x_dataset_test[n]
    input_img = np.expand_dims(img, axis=0)
    input_img_features = x_DT_model.predict(input_img)

    prediction_DT = x_DT_model.predict(input_img)[0]

    prediction_DT = le.inverse_transform([prediction_DT])
    image = Image.fromarray(test_imagesDT[n])
    img = ImageTk.PhotoImage(image=image, master=app5)
    image = Label(app5, text=("The prediction for this image is: ", prediction_DT), image=img)
    image.config(compound='bottom')
    image.pack()
    app5.mainloop()


btn6 = Button(app, text="DT", width=80, height=3, bg="#a09f9f", fg="white", borderwidth=0, command=DecisionTree)
btn6.pack()


def HeatMapRF():
    cm = confusion_matrix(test_labels, prediction_RF)
    sns.heatmap(cm, annot=True)
    plt.show()


btn7 = Button(app, text="Heat Map For Random Forest", width=80, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=HeatMapRF)
btn7.pack()


def HeatMapDT():
    cm = confusion_matrix(test_labels, prediction_DT)
    sns.heatmap(cm, annot=True)
    plt.show()


btn8 = Button(app, text="Heat Map For Decision Tree", width=80, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=HeatMapDT)
btn8.pack()


def DTAccuracyScores():
    X, y = d2_x_dataset_train, y_train

    max_depth_list = []
    for i in range(1, 51):
        max_depth_list.append(i)
    train_errors = []  # Log training errors for each model
    test_errors = []  # Log testing errors for each model

    train_score, test_score = validation_curve(KNeighborsClassifier(), X, y,
                                               param_name="n_neighbors",
                                               param_range=max_depth_list,
                                               cv=5, scoring="accuracy")

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(max_depth_list, mean_train_score,
             label="Training Score", color='b')
    plt.plot(max_depth_list, mean_test_score,
             label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title("Validation Curve with KNN Classifier")
    plt.xlabel("Number of Neighbours")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


btn9 = Button(app, text="Validation Curve with KNN Classifier For Decision Tree", width=80, height=3, bg="#a09f9f",
              fg="white", borderwidth=0,
              command=DTAccuracyScores)
btn9.pack()

def RFAccuracyScores():
    from sklearn.linear_model import Ridge
    X, y = train_imagesRF, train_labels

    max_depth_list = []
    for i in range(1, 51):
        max_depth_list.append(i)
    train_errors = []  # Log training errors for each model
    test_errors = []  # Log testing errors for each model

    train_score, test_score = validation_curve(RF_model, X, y


                                               ,
                                               param_name="n_estimators",
                                               param_range=max_depth_list,
                                               cv=3, scoring="accuracy",
                                               n_jobs=-1)

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(max_depth_list, mean_train_score,
             label="Training Score", color='b')
    plt.plot(max_depth_list, mean_test_score,
             label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title("Validation Curve with KNN Classifier")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


btn10 = Button(app, text="Validation Curve with KNN Classifier for Random Forest", width=80, height=3, bg="#a09f9f",
               fg="white", borderwidth=0,
               command=RFAccuracyScores)
btn10.pack()

# def RFAccuracyScores2():
#     from sklearn.linear_model import Ridge
#     X, y = x_testRF, y_test
#
#     max_depth_list = []
#     for i in range(1, 51):
#         max_depth_list.append(i)
#     train_errors = [] # Log training errors for each model
#     test_errors = [] # Log testing errors for each model
#
#
#     train_score, test_score = validation_curve(Ridge(), X, y,
#                                                param_name = "alpha",
#                                                param_range = max_depth_list,
#                                                cv = 5, scoring = "accuracy")
#
#     # Calculating mean and standard deviation of training score
#     mean_train_score = np.mean(train_score, axis = 1)
#     std_train_score = np.std(train_score, axis = 1)
#
#     # Calculating mean and standard deviation of testing score
#     mean_test_score = np.mean(test_score, axis = 1)
#     std_test_score = np.std(test_score, axis = 1)
#
#     # Plot mean accuracy scores for training and testing scores
#     plt.plot(max_depth_list, mean_train_score,
#              label = "Training Score", color = 'b')
#     plt.plot(max_depth_list, mean_test_score,
#              label = "Cross Validation Score", color = 'g')
#
#     # Creating the plot
#     plt.title("Validation Curve with KNN Classifier")
#     plt.xlabel("Number of Neighbours")
#     plt.ylabel("Accuracy")
#     plt.tight_layout()
#     plt.legend(loc = 'best')
#     plt.show()
#
# btn11 = Button(app, text="Validation Curve with KNN Classifier for Random Forest", width=80, height=3, bg="#a09f9f", fg="white", borderwidth=0,
#                command=RFAccuracyScores2)
# btn11.pack()


app.mainloop()
