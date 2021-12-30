import matplotlib
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns

print(os.listdir("C:/Users/roshd/IdeaProjects/object/"))

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential

SIZE = 256

train_imagesDT = []
train_imagesRF = []
train_labels = []

for directory_path_Train in glob.glob('C:/Users/roshd/IdeaProjects/object/Train/*'):
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

#####################

test_imagesDT = []
test_imagesRF = []
test_labels = []

for directory_path in glob.glob("C:/Users/roshd/IdeaProjects/object/Test/*"):

    fruitLabel = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        imgRF = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgDT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        test_imagesRF.append(imgRF)
        test_imagesDT.append(imgDT)
        test_labels.append(fruitLabel)

test_imagesDT = np.array(test_imagesDT)
test_imagesRF = np.array(test_imagesRF)
test_labels = np.array(test_labels)

################################ label encoding & fitting بتغير الكلمات لارقام ##########################

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

############################### 3d arrey to 2d arrey to train and test dtmodel ###############

n_samples_train, nx_train, ny_train = x_trainDT.shape
d2_x_dataset_train = x_trainDT.reshape((n_samples_train, nx_train * ny_train))

n_samples_test, nx_test, ny_test = x_testDT.shape
d2_x_dataset_test = x_testDT.reshape((n_samples_test, nx_test * ny_test))
############################### categorizing train and test labels for rf using one hot encoding ##############

from keras.utils.np_utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

############################## declaring dt classifier using entropy criterion with max_depth50 & rs42 then traing and fittting the moudule 
# and inversing transform labels to its orginal####

from sklearn.tree import DecisionTreeClassifier

x_DT_model = DecisionTreeClassifier(criterion='entropy', max_depth=50, random_state=42)
x_DT_model.fit(d2_x_dataset_train, y_train)
prediction_DT = x_DT_model.predict(d2_x_dataset_test)
prediction_DT = le.inverse_transform(prediction_DT)
################################ calculate accuracy for dt ########### 

from sklearn import metrics

print("Accuracy for Decision Tree = ", metrics.accuracy_score(test_labels, prediction_DT))

####### to bulid the confusion matrix and build a heat map #############

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_DT)
print('Confusion matrix\n\n', cm)

######################## GUI predect a specific image & printing its orginal label ########################

# for n in range(0, 12):
#     img = d2_x_dataset_test[n]
#     plt.imshow(x_testDT[n])
#     plt.show()

#     input_img = np.expand_dims(img, axis=0)
#     input_img_features = x_DT_model.predict(input_img)

#     prediction_DT = x_DT_model.predict(input_img)[0]

#     prediction_DT = le.inverse_transform([prediction_DT])
#     print("I think this image is : ", prediction_DT)
#     print("The actual label for this image is: ", test_labels[n])

####################################################################################################

################################ declaring a module of type VGG16 with weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3) 
# and feature extractring and traing module #########
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

feature_extractor = VGG_model.predict(x_trainRF)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features

############################### declaring a RandomForestClassifier with n_estimators=50, random_state=42 then fitting and traing

from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

RF_model.fit(X_for_RF, y_train)

x_test_feature = VGG_model.predict(x_testRF)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)
prediction_RF = RF_model.predict(x_test_features)
prediction_RF = le.inverse_transform(prediction_RF)

############################### calc accuracy of rf #########

from sklearn import metrics

print('Accuracy for Random forest = ', metrics.accuracy_score(test_labels, prediction_RF))
######### to bulid the confusion matrix and build a heat map #########
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)

sns.heatmap(cm, annot=True)

######################## GUI predect a specific image & printing its orginal label ########################

# for n in range(0, 20):
#     img = x_testRF[n]
#     plt.imshow(img)
#     plt.show()

#     input_img = np.expand_dims(img, axis=0)
#     input_img_features = VGG_model.predict(input_img)
#     input_img_features = input_img_features.reshape(input_img_features.shape[0], -1)

#     prediction_RF = RF_model.predict(input_img_features)[0]
#     prediction_RF = le.inverse_transform([prediction_RF])
#     print("The prediction for this image is: ", prediction_RF)
#     print("The actual label for this image is: ", test_labels[n])

###################################################
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


btn1 = Button(app, text="DecisionTree", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=DecisionTreeAccuracy)
btn1.pack()


def RandomForestAccuracy():
    messagebox.showinfo("Accuracy for Random Forests.\n", str(metrics.accuracy_score(test_labels, prediction_RF)))


btn2 = Button(app, text="Random Forest", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0,
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


btn5 = Button(app, text="RF", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0, command=RandomForest)
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


btn6 = Button(app, text="DT", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0, command=DecisionTree)
btn6.pack()


def HeatMapRF():
    cm = confusion_matrix(test_labels, prediction_RF)
    sns.heatmap(cm, annot=True)
    plt.show()


btn7 = Button(app, text="Heat Map For Random Forest", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=HeatMapRF)
btn7.pack()


def HeatMapDT():
    cm = confusion_matrix(test_labels, prediction_DT)
    sns.heatmap(cm, annot=True)
    plt.show()


btn8 = Button(app, text="Heat Map For Decision Tree", width=20, height=3, bg="#a09f9f", fg="white", borderwidth=0,
              command=HeatMapDT)
btn8.pack()
app.mainloop()

###############################################
# from tkinter import *
# app3= Tk()
# app3.title ("welcome to Project")
# app3.geometry("500x500+150+150")

# def app4 ():
#     app4 = Tk()
#     app4.geometry("600x600")
#     image = Label (app4,text= print("The prediction for this image is: ", prediction_RF))
#     print("The actual label for this image is: ", test_labels[n])
#     image.pack()
#     app4.mainloop()

# btn5 = Button (app3,text="RF",width= 20,height=3,bg="#a09f9f",fg="white",borderwidth=0,command=app4)
# btn5.pack()
# app3.mainloop()
