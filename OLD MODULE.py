import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from keras.models import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization, MaxPooling2D, Flatten, Conv2D, Dense
)
from keras.applications.vgg16 import VGG16

print(os.listdir("C:/Users/roshd/IdeaProjects/object/"))

SIZE = 256

train_images = []
train_labels = []

for directory_path_Train in glob.glob("C:/Users/roshd/IdeaProjects/object/Train/*"):
    label = directory_path_Train.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path_Train, "*.jpg")):
        print(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

#####################

test_images = []
test_labels = []

for directory_path in glob.glob("C:/Users/roshd/IdeaProjects/object/Test/*"):
    print(label)
    fruitLabel = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruitLabel)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

x_train, x_test = x_train / 255.0, x_test / 255.0

###############################

activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation=activation, padding='same', input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

X_for_RF = feature_extractor.predict(x_train)

RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

RF_model.fit(X_for_RF, y_train)

x_test_feature = feature_extractor.predict(x_test)

prediction_RF = RF_model.predict(x_test_feature)

prediction_RF = le.inverse_transform(prediction_RF)

from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
print(test_labels)
print(prediction_RF)
from sklearn.metrics import confusion_matrix

# TODO Photos are the same and need to be changed


cm = confusion_matrix(test_labels, prediction_RF)
print(cm)

sns.heatmap(cm, annot=True)

n = 5
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0)
input_img_features = feature_extractor.predict(input_img)

prediction_RF = RF_model.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF])
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
