import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from keras.models import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization, MaxPooling2D, Flatten, Conv2D, Dense
)
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from keras.applications.vgg16 import VGG16

SIZE = 256

data = pd.read_csv("C:/Users/roshd/IdeaProjects/object/csvTestImages 10k x 784.csv")
trainData = data.iloc[:9999, 1:]
trainLabel = data.iloc[:9999, 0]
x_train, x_test, y_train, y_test = train_test_split(trainData, trainLabel, test_size=0.1, random_state=20)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(y_test)
test_labels_encoded = le.transform(y_test)
le.fit(y_train)
train_labels_encoded = le.transform(y_train)

feature_extractor = Sequential()

X_for_RF = feature_extractor.predict(x_train)

from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

RF_model.fit(X_for_RF, y_train)

x_test_feature = feature_extractor.predict(x_test)

prediction_RF = RF_model.predict(x_test_feature)

prediction_RF = le.inverse_transform(prediction_RF)

from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(y_test, prediction_RF))
print(y_test)
print(prediction_RF)
from sklearn.metrics import confusion_matrix

# TODO Photos are the same and need to be changed


cm = confusion_matrix(y_test, prediction_RF)
print(cm)


