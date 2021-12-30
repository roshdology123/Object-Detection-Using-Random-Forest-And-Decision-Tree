# 4 = to how many classes to distinguish from
prediction_layer = Dense(3, activation='softmax')(x)

cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(cnn_model.summary())

history = cnn_model.fit(x_train, y_train_one_hot, epochs=5, validation_data=(x_test, y_test_one_hot))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

prediction_NN = cnn_model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

print(prediction_NN)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
sns.heatmap(cm,annot=True)

n=5
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0)
prediction = np.argmax(cnn_model.predict(input_img))
prediction = le.inverse_transform([prediction])
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])