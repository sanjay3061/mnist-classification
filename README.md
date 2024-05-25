# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![image](https://github.com/sanjay3061/mnist-classification/assets/121215929/e4d81804-92d4-4618-9f76-38cda67147e2)


## DESIGN STEPS

### STEP 1:
Load the dataset from the tensorflow library.

### STEP 2:
Preprocess the dataset. MNIST dataset using to classify handwritten written digit.
### STEP 3:
Create and train your model.
### STEP 4:
plot the training loss, validation loss vs iteration plot.
### STEP 5:
Test the model for your handwritten scanned images.







## PROGRAM
```python
### Name:Sanjay.R
### Register Number:212222220038
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape

X_test.shape

single_image.shape
plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
inputs1=keras.Input(shape=(28,28,1))
model.add(inputs1)
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(15,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=6,batch_size=64,validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)

metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Name:Sanjay.R")
print("Reg no:212222220038")
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('1.png')

img = image.load_img('1.png')
img = image.load_img('1.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)



print(x_single_prediction)


plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

print("Name:Sanjay.R")
print("Reg no:212222220038")

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/sanjay3061/mnist-classification/assets/121215929/579de983-b647-4b4e-941f-595afcb09868)

![image](https://github.com/sanjay3061/mnist-classification/assets/121215929/00d4fae5-ad4e-40a3-9f6d-90c79337ca2a)


![image](https://github.com/sanjay3061/mnist-classification/assets/121215929/284a556d-0995-459b-9a50-e20f3683a5e7)


### Classification Report

![image](https://github.com/sanjay3061/mnist-classification/assets/121215929/d3f08ca9-4c72-4f14-a4b5-98e81d9dcafe)

### Confusion Matrix

![image](https://github.com/sanjay3061/mnist-classification/assets/121215929/94e45f59-13ae-42d9-9280-76715554bc40)

### New Sample Data Prediction
![image](https://github.com/sanjay3061/mnist-classification/assets/121215929/4f36c2e3-adc4-4c0c-8250-ff92bc7ad9e4)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
