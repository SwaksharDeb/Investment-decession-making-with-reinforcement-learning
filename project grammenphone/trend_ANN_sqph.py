# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Importing the dataset
dataset_train = pd.read_csv('SQPH Historical Data 2009-2017.csv')
X = dataset_train.iloc[50:, [7,11,12,13,14]].values
y = dataset_train.iloc[50:, 15:16].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))

# Adding the second hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

#classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32,  validation_split=0.3, epochs = 300)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i][0]>0.5:
        y_pred[i][0] = 1
    else:
        y_pred[i][0] = 0

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

### Saving the architecture (topology) of the network
classifier_json = classifier.to_json()
with open("sqph classification with factor.json", "w") as json_file:
    json_file.write(classifier_json)

### Saving network weights
classifier.save_weights("sqph classification with factor.json.h5")

#load the pretrain model
with open('sqph classification with factor.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("sqph classification with factor.json.h5")

#loading the dataset
dataset_test = pd.read_csv('SQPH Historical Data2018 practice.csv')
X = dataset_test.iloc[16:, [7,11,12,13,14]].values
y = dataset_test.iloc[16:, 15:16].values

# Feature Scaling
sc = MinMaxScaler()
X = sc.fit_transform(X)

#prediction
y_pred = model.predict(X)

for i in range(len(y_pred)):
    if y_pred[i][0]>0.5:
        y_pred[i][0] = 1
    else:
        y_pred[i][0] = 0