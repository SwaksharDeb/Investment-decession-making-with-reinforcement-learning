# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Importing the training set
dataset_train = pd.read_csv('GRAE Historical Data 2009-2017.csv')
training_set = dataset_train.iloc[16:, [7,11,12,13,14]].values
training_set_y = dataset_train.iloc[16:, 1:2].values


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled_y = sc.fit_transform(training_set_y)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, :])
    y_train.append(training_set_scaled_y[i, 0])
    
X_train[0] = np.reshape(X_train[0], (1,-1))
array = np.reshape(X_train[0],(1,60,-1))
 
for i in range(1,len(X_train)):
    X_train[i] = np.reshape(X_train[i],(1,-1))
    X_train[i] = np.reshape(X_train[i],(1,60,-1))
    array = np.vstack((array,X_train[i]))
    
X_train = array
y_train = np.array(y_train)
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout

# Initialising the RNN
regressor = tf.keras.models.Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = False, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(tf.keras.layers.Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
#regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
#regressor.add(tf.keras.layers.Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#regressor.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
#regressor.add(tf.keras.layers.Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#regressor.add(tf.keras.layers.LSTM(units = 50))
#regressor.add(tf.keras.layers.Dropout(0.2))

# Adding the output layer
regressor.add(tf.keras.layers.Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)  ##early stopping
history = regressor.fit(X_train, y_train, epochs=1000, batch_size = 32,
                    validation_split = 0.3, verbose=1)#, callbacks=[early_stop])

#creating loss vs no of iteration curve
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
#dataset_test_ = pd.read_csv('GRAE Historical Data 2009-2017.csv')
#dataset_test__ = pd.read_csv('GRAE Historical Data 2018 practice.csv')
dataset_test_1 = pd.read_csv('GRAE Historical Data 2009-2017.csv').iloc[:,[1,7,11,12,13,14]]
dataset_test_1 = dataset_test_1.iloc[-60:,:] 
dataset_test_2 = pd.read_csv('GRAE Historical Data 2018 practice.csv').iloc[:,[1,7,11,12,13,14]]
#dataset_test_2 = dataset_test_2.iloc[0:200,:]
dataset_test = pd.concat([dataset_test_1, dataset_test_2], axis = 0, ignore_index=True, sort=False)
test_set = dataset_test.iloc[:,1:].values
test_set_y = dataset_test.iloc[:, 0:1].values

inputs = test_set[:,:]
sc = MinMaxScaler(feature_range = (0, 1))
sc_1 = MinMaxScaler(feature_range = (0, 1))
inputs = sc.fit_transform(inputs)
test_set_scaled_y = sc_1.fit_transform(test_set_y)

X_test = []
for i in range(60, len(test_set)):
    X_test.append(inputs[i-60:i, :])

X_test[0] = np.reshape(X_test[0], (1,-1))
array = np.reshape(X_test[0],(1,60,-1))
 
for i in range(1,len(X_test)):
    X_test[i] = np.reshape(X_test[i],(1,-1))
    X_test[i] = np.reshape(X_test[i],(1,60,-1))
    array = np.vstack((array,X_test[i]))

X_test = array

y_test = []
for i in range(60,len(test_set_scaled_y)):
    y_test.append(test_set_scaled_y[i,0])

y_test = np.array(y_test)
y_test = np.reshape(y_test, (-1,1))

predicted_stock_price = sc_1.inverse_transform(regressor.predict(X_test))
real_stock_price = sc_1.inverse_transform(y_test)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real GP')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted GP')
plt.title('GP Price Prediction')
plt.xlabel('Time')
plt.ylabel('GP Price')
plt.legend()
plt.show()

### Saving the architecture (topology) of the network
regressor_json = regressor.to_json()
with open("gp prediction with factor.json", "w") as json_file:
    json_file.write(regressor_json)

### Saving network weights
regressor.save_weights("gp prediction with factor.json.h5")

#loading the pretrain model
with open('gp prediction with factor.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("gp prediction with factor.json.h5")

dataset_test_1 = pd.read_csv('GRAE Historical Data 2009-2017.csv').iloc[:,[1,7,11,12,13,14]]
dataset_test_1 = dataset_test_1.iloc[-60:,:] 
dataset_test_2 = pd.read_csv('GRAE Historical Data 2018 practice.csv').iloc[:,[1,7,11,12,13,14]]
dataset_test = pd.concat([dataset_test_1, dataset_test_2], axis = 0, ignore_index=True, sort=False)
test_set = dataset_test.iloc[:,1:].values
test_set_y = dataset_test.iloc[:, 0:1].values

inputs = test_set[:,:]
sc = MinMaxScaler(feature_range = (0, 1))
sc_1 = MinMaxScaler(feature_range = (0, 1))
inputs = sc.fit_transform(inputs)
test_set_scaled_y = sc_1.fit_transform(test_set_y)

X_test = []
for i in range(60, len(test_set)):
    X_test.append(inputs[i-60:i, :])

X_test[0] = np.reshape(X_test[0], (1,-1))
array = np.reshape(X_test[0],(1,60,-1))
 
for i in range(1,len(X_test)):
    X_test[i] = np.reshape(X_test[i],(1,-1))
    X_test[i] = np.reshape(X_test[i],(1,60,-1))
    array = np.vstack((array,X_test[i]))

X_test = array

y_test = []
for i in range(60,len(test_set_scaled_y)):
    y_test.append(test_set_scaled_y[i,0])

y_test = np.array(y_test)
y_test = np.reshape(y_test, (-1,1))

predicted_stock_price = sc_1.inverse_transform(model.predict(X_test))
real_stock_price = sc_1.inverse_transform(y_test)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real GP')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted GP')
plt.title('GP Price Prediction')
plt.xlabel('Time')
plt.ylabel('GP Price')
plt.legend()
plt.show()

"""testing on 2019 dataset"""

with open('gp prediction with factor.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("gp prediction with factor.json.h5")

dataset_test_1 = pd.read_csv('GRAE Historical Data 2018 practice.csv').iloc[:,[1,7,11,12,13,14]]
dataset_test_1 = dataset_test_1.iloc[-60:,:] 
dataset_test_2 = pd.read_csv('GRAE Historical Data 2019 .csv').iloc[:,[1,7,11,12,13,14]]
dataset_test = pd.concat([dataset_test_1, dataset_test_2], axis = 0, ignore_index=True, sort=False)
test_set = dataset_test.iloc[:,1:].values
test_set_y = dataset_test.iloc[:, 0:1].values

inputs = test_set[:,:]
sc = MinMaxScaler(feature_range = (0, 1))
sc_1 = MinMaxScaler(feature_range = (0, 1))
inputs = sc.fit_transform(inputs)
test_set_scaled_y = sc_1.fit_transform(test_set_y)

X_test = []
for i in range(60, len(test_set)):
    X_test.append(inputs[i-60:i, :])

X_test[0] = np.reshape(X_test[0], (1,-1))
array = np.reshape(X_test[0],(1,60,-1))
 
for i in range(1,len(X_test)):
    X_test[i] = np.reshape(X_test[i],(1,-1))
    X_test[i] = np.reshape(X_test[i],(1,60,-1))
    array = np.vstack((array,X_test[i]))

X_test = array

y_test = []
for i in range(60,len(test_set_scaled_y)):
    y_test.append(test_set_scaled_y[i,0])

y_test = np.array(y_test)
y_test = np.reshape(y_test, (-1,1))

predicted_stock_price = sc_1.inverse_transform(model.predict(X_test))
real_stock_price = sc_1.inverse_transform(y_test)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real GP')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted GP')
plt.title('GP Price Prediction')
plt.xlabel('Time')
plt.ylabel('GP Price')
plt.legend()
plt.show()


