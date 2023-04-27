ANN_model = keras.Sequential()   
# First layer
ANN_model.add(Dense(50, input_dim = 7))     # 50 neurons, input_dim = 7 as 7 parameters in the dataset
ANN_model.add(Activation('relu'))           # using ReLU activation function

# adding another layer
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5)) # dropout=0.5 =>gonna drop 50% neurons so that network is not overfitting

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(50))
ANN_model.add(Activation('linear'))
# In the Output layer Activation is gonna be Linear as we're implementing a Regression task
# try to avoid any Activation func in output layer that is saturated, eg: Sigmoid
ANN_model.add(Dense(1))

ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()
result = ANN_model.evaluate(xtest, ytest)
accuracy_ANN = 1 - result
print("Accuracy : {}".format(accuracy_ANN))
# Looking at the progression of the network throughout the number of epochs
epochs_hist.history.keys()
# loss => progression of network throughout the no. of epochs
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress during Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])  # for label/box in the plot
newPerson = [[330, 110, 4, 4.5, 4.5, 9.5, 1]]
pred = ANN_model.predict(newPerson)
pred[0]
