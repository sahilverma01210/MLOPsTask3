# import packages
import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.backend import clear_session
import numpy

# load mnist dataset 
(train_X , train_y), (test_X , test_y) = mnist.load_data("mymnist.data")
# reshape data
test_X = test_X.reshape(-1 , 28*28)
train_X = train_X.reshape(-1 ,  28*28)
# change data type to float
test_X = test_X.astype("float32")
train_X = train_X.astype("float32")
# convert to categorical data using one hot encoding
test_y = to_categorical(test_y)
train_y = to_categorical(train_y)

model = Sequential()
model.add(Dense(units = 20 , input_dim = 28*28 , activation = 'relu'))
model.add(Dense(units=200 , input_dim = 28*28 , activation = 'relu'))
model.add(Dense(units=60 , input_dim = 28*28 , activation = 'relu'))
model.add(Dense(units=60 , input_dim = 28*28 , activation = 'relu'))
model.add(Dense(units=60 , input_dim = 28*28 , activation = 'relu'))
model.add(Dense(units=60 , input_dim = 28*28 , activation = 'relu'))
model.add(Dense(units=10 , input_dim = 28*28 , activation = 'softmax'))
model.compile( optimizer= "Adam" , loss='categorical_crossentropy', metrics=['accuracy'] )
fit_model = model.fit(train_X ,  train_y , epochs = 2 , verbose =  False)

# record model accuracy in a text file
text = fit_model.history
accuracy = text['accuracy'][1] * 100
accuracy = int(accuracy)
f= open("accuracy.txt","w+")
f.write(str(accuracy))
f.close()
print("Accuracy for the model is : " , accuracy ,"%")
