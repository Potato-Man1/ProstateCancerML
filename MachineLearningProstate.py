import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
#Use pandas to load the csv and numpy for keras
import pandas as pd
import numpy as np

#Load the Data
Data = pd.read_csv("Prostate_Cancer.csv")

#The features
X = Data[['radius',	'texture',	'perimeter', 'area',	'smoothness',	'compactness',	'symmetry',	'fractal_dimension']]

#Our target to predict
y = Data['diagnosis_result']

X = np.array(X)
y = np.array(y)

y_list = []

#The diagnose result is B and M we must change it to 1 and 0 
#Benign = 1 & Malignant = 0
for i in y:
  if i == 'M':
    y_list.append(0)
  else:
    y_list.append(1)

#Turn it into numpy array again
y = np.array(y_list)

#Split them for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=1)

#The model 
model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200)

#Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
