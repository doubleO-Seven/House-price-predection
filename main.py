import tensorflow as tf
from tensorflow import keras #ML Library
import numpy as np #Mathematics Library
import pandas as pd #Data-handling Library
import matplotlib.pyplot as plt #Used to graph our results

from google.colab import drive
drive.mount('/content/drive')
#df = data frame
df = pd.read_csv('drive/Shareddrives/doubleOseven/Keras/1. Linear Regression/IowaHousingPrices.csv')

squareFeet = df[['SquareFeet']].values
salePrice = df[['SalePrice']].values

model = keras.Sequential() #every keras contains this line. it is the starting line
model.add(keras.layers.Dense(1, input_shape = (1, ))) #input =1 , besically we pass 1 input
model.compile(keras.optimizers.Adam(lr=1), 'mean_squared_error') 
#optimizer error kom rakhe. Adam optimzer is best one of them
#lr = Learning rate and ja error pabo setar square valur jnno mean_sq use korsi

model.fit(squareFeet, salePrice, epochs = 30, batch_size = 30) #epochs = how many time you will run through to data
#train korar jnno use kora hoy model.fit()
# batch_size - how many data going through the network at a time

#Plot datapoints
df.plot(kind='scatter', x='SquareFeet', y='SalePrice', title='Predicted housing price based on square feet')

y_pred = model.predict(squareFeet) #predicted house price based on square feet

#Plot the linear regression line
plt.plot(squareFeet, y_pred, color='purple')

newSF = 1689
print(model.predict([newSF]))