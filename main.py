# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import mean

tf.__version__

# Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('deliveries-2.csv')
print(dataset.columns)
dataset.drop(labels=['delivery_date', 'slot_start_time', 'slot_end_time', 'street', 'street_number', 'zip_code', 'zip_place', 'area_segment', 'municipality'], axis=1, inplace=True)
#, 'street', 'street_number', 'zip_code','zip_place', 'area_segment', 'municipality'
dataset.dropna(inplace=True)
print(dataset.columns) # can_select_unattend

# Features
X = dataset.iloc[:, 1:-1].values


from sklearn import preprocessing
# Service time
y = dataset.iloc[:, -1].values

print("---X---")
print(X)
print("---Y---")
print(y)

# Encoding categorical data
# Label Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
X[:, 4] = le.fit_transform(X[:, 4])


print("-------------")
print(X[0])


# One Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
"""
print(type(X[0][9]))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [9, 11, 12, 13, 14])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
"""


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Dense layer

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='relu'))

# Training the ANN

# Compiling the ANN
ann.compile(optimizer='adam', loss='MAE', metrics=['MAE'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=64, epochs=50)

preds = ann.predict(X_test)

value = tf.keras.losses.MAE(
    y_test, preds
)

print(mean(value))
