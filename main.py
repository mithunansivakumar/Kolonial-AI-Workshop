# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

# Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('deliveries-2.csv')
print(dataset.columns)
dataset.drop(labels=['delivery_date', 'slot_start_time', 'slot_end_time'], axis=1)
print(dataset.columns)
# Features
X = dataset.iloc[:, 1:-1].values


# Service time
y = dataset.iloc[:, -1].values
print("---X---")
print(X)
print("---Y---")
print(y)

# Encoding categorical data
# Label Encoding the binary columns -> 0 or 1, randomly chosen by the machine
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
X[:, 4] = le.fit_transform(X[:, 4])


# One Hot Encoding the "non-binary" columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

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
# The dense layer is a neural network layer that is connected deeply,
# which means each neuron in the dense layer receives input from all neurons of its previous layer.
# The dense layer is found to be the most commonly used layer in the models.

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the ANN

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=20)
