import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Split data into training and test partitions
def createSet(dataset):
    x_cols = [col for col in dataset.columns if (col != 'label' and col != 'record')]
    X_data = dataset[x_cols].values
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], -1))
    Y_data = dataset['label'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, random_state=0, test_size = 0.3, train_size = 0.7)

    num_classes = len(np.unique(Y_data))

    return num_classes,X_train, X_test, Y_train, Y_test 


# Convert class vectors to binary class matrices
def binaryConvertion(num_classes, Y_train, Y_test):

    Y_train_encoder = sklearn.preprocessing.LabelEncoder()
    Y_train_num = Y_train_encoder.fit_transform(Y_train)
    Y_train_wide = np_utils.to_categorical(Y_train_num, num_classes)

    Y_test_num = Y_train_encoder.fit_transform(Y_test)
    Y_test_wide = np_utils.to_categorical(Y_test_num, num_classes)

    return Y_train_wide, Y_test_num, Y_test_wide

