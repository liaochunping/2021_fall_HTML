import numpy as np
import pandas as pd
from pyfm import pylibfm
from imblearn.over_sampling import SMOTE
train_dataX = pd.read_csv('x_in.csv')
train_dataX = train_dataX.drop(labels='Customer ID',axis=1)
train_dataY = pd.read_csv('y_in.csv')
train_dataY = train_dataY.drop(labels='Customer ID',axis=1)
test_dataX = pd.read_csv('x_out.csv')
test_dataX = test_dataX.drop(labels='Customer ID',axis=1)