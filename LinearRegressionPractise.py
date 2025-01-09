import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

import pandas as pd

# load the data from the dataset
data = pd.read_csv('Honda_Data.csv')
print(data.head())



x_train = data[["Date", "High", "Open", "Volume"]]
y_train = data[["Adj_Close", "Close"]]






print("Typr eof x:", type(x_train))
print("First five element of x are: \n", x_train[:5])
print("the shape of X is:", x_train.shape)







