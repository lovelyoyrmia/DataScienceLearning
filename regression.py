import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn import linear_model, model_selection


boston = load_boston()
# print((boston))
# path = open('Data.txt','a')
# path.write(str(boston))

dataBoston = pd.DataFrame(
    data=np.c_[boston["data"], boston["target"]],
    columns=list(boston["feature_names"]) + ["MEDV"],
)
dataBoston = dataBoston[["CHAS", "RM", "AGE", "TAX", "LSTAT", "MEDV"]]

print("-" * 27)
print("           HEAD")
print("-" * 27)

print(dataBoston.head())
# dataBoston = pd.read_csv(boston['filename'])


"""EDA"""
print("-" * 27)
print("           EDA")
print("-" * 27)
print(dataBoston.describe())
# print(plt.figure(figsize=(15,10)))
# print(sea.heatmap(dataBoston.corr(),annot=True))

"""Simple Linear Regression"""
# print('-'*49);print('           Simple Linear Regression');print('-'*49)
# x = dataBoston['RM'].values.reshape(-1,1)
# # print(x)
# y = dataBoston['MEDV'].values

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(x,y)
# print(model.coef_)
# print(model.intercept_)
# print('Prediction the price of houses that have 12 bedroom: '\
#      f'{float(model.predict(np.array([[12]])))} $')

"""DATA SPLITTING"""
print("-" * 49)
print("               SPLIT DATA")
print("-" * 49)
x = np.array(dataBoston.drop(["MEDV"], 1))
y = np.array(dataBoston["MEDV"])

print("X", x.shape)
print("Y", y.shape)

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(
    x, y, test_size=0.1, random_state=5
)

print("X TRAIN", xTrain.shape)
print("X TEST", xTest.shape)

"""TRAINING DATA"""
print("-" * 49)
print("               TRAIN DATA")
print("-" * 49)
model = linear_model.LinearRegression()
model.fit(x, y)
accuracy = model.score(xTest, yTest)
print(f"COEFICIENTS {model.coef_}")
print(f"INTERCEPT {model.intercept_}")
print(f"ACCURACY {round(accuracy*100,3)}%")

"""TESTING DATA"""
print("-" * 49)
print("               TEST DATA")
print("-" * 49)
testVal = model.predict(xTest)
# print(testVal.shape)
# error = []
# for i,testVals in enumerate(testVal):
#     error.append(yTest[i]-testVals)
#     print(f'Actual Value: {yTest[i]}    Prediction Value: {int(testVals)}   Error: {int(error[i])}')
