import sys
sys.path.append('../')
from ex05.mylinearregression import MyLinearRegression as MyLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def output_scatter(x, y, y_hat, labels):
    plt.scatter(x, y, label=labels[0])
    plt.scatter(x, y_hat, label=labels[1])
    plt.xlabel(labels[2])
    plt.ylabel(labels[3])
    plt.grid()
    plt.legend()
    plt.show()

def perform_uni_linear_regression(X, Y, theta, labels):
    myLR =  MyLR(theta, alpha = 2.5e-5, max_iter = 1500000)
    y_pred = myLR.predict_(X)
    output_scatter(X, Y, y_pred, labels)
    myLR.fit_(X, Y)
    y_pred = myLR.predict_(X)
    output_scatter(X, Y, y_pred, labels)
    print(myLR.mse_(y_pred,Y))



def univariate_linear_regression(data):
    XAge = np.array(data[["Age"]])[:,0].reshape(-1,1)
    Y = np.array(data[["Sell_price"]])[:,0].reshape(-1,1)
    perform_uni_linear_regression(XAge, Y, np.array([[1000], [-1]]), ["Sell price", "Predicted sell price", "x1: age (in years)", "y: sell price (in kiloeuros)"])
    XThrust_power = np.array(data[["Thrust_power"]])[:,0].reshape(-1,1)
    perform_uni_linear_regression(XThrust_power, Y, np.array([[1000], [-1]]), ["Sell price", "Predicted sell price", "x2: thrust power (in 10Km/s", "y: sell price (in kiloeuros)"])
    XTerameters = np.array(data[["Terameters"]])[:,0].reshape(-1,1)
    perform_uni_linear_regression(XTerameters, Y, np.array([[1000], [-1]]), ["Sell price", "Predicted sell price", "x3: distance totalizer value of spacecraft (in Tmeters) ", "y: sell price (in kiloeuros)"])


def mulltivariate_linear_regression(data):
    X = np.array(data[["Age","Thrust_power","Terameters"]])
    Y = np.array(data[["Sell_price"]])
    my_lreg = MyLR(np.array([[1.0], [1.0], [1.0], [1.0]]) , alpha = 0.00005, max_iter = 600000)
    y_pred = my_lreg.predict_(X)
    print(MyLR.mse_(Y,y_pred))
    my_lreg.fit_(X,Y)
    print(my_lreg.thetas)
    y_pred = my_lreg.predict_(X)
    print(MyLR.mse_(Y,y_pred))
    output_scatter(X[:,0].reshape(-1, 1), Y, y_pred, ["Sell price", "Predicted sell price", "x1: age (in years)", "y: sell price (in kiloeuros)"])
    output_scatter(X[:,1].reshape(-1, 1), Y, y_pred, ["Sell price", "Predicted sell price", "x2: thrust power (in 10Km/s", "y: sell price (in kiloeuros)"])
    output_scatter(X[:,2].reshape(-1, 1), Y, y_pred, ["Sell price", "Predicted sell price", "x3: distance totalizer value of spacecraft (in Tmeters) ", "y: sell price (in kiloeuros)"])

def main():
    data = pd.read_csv("spacecraft_data.csv")
    univariate_linear_regression(data)
    mulltivariate_linear_regression(data)
    return 

if __name__ == "__main__":
    main()
