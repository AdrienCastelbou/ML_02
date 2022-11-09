import sys
sys.path.append('../')
from ex05.mylinearregression import MyLinearRegression as MyLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_polynomial_features(x, power):
    try:
        if type(x) != np.ndarray or x.shape[1] != 1 or not isinstance(power, (int, float)):
            return None
        return np.vander(x.reshape(-1), power + 1 , increasing=True)[:,1:]
    except:
        return None

def perform_regression(X, Y, theta):
    myLR =  MyLR(theta, alpha = 1.5e-10, max_iter = 1500000) 
    y_pred = myLR.predict_(X)
    myLR.fit_(X, Y)
    y_pred = myLR.predict_(X)
    print(myLR.mse_(y_pred,Y))
    return y_pred

def output_graph(x, y, y_preds):
    plt.scatter(x, y, label="Pills Score")
    for i, y_pred in zip(range(len(y_preds)), y_preds):
        plt.plot(x, y_pred, label=f"degree{i + 1} polynomial expression")
    plt.xlabel("Micrograms")
    plt.ylabel("Scores")
    plt.grid()
    plt.legend()
    plt.show()

def polynomial_regression(data):
    XMicrograms = np.array(data[["Micrograms"]])[:,0].reshape(-1,1)
    Y = np.array(data[["Score"]])[:,0].reshape(-1,1)
    y_preds = []
    for i in range(1, 7):
        poly_X = add_polynomial_features(XMicrograms, i)
        theta = np.ones((poly_X.shape[1] + 1, 1)).reshape(-1, 1)
        if i == 1:
            theta = np.array([[88.67822162],[-8.92106849]]).reshape(-1,1)
        elif i == 2:
            theta = np.array([[65.92627185],[ 3.63987692],[-1.47092799]]).reshape(-1, 1)
        elif i == 3:
          theta = np.array([[ 52.01405471], [ 33.19711719], [-12.9595629 ], [  1.15407571]]).reshape(-1, 1)
        if i == 4:
            theta = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
        elif i == 5:
            theta = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
        elif i == 6:
            theta = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)
        y_preds.append(perform_regression(poly_X, Y, theta))
    output_graph(XMicrograms, Y, y_preds)

def main():
    data = pd.read_csv("are_blue_pills_magics.csv")
    polynomial_regression(data)
    return 

if __name__ == "__main__":
    main()