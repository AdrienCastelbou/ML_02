import sys
sys.path.append('../')
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex09.data_spliter import data_spliter
from ex08.polynomial_train import add_polynomial_features
from  ex07.polynomial_model import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import *
import csv

def vizualize_pred(preds, X, Y, labelX, labelY):
    print(X)
    print(X.shape, Y.shape)
    plt.plot(X, Y, label="Reals values")
    for pred in preds:
        plt.scatter(X, preds[pred], label=pred)
    plt.legend()
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.show()

def compare_preds(preds, X, Y):
    print(X.shape, Y.shape)
    vizualize_pred(preds, X[:,0], Y, "weight", "target")
    vizualize_pred(preds, X[:,1], Y, "distance", "target")
    vizualize_pred(preds, X[:,2], Y, "time", "target")

def compare_mses(mses):
    for mse in mses:
        print(mses[mse])
        plt.scatter(mse, mses[mse], label=mse)
    plt.legend()
    plt.xlabel("models")
    plt.ylabel("mse")
    plt.show()

def compare_models(models, X, Y):
    mses = {}
    preds = {}
    for model in models:
        myLR =  MyLR(models[model])
        X_weight = add_polynomial_features(X[:, 0].reshape(-1, 1), int(model[1]))
        X_distance = add_polynomial_features(X[:, 1].reshape(-1, 1), int(model[3]))
        X_time = add_polynomial_features(X[:, 2].reshape(-1, 1), int(model[5]))
        enginered_X = np.hstack((X_weight, X_distance, X_time))
        Y_pred = myLR.predict_(enginered_X)
        preds[model] =  Y_pred
        mses[model] = MyLR.mse_(Y, Y_pred)
    #compare_preds(preds, X, Y)
    compare_mses(mses)

def get_models(data):
    models = {}
    for index, row in data.iterrows():
        thetas = np.empty((0, 0))
        for i in range(1, 8):
            if np.isnan(row[i]):
                break
            thetas = np.append(thetas, row[i])
        models[row[0]] =  thetas.reshape(-1, 1)
    return models

def get_values(data):
    X = np.array(data[["weight", "prod_distance", "time_delivery"]])
    Y = np.array(data[["target"]])
    return X, Y

def main():
    models_data = pd.read_csv("models.csv")
    models = get_models(models_data)
    values_data = pd.read_csv("space_avocado.csv")
    X, Y = get_values(values_data)
    compare_models(models, X, Y)


if __name__ == "__main__":
    main()