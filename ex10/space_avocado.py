import sys
sys.path.append('../')
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex09.data_spliter import data_spliter
from ex08.polynomial_train import add_polynomial_features
from ex07.polynomial_model import *
from ex10.benchmark_train import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import *
import csv
import pickle


def vizualize_pred(pred, X, Y, labelX, labelY):
    plt.scatter(X, pred, label="prediction")
    plt.scatter(X, Y, label="Reals values")
    plt.legend()
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.show()

def compare_pred(pred, X, Y):
    vizualize_pred(pred, X[:,0], Y, "weight", "target")
    vizualize_pred(pred, X[:,1], Y, "distance", "target")
    vizualize_pred(pred, X[:,2], Y, "time", "target")

def compare_mses(mses):
    for mse in mses:
        plt.scatter(mse, mses[mse], label=mse)
    plt.legend()
    plt.xlabel("models")
    plt.ylabel("mse")
    plt.show()


def get_values(data):
    X = np.array(data[["weight", "prod_distance", "time_delivery"]])
    Y = np.array(data[["target"]])
    return X, Y

def compare_models(models, X, Y):
    mses = {}
    preds = {}
    best_mse = -1
    best_model = ""
    for model in models:
        myLR =  models[model]
        X_weight = add_polynomial_features(X[:, 0].reshape(-1, 1), int(model[1]))
        X_distance = add_polynomial_features(X[:, 1].reshape(-1, 1), int(model[3]))
        X_time = add_polynomial_features(X[:, 2].reshape(-1, 1), int(model[5]))
        enginered_X = np.hstack((X_weight, X_distance, X_time))
        Y_pred = myLR.predict_(enginered_X)
        preds[model] =  Y_pred
        current_mse = MyLR.mse_(Y, Y_pred)
        mses[model] = current_mse
        if best_mse < 0 or current_mse < best_mse:
            best_mse = current_mse
            best_model = model
    print(best_mse)
    compare_pred(preds[best_model], X, Y)
    compare_mses(mses)

def train_model(myLR, model, X, Y):
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.5)
    X_train_weight = engine_features(X_train[:, 0].reshape(-1, 1))
    X_train_distance = engine_features(X_train[:, 1].reshape(-1, 1))
    X_train_time = engine_features(X_train[:, 2].reshape(-1, 1))
    X_train_features = np.hstack((X_train_weight[:,:int(model[1])], X_train_distance[:,:int(model[3])], X_train_time[:, :int(model[5])]))
    Y_train = normalize(Y_train)
    myLR.fit_(X_train_features, Y_train)
    preds = myLR.predict_(X_test)
    compare_pred(preds, X_test, Y_test)

def main():
    file = open('models.pickle', 'rb')
    models = pickle.load(file)
    file.close()
    values_data = pd.read_csv("space_avocado.csv")
    X, Y = get_values(values_data)
    compare_models(models, X, Y)

if __name__ == "__main__":
    main()