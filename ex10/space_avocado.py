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
import pickle

def transform_data(X, model):
    X_weight = engine_features(X[:, 0].reshape(-1, 1))
    X_distance = engine_features(X[:, 1].reshape(-1, 1))
    X_time = engine_features(X[:, 2].reshape(-1, 1))
    X_features = np.hstack((X_weight[:,:int(model[1])], X_distance[:,:int(model[3])], X_time[:, :int(model[5])]))
    return X_features


def compare_pred(pred, X, Y):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Predictions comparisions")
    ax1.scatter(X[:,0], Y, label="Real values")
    ax1.scatter(X[:,0], pred, label="Predictions")
    ax1.legend()
    ax1.set_xlabel("weight")
    ax1.set_ylabel("price")
    ax2.scatter(X[:,1], Y, label="Real values")
    ax2.scatter(X[:,1], pred, label="Predictions")
    ax2.legend()
    ax2.set_xlabel("prod_distance")
    ax2.set_ylabel("price")
    ax3.scatter(X[:,2], Y, label="Real values")
    ax3.scatter(X[:,2], pred, label="Predictions")
    ax3.legend()
    ax3.set_xlabel("time_delivery")
    ax3.set_ylabel("price")
    fig.tight_layout()
    plt.show()

def compare_mses(mses):
    plt.rcParams["figure.figsize"] = (20,7)
    for mse in mses:
        plt.scatter(mse, mses[mse], label=mse)
    plt.grid()
    plt.xticks(rotation="vertical")
    plt.xlabel("models")
    plt.ylabel("mse")
    plt.show()


def get_values(data):
    X = np.array(data[["weight", "prod_distance", "time_delivery"]])
    Y = np.array(data[["target"]])
    return X, Y

def evaluate_models(models, X, Y):
    mses = {}
    best_mse = -1
    best_model = ""
    X_weight = engine_features(X[:, 0].reshape(-1, 1))
    X_distance = engine_features(X[:, 1].reshape(-1, 1))
    X_time = engine_features(X[:, 2].reshape(-1, 1))
    n_Y = normalize(Y)
    for model in models:
        myLR =  models[model]
        X_features = np.hstack((X_weight[:,:int(model[1])], X_distance[:,:int(model[3])], X_time[:, :int(model[5])]))
        Y_pred = myLR.predict_(X_features)
        current_mse = MyLR.mse_(n_Y, Y_pred)
        mses[model] = current_mse
        if best_mse < 0 or current_mse < best_mse:
            best_mse = current_mse
            best_model = model
    print(best_model, best_mse)
    compare_mses(mses)
    return best_model, models[best_model]

def train_model(myLR, model, X, Y):
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.8)
    X_train_features = transform_data(X_train, model)
    Y_train = normalize(Y_train)
    myLR.fit_(X_train_features, Y_train)
    X_test_features =  transform_data(X_test, model)
    preds = myLR.predict_(X_test_features)
    preds = unormalize(preds, Y_test)
    compare_pred(preds, X_test, Y_test)

def main():
    file = open('models.pickle', 'rb')
    models = pickle.load(file)
    file.close()
    values_data = pd.read_csv("space_avocado.csv")
    X, Y = get_values(values_data)
    best_model, model = evaluate_models(models, X, Y)
    train_model(model, best_model, X, Y)

if __name__ == "__main__":
    main()