from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def test_multi_target_regression(data_x, data_y):
    print(len(data_x))
    n_train = int (len(data_x)*0.80)
    X_train, Y_train = data_x[:n_train], data_y[:n_train]
    X_test, Y_test = data_x[n_train:], data_y[n_train:]
    references = np.zeros_like(Y_test)
    for n in range(2):
        rgr = GradientBoostingRegressor(random_state=0)
        rgr.fit(X_train, Y_train[:, n])
        references[:,n] = rgr.predict(X_test)
    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    rgr.fit(X_train, Y_train)
    "" """"
    # Create matrix and vectors
    X = [[0.44, 0.68], [0.99, 0.23]]
    y = [109.85, 155.72]
    X_test = [0.49, 0.18]
    "" """[1]
    # PolynomialFeatures (prepreprocessing)
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(X_train)
    X_test_ = poly.fit_transform(X_test)
    # Instantiate
    lg = LinearRegression()
    # Fit
    lg.fit(X_, Y_train)
    # Obtain coefficients
    lg.coef_
    # Predict
    Y_train_pred = lg.predict(X_train)
    Y_test_pred = lg.predict(X_test_)
    "" """"
        Y_train_pred = rgr.predict(X_train)
        Y_test_pred = rgr.predict(X_test)
    "" """[2]

    sequence_arr = np.arange(1,len(data_x)+1).reshape(len(data_x),1)
    print(len(Y_test_pred))
    print(len(sequence_arr[n_train:]))

    plot_result(sequence_arr[:n_train], Y_train[:,0], sequence_arr[n_train:],Y_test[:,0], Y_train_pred[:,0], Y_test_pred[:,0])
    plot_result(sequence_arr[:n_train], Y_train[:,1], sequence_arr[n_train:],Y_test[:,1], Y_train_pred[:,1], Y_test_pred[:,1])

    rmse_infer = sqrt(mean_squared_error(Y_test_pred[:,0], Y_test[:,0]))
    rmse_power = sqrt(mean_squared_error(Y_test_pred[:,1], Y_test[:,1]))
    rmse_train_infer = sqrt(mean_squared_error(Y_train_pred[:,0], Y_train[:,0]))
    rmse_train_power = sqrt(mean_squared_error(Y_train_pred[:,1], Y_train[:,1]))
    print('Train RMSE for Inference Time: %.3f' % rmse_train_infer)
    print('Train RMSE for Power Consumption: %.3f' % rmse_train_power)
    print('Test RMSE for Inference Time: %.3f' % rmse_infer)
    print('Test RMSE for Power Consumption: %.3f' % rmse_power)


def test_multi_target_regression_poly(data_x, data_y):
    print(len(data_x))
    n_train = int (len(data_x)*0.80)
    X_train, Y_train = data_x[:n_train], data_y[:n_train]
    X_test, Y_test = data_x[n_train:], data_y[n_train:]
    references = np.zeros_like(Y_test)

    for n in range(2):
        rgr = GradientBoostingRegressor(random_state=0)
        rgr.fit(X_train, Y_train[:, n])
        references[:,n] = rgr.predict(X_test)
    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    rgr.fit(X_train, Y_train)

    Y_train_pred = rgr.predict(X_train)
    Y_test_pred = rgr.predict(X_test)
    sequence_arr = np.arange(1,len(data_x)+1).reshape(len(data_x),1)
    print(len(Y_test_pred))
    print(len(sequence_arr[n_train:]))

    plot_result(sequence_arr[:n_train], Y_train[:,0], sequence_arr[n_train:],Y_test[:,0], Y_train_pred[:,0], Y_test_pred[:,0])
    plot_result(sequence_arr[:n_train], Y_train[:,1], sequence_arr[n_train:],Y_test[:,1], Y_train_pred[:,1], Y_test_pred[:,1])

    rmse_infer = sqrt(mean_squared_error(Y_test_pred[:,0], Y_test[:,0]))
    rmse_power = sqrt(mean_squared_error(Y_test_pred[:,1], Y_test[:,1]))
    rmse_train_infer = sqrt(mean_squared_error(Y_train_pred[:,0], Y_train[:,0]))
    rmse_train_power = sqrt(mean_squared_error(Y_train_pred[:,1], Y_train[:,1]))
    print('Train RMSE for Inference Time: %.3f' % rmse_train_infer)
    print('Train RMSE for Power Consumption: %.3f' % rmse_train_power)
    print('Test RMSE for Inference Time: %.3f' % rmse_infer)
    print('Test RMSE for Power Consumption: %.3f' % rmse_power)


def plot_result(X_train, Y_train, X_test,Y_test, Y_train_pred, Y_test_pred):
    pyplot.plot(X_train,Y_train_pred,'b',label='pred-train')
    pyplot.plot(X_test,Y_test_pred,'g',label='pred-test')
    pyplot.plot(X_train,Y_train,'rx',label='ground truth')
    pyplot.plot(X_test,Y_test,'rx')
    pyplot.legend(loc='best')
    pyplot.show()

dir_path = '/home/rick/Project_DNN/Experiment Result'
dataset = pd.read_csv(dir_path+'/23Nov_30Samples.csv')
dataset = dataset.values[:,:-2]
dataset = dataset.astype('float32')
data_x, data_y = dataset[:,:-2], dataset[:,-2:]
test_multi_target_regression_poly(data_x, data_y)
#rmse = sqrt(mean_squared_error(predict, test_y))
#print('Test RMSE: %.3f' % rmse)
