from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial(data_x, data_y, split, init_degree=2, max_degree = 2):
    X_train, Y_train = data_x[:split], data_y[:split]
    X_test, Y_test = data_x[split:], data_y[split:]

    # explore various polynomial function degree
    pred = []
    degrees = np.arange(init_degree,max_degree+1)
    for i in range(len(degrees)):
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degrees[i])),
            ('linreg', LinearRegression(normalize=True))
        ])

        # fit the model and predict both train and test dataset
        model.fit(X_train, Y_train)
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        pred.append(Y_test_pred)

        # read intercept and coefficient values
        intercept = model.named_steps['linreg'].intercept_[0]
        coef = model.named_steps['linreg'].coef_[0]
        features = model.named_steps['poly'].get_feature_names()

        # assert that they are equal
        assert(len(coef) == len(features))

        # map coefficient to polynomial function
        estimated_inf_f = '{:+.2f}'.format(intercept)
        for j in range(0, len(coef)-1):
            if float ('{:+.1f}'.format(coef[j]).replace('\U00002013', '-')) != 0.0:
                estimated_inf_f += ' + {:+.1f} {}'.format(np.round(coef[j], decimals=2), features[j+1])

        # plot function we want to learn
        rmse_infer = sqrt(mean_squared_error(Y_test_pred[:,0], Y_test[:,0]))
        rmse_power = sqrt(mean_squared_error(Y_test_pred[:,1], Y_test[:,1]))

        print('Test RMSE for Inference Time: %.3f' % rmse_infer)
        print('Test RMSE for Power Consumption: %.3f' % rmse_power)
    return pred

def test_multi_target_regression(data_x, data_y):
    n_train = int (len(data_x)*0.80)
    X_train, Y_train = data_x[:n_train], data_y[:n_train]
    X_test, Y_test = data_x[n_train:], data_y[n_train:]

    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    rgr.fit(X_train, Y_train)
    Y_train_pred = rgr.predict(X_train)
    Y_test_pred = rgr.predict(X_test)

    plot_result(Y_train[:,0],Y_test[:,0], Y_train_pred[:,0], Y_test_pred[:,0])
    plot_result(Y_train[:,1],Y_test[:,1], Y_train_pred[:,1], Y_test_pred[:,1])

    rmse_infer = sqrt(mean_squared_error(Y_test_pred[:,0], Y_test[:,0]))
    rmse_power = sqrt(mean_squared_error(Y_test_pred[:,1], Y_test[:,1]))
    rmse_train_infer = sqrt(mean_squared_error(Y_train_pred[:,0], Y_train[:,0]))
    rmse_train_power = sqrt(mean_squared_error(Y_train_pred[:,1], Y_train[:,1]))

    print('Train RMSE for Inference Time: %.3f' % rmse_train_infer)
    print('Train RMSE for Power Consumption: %.3f' % rmse_train_power)
    print('Test RMSE for Inference Time: %.3f' % rmse_infer)
    print('Test RMSE for Power Consumption: %.3f' % rmse_power)


def plot_min_iteration(bay_y, rand_y):
    assert (len(bay_y) == len(rand_y))
    plt.plot(np.arange(0, bay_y.shape[0]),np.minimum.accumulate(bay_y[:,0]) ,'darkslateblue',label='Baysian Opt Inference')
    plt.plot(np.arange(0, rand_y.shape[0]),np.minimum.accumulate(rand_y[:,0]) ,'darkseagreen',label='Random Inference')
    plt.ylabel('fmin')
    plt.xlabel('Number of evaluated points')
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, bay_y.shape[0]),np.minimum.accumulate(bay_y[:,1]) ,'firebrick',label='Baysian Opt Power Consumption')
    plt.plot(np.arange(0, rand_y.shape[0]),np.minimum.accumulate(rand_y[:,1]) ,'royalblue',label='Random Power Consumption')
    plt.ylabel('fmin')
    plt.xlabel('Number of evaluated points')
    plt.legend()
    plt.show()

def plot_result(Y_train,Y_test, Y_train_pred, Y_test_pred):
    data_y = np.concatenate((Y_train, Y_test), axis=0)
    data_y = data_y.reshape((data_y.shape[0], 1))
    sequence_arr = np.arange(1,len(data_y)+1).reshape(len(data_y),1)

    pyplot.plot(sequence_arr,data_y, 'o', color='firebrick', label='ground truth')
    pyplot.plot(sequence_arr[:len(Y_train)],Y_train_pred,'cornflowerblue',label='pred-train')
    pyplot.plot(sequence_arr[len(Y_train):],Y_test_pred,'darkslategray',label='pred-test')
    pyplot.legend(loc='best')
    pyplot.show()

dir_path = 'C:/Path_to/Experiment Result'
bays_dataset = pd.read_csv(dir_path+'/Bays_VDD_40.csv')
rand_dataset = pd.read_csv(dir_path+'/Random_vgg_40.csv') #

# baysian Optimization Dataset
bays_dataset = bays_dataset.values[:,:-2]
bays_dataset = bays_dataset.astype('float32')

# random Optimization Dataset
rand_dataset = rand_dataset.values[:,:-2]
rand_dataset = rand_dataset.astype('float32')


# exclude CPU & GPU consumption
bays_data_x, bays_data_y = bays_dataset[:,:-2], bays_dataset[:,-2:]
rand_data_x, rand_data_y = rand_dataset[:,:-2], rand_dataset[:,-2:]

test_multi_target_regression(bays_data_x, bays_data_y)
test_multi_target_regression(rand_data_x, rand_data_y)

sequence_arr = np.arange(1,len(bays_data_x)+1).reshape(len(bays_data_x),1)
n_train = int (len(bays_data_x)*0.8)
max_degree = 3
degrees = np.arange(2,max_degree+1)
pyplot.plot(sequence_arr[n_train:],bays_data_y[n_train:,0],'o', color='navy', linewidth="2", marker='o', label='ground truth')
pyplot.plot(sequence_arr[n_train:],bays_data_y[n_train:,0], color='cornflowerblue', linewidth="2", label='ground truth')
pred_list = polynomial(bays_data_x, bays_data_y, n_train, init_degree=3, max_degree=max_degree)
colors = ["navy", "brown", "teal", "darkslategray"]
for i in range(len(pred_list)):
    pyplot.plot(sequence_arr[n_train:],pred_list[i][:,0],color=colors[i],linewidth=2,label='Degree %d' %degrees[i])
pyplot.legend(loc='best')
pyplot.show()

pyplot.plot(sequence_arr[n_train:],bays_data_y[n_train:,1],'o', color='navy', linewidth="2", marker='o', label='ground truth')
pyplot.plot(sequence_arr[n_train:],bays_data_y[n_train:,1], color='cornflowerblue', linewidth="2", label='ground truth')
colors = ["navy", "brown", "teal", "darkslategray"]
for i in range(len(pred_list)):
    pyplot.plot(sequence_arr[n_train:],pred_list[i][:,1],color=colors[i],linewidth=2,label='Degree %d' %degrees[i])
pyplot.legend(loc='best')
pyplot.show()

# plot minimum values explored along iteration
plot_min_iteration(bays_data_y, rand_data_y)