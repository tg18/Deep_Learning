### Ayush Bisen - 21105025 ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

def problem2_with_linear_regression(iteration,neu):

    ## INITIALIZATION ##

    ### Note : housing data has been modified and a column has been added at the beggining of the dataset with values as
    #          "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "price" ,kindly use the updated housing dataset given in the assignment file###

    columns = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "price"]
    data = pd.read_csv("housing.csv", usecols=columns)
    y_data = data.pop('price')

    parameters = 14

    w = np.ones(parameters, dtype=np.float64)
    w.reshape((parameters, 1))

    X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=0.1, random_state=143)
    x1 = X_test['p8']
    x1 = np.array(x1)

    X0 = np.ones((len(X_train)))

    X_train.insert(loc=0, column='basis', value=X0)

    X_train = X_train.to_numpy(dtype='float32')

    grad_error = np.zeros(parameters, dtype=np.float64)

    ## COMPUTATION OF OPTIMAL WEIGHTS ##

    for n in range(iteration):
        for i in range(parameters):
            grad_error[i] = np.matmul(np.matmul(X_train, np.transpose(w)) - y_train, X_train[:, i])
        grad_error.reshape((parameters, 1))
        w = w - neu * grad_error

    ## TESTING ##

    Y_pred = np.zeros(len(y_test))
    X0 = np.ones((len(X_test)))

    X_test.insert(loc=0, column='basis', value=X0)

    Y_pred = np.matmul(X_test, w)

    mae = sklearn.metrics.mean_absolute_error(y_test, Y_pred)

    return (x1, Y_pred, y_test, mae)

def problem2_with_radial_gaussian_function(iterations,neu):

    ## INITIALIZATION ##

    def gaussian(x, mean, cov_inv):
        return np.exp(-np.matmul(np.matmul(np.transpose(x - mean), cov_inv), (x - mean)) / 2)

    columns = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "price"]
    data = pd.read_csv("housing.csv", usecols=columns)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    y_data = data[:, 13]
    data = data[:, 0:13]

    parameters = 13
    degree = 4

    w = np.ones(degree, dtype=np.float64)
    w.reshape((degree, 1))

    ## TRAINING ##

    X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=0.1, random_state=143)

    X_train = pd.DataFrame(X_train,columns=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13"])
    covariance = X_train.cov()
    cov_inv = np.linalg.inv(covariance)
    X_train = np.array(X_train)
    x1 = X_test[:, 7]
    x1 = np.array(x1)

    kmeans = MiniBatchKMeans(n_clusters=degree - 1, random_state=0, batch_size=6, max_iter=10).fit(X_train)
    mean = kmeans.cluster_centers_

    mean0 = np.ones((1, parameters))
    mean0.reshape((1, parameters))
    mean = np.insert(mean, 0, mean0, axis=0)
    phi = np.ones((len(X_train), degree))

    ## CALCULATION OF PHI MATRIX ##

    for i in range(1, degree):
        for j in range(len(X_train)):

            phi[j][i] = gaussian(X_train[j, :].reshape(-1, 1), mean[i].reshape(-1, 1), cov_inv)

    grad_error = np.zeros(degree, dtype=np.float64)

    ## OPTIMIZATION OF WEIGHT VALUES ##

    for n in range(iterations):
        for i in range(degree):
            grad_error[i] = np.matmul(np.matmul(phi, w) - y_train, phi[:, i])
        grad_error.reshape((degree, 1))
        w = w - neu * grad_error


    Y_pred = np.zeros(len(y_test))
    phi2 = np.ones((len(X_test), degree))
    for i in range(1, degree):
        for j in range(len(X_test)):
            phi2[j][i] = gaussian(X_test[j, :].reshape(-1, 1), mean[i, :].reshape(-1, 1), cov_inv)

    ## TESTING ##

    Y_pred = np.matmul(phi2, w)

    mae = sklearn.metrics.mean_absolute_error(y_test, Y_pred)

    return (x1, Y_pred, y_test, mae)



### Note: Kindly choose the number of iterations and learning rate for which you would like to see the result. ###

iterations = [1000, 10000]  # number of iterations performed
neu = [0.001,0.00005,0.00000001]  # learning rate

for i in range(len(iterations)):

    ### Note: Kindly uncomment the function, for which you would like to see the results.  ###

    #(x1,y_pred,y_test,mae)=problem2_with_linear_regression(iterations[i],neu[2])

    (x1, y_pred, y_test, mae) = problem2_with_radial_gaussian_function(iterations[i], neu[0])

    plt.scatter(x1, y_pred, label='degree=' + str(iterations[i]))
    print('Mean Absolute error for ', iterations[i], 'iterations for learning rate',neu[0], 'is', mae)

plt.scatter(x1,y_test,c='blue',marker='s',label='target')
plt.legend(scatterpoints=1)
plt.title('PROBLEM 2')
plt.xlabel('testing sample values for parameter 8')
plt.ylabel('values')
plt.show()