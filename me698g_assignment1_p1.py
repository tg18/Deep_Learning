### Ayush Bisen - 21105025 ###

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split


def problem1_without_regularizer(degree, iteration, neu):

    ## INITIALIZATION ##

    data = np.genfromtxt('Sinedata.dat')

    x_data = data[0, :]
    y_data = data[1, :]

    w = np.ones(degree, dtype=np.float64)
    w.reshape((degree, 1))

    ## CALCULATION OF PHI MATRIX ##

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=143)
    phi = np.zeros((len(X_train), degree))
    for i in range(degree):
        for j in range(len(X_train)):
            phi[j][i] = X_train[j] ** i

    grad_error = np.zeros(degree, dtype=np.float64)

    ## COMPUTATION OF OPTIMAL WEIGHTS ##

    for n in range(iteration):
        for i in range(degree):
            grad_error[i] = np.matmul(np.matmul(phi, w) - y_train, phi[:, i])
        grad_error.reshape((degree, 1))
        w = w - neu * grad_error

    ## TESTING ##

    Y_pred = np.zeros(len(y_test))
    phi2 = np.zeros((len(X_test), degree))
    for i in range(degree):
        for j in range(len(X_test)):
            phi2[j][i] = X_test[j] ** i
    Y_pred = np.matmul(phi2, w)

    mae = sklearn.metrics.mean_absolute_error(y_test, Y_pred)

    return (X_test, Y_pred, y_test, mae)


def problem1_with_regularizer(degree, iteration, neu):

    ## INITIALIZATION ##

    data = np.genfromtxt('Sinedata.dat')

    x_data = data[0, :]
    y_data = data[1, :]
    alpha = 10  # regularizer constant
    w = np.ones(degree, dtype=np.float64)
    w.reshape((degree, 1))

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=143)

    ## CALCULATION OF PHI MATRIX ##

    phi = np.zeros((len(X_train), degree))
    for i in range(degree):
        for j in range(len(X_train)):
            phi[j][i] = X_train[j] ** i

    grad_error = np.zeros(degree, dtype=np.float64)

    ## COMPUTATION OF OPTIMAL WEIGHTS ##

    for n in range(iteration):
        for i in range(degree):
            grad_error[i] = np.matmul(np.matmul(phi, w) - y_train, phi[:, i]) + alpha * w[i]  # changes for a regularizer term
        grad_error.reshape((degree, 1))
        w = w - neu * grad_error

    ## TESTING ##

    Y_pred = np.zeros(len(y_test))
    phi2 = np.zeros((len(X_test), degree))
    for i in range(degree):
        for j in range(len(X_test)):
            phi2[j][i] = X_test[j] ** i
    Y_pred = np.matmul(phi2, w)

    mae = sklearn.metrics.mean_absolute_error(y_test, Y_pred)

    return (X_test, Y_pred, y_test, mae)


def problem1_with_gaussian(degree, iteration, neu):

    ## INITIALIZATION ##

    def gaussian(x, mean, sd):
        return np.exp(-(x - mean) ** 2 / (2 * (sd ** 2)))

    data = np.genfromtxt('Sinedata.dat')

    x_data = data[0, :]
    y_data = data[1, :]
    mean = []
    l = 0.0
    for i in range(degree):
        mean.append(l)
        l = l + 0.2
    # l=np.arange(0,1,1/(degree+1))

    sd = 0.2
    w = np.zeros(degree, dtype=np.float64)
    w.reshape((degree, 1))

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=0)

    ## CALCULATION OF PHI MATRIX ##

    phi = np.ones((len(X_train), degree))
    for i in range(1, degree):
        for j in range(len(X_train)):
            phi[j][i] = gaussian(X_train[j], mean[i], sd)

    grad_error = np.zeros(degree)

    ## COMPUTATION OF OPTIMAL WEIGHTS ##

    for n in range(iteration):
        for i in range(degree):
            grad_error[i] = np.matmul(np.matmul(phi, w) - y_train, phi[:, i])
        grad_error.reshape((degree, 1))
        w = w - neu * grad_error

    ## TESTING ##

    Y_pred = np.zeros(len(y_test))
    phi2 = np.ones((len(X_test), degree))
    for i in range(1, degree):
        for j in range(len(X_test)):
            phi2[j][i] = gaussian(X_test[j], mean[i], sd)
    Y_pred = np.matmul(phi2, w)

    mae = sklearn.metrics.mean_absolute_error(y_test, Y_pred)

    return (X_test, Y_pred, y_test, mae)


### Note: Kindly choose the number of degree of the polynomial, number of iterations and learning rate for which you would like to see the result. ###

degree = [3, 4, 5, 6]  # degree of the polynomial approximated
iterations = [500, 1000, 1500]  # number of iterations performed
neu = [0.0005, 0.00005, 0.00005]  # learning rate

for i in range(len(degree)):
    ### Note: Kindly uncomment the function, for which you would like to see the results.  ###

    #(x_test,y_pred,y_test,mae)=problem1_without_regularizer(degree[i],iterations[1],neu[0])

    #(x_test, y_pred, y_test, mae) = problem1_with_regularizer(degree[i], iterations[1],neu[0])

    (x_test, y_pred, y_test, mae) = problem1_with_gaussian(degree[i], iterations[1], neu[0])

    plt.scatter(x_test, y_pred, label='y_pred with degree=' + str(degree[i]))
    print('Mean Absolute error for ', degree[i], 'degree polynomial approximation for',iterations[1], 'iterations is', mae)

plt.scatter(x_test, y_test, marker='s', label='target')
plt.legend(scatterpoints=1)
plt.title('PROBLEM 1')
plt.xlabel('testing sample')
plt.ylabel('values')
plt.show()
