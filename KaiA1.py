import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def hypothesis(X, theta):
    return np.dot(X, theta)


def cost(X, y, theta):
    J = 0.0
    for i in range(X.shape[0]):
        h = hypothesis(X[i], theta)
        J += np.dot((h - y[i]).transpose(), (h - y[i]))
    J /= y.shape[0]
    return J


def gradient(X, y, theta):
    grad = np.zeros((X.shape[1], 1))
    for i in range(X.shape[0]):
        h = hypothesis(X[i], theta)
        grad += 2 * (h - y[i]) * X[i, :, None]
    grad /= y.shape[0]
    return grad


def GD(X, y, learning_rate=0.03, max_iters=500, tolerance=1e-4):
    theta = np.zeros((X.shape[1], 1))
    error_list = []
    tc1 = time.time()
    time_list = []
    theta_old = theta.copy()

    for itr in range(max_iters):
        theta = theta_old - learning_rate * gradient(X, y, theta)
        if (np.linalg.norm(theta - theta_old) < tolerance):
            break
        time_list.append(time.time() - tc1)
        theta_old = theta.copy()
        error_list.append(cost(X, y, theta))
    return theta, error_list, time_list


def SGD(X, y, learning_rate=0.03, max_iters=500, tolerance=1e-4):
    theta = np.zeros((X.shape[1], 1))
    error_list = []
    tc1 = time.time()
    time_list = []
    theta_old = theta.copy()
    data = np.hstack((X, y))

    for itr in range(max_iters):
        index = np.random.randint(0, data.shape[0])
        X_mini = data[index, None, :-1]
        y_mini = data[index, -1].reshape((-1, 1))
        print(X_mini.shape)
        print(y_mini.shape)
        theta = theta_old - learning_rate * gradient(X_mini, y_mini, theta)

        if (np.linalg.norm(theta - theta_old) < tolerance):
            break
        time_list.append(time.time() - tc1)
        theta_old = theta.copy()
        error_list.append(cost(X_mini, y_mini, theta))

    return theta, error_list, time_list


def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    # n_minibatches = data.shape[0] // batch_size
    i = 0

    mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
    X_mini = mini_batch[:, :-1]
    Y_mini = mini_batch[:, -1].reshape((-1, 1))
    mini_batches.append((X_mini, Y_mini))
    return mini_batches


def MiniSGD(X, y, learning_rate=0.03, batch_size=35, max_iters=500, tolerance=1e-4):
    theta = np.zeros((X.shape[1], 1))
    error_list = []
    tc1 = time.time()
    time_list = []
    theta_old = theta.copy()
    for itr in range(max_iters):
        mini_batches = create_mini_batches(X, y, batch_size)
        X_mini, y_mini = mini_batches[0]
        theta = theta_old - learning_rate * gradient(X_mini, y_mini, theta)
        if (np.linalg.norm(theta - theta_old) < tolerance):
            break
        time_list.append(time.time() - tc1)
        theta_old = theta.copy()
        error_list.append(cost(X_mini, y_mini, theta))

    return theta, error_list, time_list


from sklearn.preprocessing import MinMaxScaler

# data processing
# data1 = pd.read_csv('AirQualityUCI.csv', sep=",",skip_blank_lines=True)
data1 = pd.read_excel('AirQualityUCI.xlsx')

y_train = data1.iloc[:9357, 5].values
data = data1.drop('C6H6(GT)', axis=1)
X_train = data.iloc[:9357, 2:14].values


X_train = MinMaxScaler().fit_transform(X_train)
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

y_train = np.array(y_train).reshape(-1, 1)
y_train = MinMaxScaler().fit_transform(y_train)

tc1 = time.time()
theta, error_list, time_list = GD(X_train, y_train, max_iters=100)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])

# visualising gradient descent
plt.figure(1)
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("GD")

# visualising gradient descent
plt.figure(2)
plt.plot(time_list)
plt.xlabel("Time (second)")
plt.ylabel("Cost")
plt.title("GD")

# SGD batch size = 1
tc1 = time.time()
theta, error_list, time_list = SGD(X_train, y_train, max_iters=100)

# visualising gradient descent
plt.figure(3)
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("SGD")

# visualising gradient descent
plt.figure(4)
plt.plot(time_list)
plt.xlabel("Time (second)")
plt.ylabel("Cost")
plt.title("SGD")
'''
# Mini batch SGD
tc1 = time.time()
theta, error_list, time_list = MiniSGD(X_train, y_train, max_iters=100)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])

# visualising gradient descent
plt.figure(5)
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Mini Batch SGD")

plt.figure(6)
plt.plot(time_list, error_list)
plt.xlabel("Time (second)")
plt.ylabel("Cost")
plt.title("Mini Batch SGD")
'''
plt.show()
