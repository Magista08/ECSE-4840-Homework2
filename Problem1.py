import pandas as pd
import numpy as np
import time
from math import exp
import matplotlib.pyplot as plt


# Functions with no relationships with GD
def GetData1(_FileName):
    # Prepare for the output data
    ProcessedData = {
        'x': [],
        'y': []
    }

    # Test the file opened is correct or not
    try:
        OpenSignal = open(_FileName)
        OpenSignal.close()
    except FileNotFoundError:
        print("File is not found.")
        exit(1)
    except PermissionError:
        print("You don't have permission to access this file.")
        exit(-1)

    # Get the row data from the excel
    RowData = pd.read_excel('AirQualityUCI.xlsx')

    # Add the Row Data into different sector
    for i in RowData.values:
        for j in range(len(i)):
            if i[j] == -200:
                i[j] = 0.0

        ProcessedData['y'].append([i[5]])
        # ProcessedData['x'].append(list(i[2:5]) + list(i[6:]))
        ProcessedData['x'].append(list(i[2:5]) + list(i[6:]))

    for i in ProcessedData.keys():
        ProcessedData[i] = np.array(ProcessedData[i])
    # Add the index num
    '''
    # Test the row data is correct or not
    PrintData = RowData.values[0]
    print(PrintData)
    '''

    return ProcessedData


def GetData2():
    # Prepare for the output data
    ProcessedData = {
        'x': [],
        'y': []
    }

    # Test the file opened is correct or not
    try:
        OpenSignal = open("ionosphere.data")
        OpenSignal.close()
    except FileNotFoundError:
        print("File is not found.")
        exit(1)
    except PermissionError:
        print("You don't have permission to access this file.")
        exit(-1)

    # Get the row data from the excel
    FilePtr = open("ionosphere.data")
    for i in FilePtr:
        line_list = i.split(",")
        x_list = list()
        for j in range(len(line_list) - 1):
            x_list.append(float(line_list[j]))
        ProcessedData['x'].append(x_list)
        if line_list[-1] == 'b\n':
            ProcessedData['y'].append([-1.0])
        elif line_list[-1] == 'g\n':
            ProcessedData['y'].append([1.0])
        else:
            ProcessedData['y'].append([0.0])

    for i in ProcessedData.keys():
        ProcessedData[i] = np.array(ProcessedData[i])

    return ProcessedData


def WholeData(_x, _y, _problem_seq):
    output = _x.copy()
    if _problem_seq == 1:
        output = np.insert(output, [2], _y, axis=1)
    return output


def MiniBatch(_x, _y, batch_size):
    # Get the basic parameter
    whole_data = WholeData(_x, _y, 1)

    np.random.shuffle(whole_data)
    whole_data = whole_data[:batch_size + 1]
    y_mini_batch = whole_data[:, 3:4]
    x_mini_batch_1 = whole_data[:, :3]
    x_mini_batch_2 = whole_data[:, 4:]
    x_mini_batch = np.hstack((np.array(x_mini_batch_1),
                              np.array(x_mini_batch_2)))
    y_mini_batch = np.array(y_mini_batch)
    x_mini_batch = np.array(x_mini_batch)

    return x_mini_batch, y_mini_batch


def GradientP1(_x, _y, _theta):
    (x_column, x_row) = _x.shape
    (y_column, y_row) = _y.shape

    output_gradient = np.zeros((x_row, 1))
    for i in range(x_column):
        output_gradient += 2 * ((np.dot(_x[i], _theta)) - _y[i]) * _x[i, :, None]
    return output_gradient / y_column


def GradientP2(_x, _y, _theta, _lambda):
    (x_column, x_row) = _x.shape
    (y_column, y_row) = _y.shape

    output_gradient = np.zeros((x_row, 1))

    for i in range(x_column):
        output_gradient += (1 / (1 + exp(-_y[i, 0] * np.dot(_x[i], _theta)[0])) - 1) * _y[i, 0] * _x[i, :, None]
    return (output_gradient / y_column) + (_lambda * _theta)


# Functions with relationships with ml


def CostFunction(_x, _y, _theta):
    output_cost = np.dot((np.dot(_x, _theta) - _y).T, (np.dot(_x, _theta) - _y))
    return output_cost[0][0] / _y.shape[0]


def GradientDescent(_x, _y, _problem_seq, _looping_times=0, _lambda=0.0):
    # Basic Parameter Set up
    (x_col, x_row) = _x.shape
    learning_rate = 3e-10
    learning_rate_1 = 5.5e-4
    if _looping_times == 0:
        _looping_times = _y.shape[0]

    # Prepare the output
    theta = np.array([[0.0]] * x_row)
    used_time, diff = ([] for i in range(2))

    # Begin the ml
    beg_time = time.time()
    for i in range(_looping_times):
        # Update the slope rate
        if _problem_seq == 1:
            theta -= learning_rate * GradientP1(_x, _y, theta)
        else:
            theta -= learning_rate_1 * GradientP2(_x, _y, theta, _lambda)

        # Record the data for the plotting
        end_time = time.time()
        used_time.append((end_time - beg_time))
        diff.append(CostFunction(_x, _y, theta))

    return theta, used_time, diff


def StochasticGD(_x, _y, _problem_seq, _looping_times=0, _lambda=0.0):
    # Basic Parameter Set up
    (x_col, x_row) = _x.shape
    learning_rate = 3e-10
    learning_rate_1 = 5.5e-4
    if _looping_times == 0:
        _looping_times = _y.shape[0]

    # Prepare the output
    theta = np.array([[0.0]] * x_row)
    used_time, diff = ([] for i in range(2))

    whole_data = WholeData(_x, _y, _problem_seq)
    (wb_col, wb_row) = whole_data.shape

    beg_time = time.time()
    for i in range(_looping_times):
        test_Data_posi = np.random.randint(0, wb_col)

        x_sample = np.array([_x[test_Data_posi]])
        y_sample = np.array([_y[test_Data_posi]])

        if _problem_seq == 1:
            theta -= learning_rate * GradientP1(x_sample, y_sample, theta)
        else:
            theta -= learning_rate_1 * GradientP2(x_sample, y_sample, theta, _lambda)
        end_time = time.time()
        used_time.append((end_time - beg_time))
        diff.append(CostFunction(_x, _y, theta))

    return theta, used_time, diff


def MiniBatchStochasticGD(_x, _y, _problem_seq, _looping_times=0, _lambda=0.0):
    # Basic Parameter Set up
    (x_col, x_row) = _x.shape
    learning_rate = 3e-11
    learning_rate_1 = 5.5e-4
    if _looping_times == 0:
        _looping_times = _y.shape[0]

    # Prepare the output
    theta = np.array([[0.0]] * x_row)
    used_time, diff = ([] for i in range(2))

    # Begin the process
    beg_time = time.time()
    for i in range(_looping_times):
        x_sample, y_sample = MiniBatch(_x, _y, 49)
        if _problem_seq == 1:
            theta -= learning_rate * GradientP1(x_sample, y_sample, theta)
        else:
            theta -= learning_rate_1 * GradientP2(x_sample, y_sample, theta, _lambda)
        end_time = time.time()
        used_time.append((end_time - beg_time))
        diff.append(CostFunction(_x, _y, theta))

    return theta, used_time, diff


def DrawPlot(_x_data, _figure_name, _xlabel, _ylabel, _title, _y_data=list()):
    plt.figure()
    if _y_data == list():
        plt.plot(_x_data)
    else:
        plt.plot(_x_data, _y_data)

    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(_title)

    plt.savefig(_figure_name)


# Two main function begins at here
def Problem1Main():
    OriginalData = GetData1("AirQualityUCI.xlsx")

    # Gradient Descent Output
    Problem1_gd_theta, Problem1_gd_time, Problem1_gd_diff = GradientDescent(OriginalData['x'], OriginalData['y'], 1,
                                                                            1000)
    # Draw the figure
    DrawPlot(Problem1_gd_diff, "Problem1_Gradient_Descent_Cost.jpg", "Iteration Index", "Objective Error",
             "The cost of gradient descent for problem 1")
    DrawPlot(Problem1_gd_time, "Problem1_Gradient_Descent_Time.jpg", "CPU time", "Objective Error",
             "The CPU time of gradient descent for problem 1", Problem1_gd_diff)

    # Stochastic Gradient Descent Output
    Problem1_sgd_theta, Problem1_sgd_time, Problem1_sgd_diff = StochasticGD(OriginalData['x'], OriginalData['y'], 1,
                                                                            1000)
    # Draw the figure
    DrawPlot(Problem1_sgd_diff, "Problem1_Stochastic_Gradient_Descent_Cost.jpg", "Iteration Index", "Objective Error",
             "The cost of stochmatic gradient descent for problem 1")
    DrawPlot(Problem1_sgd_time, "Problem1_Stochastic_Gradient_Descent_Time.jpg", "CPU time", "Objective Error",
             "The CPU time of stochmatic gradient descent for problem 1", Problem1_sgd_diff)

    # Mini Batch Stochastic Gradient Descent Output
    Problem1_mbsgd_theta, Problem1_mbsgd_time, Problem1_mbsgd_diff = \
        MiniBatchStochasticGD(OriginalData['x'], OriginalData['y'], 1, 1000)

    # Draw the figure
    DrawPlot(Problem1_mbsgd_diff, "Problem1_MBS_Gradient_Descent_Cost.jpg", "Iteration Index", "Objective Error",
             "The cost of gradient descent for problem 1")
    DrawPlot(Problem1_mbsgd_time, "Problem1_MBS_Gradient_Descent_Time.jpg", "CPU time",
             "Objective Error",
             "The CPU time of mini batch stochmatic gradient descent for problem 1", Problem1_mbsgd_diff)


def Problem2Main():
    OriginalData = GetData2()

    # Gradient Descent Output
    Problem2_gd_theta, Problem2_gd_time, Problem2_gd_diff = GradientDescent(OriginalData['x'], OriginalData['y'], 2,
                                                                            1000, 0)
    # Draw the figure
    DrawPlot(Problem2_gd_diff, "Problem2_Gradient_Descent_Cost.jpg", "Iteration Index", "Objective Error",
             "The cost of gradient descent for problem 2")
    DrawPlot(Problem2_gd_time, "Problem2_Gradient_Descent_Time.jpg", "CPU time", "Objective Error",
             "The CPU time of gradient descent for problem 2", Problem2_gd_diff)

    # Stochastic Gradient Descent Output
    Problem2_sgd_theta, Problem2_sgd_time, Problem2_sgd_diff = StochasticGD(OriginalData['x'], OriginalData['y'], 2,
                                                                            1000, 0)
    # Draw the figure
    DrawPlot(Problem2_sgd_diff, "Problem2_Stochastic_Gradient_Descent_Cost.jpg", "Iteration Index", "Objective Error",
             "The cost of stochmatic gradient descent for problem 2")
    DrawPlot(Problem2_sgd_time, "Problem2_Stochastic_Gradient_Descent_Time.jpg", "CPU time", "Objective Error",
             "The CPU time of stochmatic gradient descent for problem 2", Problem2_sgd_diff)

    # Mini Batch Stochastic Gradient Descent Output
    Problem2_mbsgd_theta, Problem2_mbsgd_time, Problem2_mbsgd_diff = \
        MiniBatchStochasticGD(OriginalData['x'], OriginalData['y'], 2, 1000, 0)

    # Draw the figure
    DrawPlot(Problem2_mbsgd_diff, "Problem2_MBS_Gradient_Descent_Cost.jpg", "Iteration Index", "Objective Error",
             "The cost of gradient descent for problem 2")
    DrawPlot(Problem2_mbsgd_time, "Problem2_MBS_Gradient_Descent_Time.jpg", "CPU time",
             "Objective Error",
             "The CPU time of mini batch stochmatic gradient descent for problem 2", Problem2_mbsgd_diff)

    # Gradient Descent Output
    Problem2_gd_theta, Problem2_gd_time, Problem2_gd_diff = GradientDescent(OriginalData['x'], OriginalData['y'], 2,
                                                                            1000, 0.01)
    # Draw the figure
    DrawPlot(Problem2_gd_diff, "Problem2_Gradient_Descent_Cost_lambda.jpg", "Iteration Index", "Objective Error",
             "The cost of gradient descent for problem 2, lambda = 0.01")
    DrawPlot(Problem2_gd_time, "Problem2_Gradient_Descent_Time_lambda.jpg", "CPU time", "Objective Error",
             "The CPU time of gradient descent for problem 2, lambda = 0.01", Problem2_gd_diff)

    # Stochastic Gradient Descent Output
    Problem2_sgd_theta, Problem2_sgd_time, Problem2_sgd_diff = StochasticGD(OriginalData['x'], OriginalData['y'], 2,
                                                                            1000, 0.01)
    # Draw the figure
    DrawPlot(Problem2_sgd_diff, "Problem2_Stochastic_Gradient_Descent_Cost_lambda.jpg", "Iteration Index", "Objective Error",
             "The cost of stochmatic gradient descent for problem 2, lambda = 0.01")
    DrawPlot(Problem2_sgd_time, "Problem2_Stochastic_Gradient_Descent_Time_lambda.jpg", "CPU time", "Objective Error",
             "The CPU time of stochmatic gradient descent for problem 2, lambda = 0.01", Problem2_sgd_diff)

    # Mini Batch Stochastic Gradient Descent Output
    Problem2_mbsgd_theta, Problem2_mbsgd_time, Problem2_mbsgd_diff = \
        MiniBatchStochasticGD(OriginalData['x'], OriginalData['y'], 2, 1000, 0.01)

    # Draw the figure
    DrawPlot(Problem2_mbsgd_diff, "Problem2_MBS_Gradient_Descent_Cost_lambda.jpg", "Iteration Index", "Objective Error",
             "The cost of gradient descent for problem 2, lambda = 0.01")
    DrawPlot(Problem2_mbsgd_time, "Problem2_MBS_Gradient_Descent_Time_lambda.jpg", "CPU time", "Objective Error",
             "The CPU time of mini batch stochmatic gradient descent for problem 2, lambda =0.01", Problem2_mbsgd_diff)


def main():
    Problem1Main()
    Problem2Main()


main()
