import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


# Functions with no relationships with GD
def GetData(_FileName):
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


def WholeData(_x, _y, _problem_seq):
    output = _x.copy()
    if _problem_seq == 1:
        output = np.insert(output, [2], _y, axis=1)
    return output


def GradientP1(_x, _y, _theta):
    (x_column, x_row) = _x.shape
    (y_column, y_row) = _y.shape

    output_gradient = np.zeros((x_row, 1))
    for i in range(x_column):
        output_gradient += 2 * ((np.dot(_x[i], _theta)) - _y[i]) * _x[i, :, None]
    return output_gradient / y_column


def GradientP2(_x, _y, _theta):
    return _theta


# Functions with relationships with ml


def CostFunction(_x, _y, _theta):
    output_cost = np.dot((np.dot(_x, _theta) - _y).T, (np.dot(_x, _theta) - _y))
    return output_cost[0][0] / _y.shape[0]


def GradientDescent(_x, _y, _problem_seq, _looping_times=0):
    # Basic Parameter Set up
    (x_col, x_row) = _x.shape
    learning_rate = 3e-10
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
            theta -= learning_rate * GradientP2(_x, _y, theta)

        # Record the data for the plotting
        end_time = time.time()
        used_time.append((end_time - beg_time))
        diff.append(CostFunction(_x, _y, theta))

    return theta, used_time, diff


def StochasticGD(_x, _y, _problem_seq, _looping_times=0):
    # Basic Parameter Set up
    (x_col, x_row) = _x.shape
    learning_rate = 3e-10
    if _looping_times == 0:
        _looping_times = _y.shape[0]

    # Prepare the output
    theta = np.array([[0.0]] * x_row)
    used_time, diff = ([] for i in range(2))

    combined_data = WholeData(_x, _y, _problem_seq)
    (cb_col, cb_row) = combined_data.shape

    beg_time = time.time()
    for i in range(_looping_times):
        test_Data_posi = np.random.randint(0, cb_col)
        print(test_Data_posi)

        x_sample = np.array([_x[test_Data_posi]])
        print(x_sample.shape)
        y_sample = np.array([_y[test_Data_posi]])

        if _problem_seq == 1:
            theta -= learning_rate * GradientP1(x_sample, y_sample, theta)
        else:
            theta -= learning_rate * GradientP2(x_sample, y_sample, theta)
        end_time = time.time()
        used_time.append((end_time - beg_time))
        diff.append(CostFunction(_x, _y, theta))

    return theta, used_time, diff


# Two main function begins at here
def Problem1Main():
    OriginalData = GetData("AirQualityUCI.xlsx")
    '''
    # Test the combination of the code
    print(OriginalData['x'].shape)
    output = WholeData(OriginalData['x'], OriginalData['y'], 1)
    print(output.shape)
    '''
    '''
    # Gradient Descent Output
    Problem1_gd_theta, Problem1_gd_time, Problem1_gd_diff = GradientDescent(OriginalData['x'], OriginalData['y'], 1,
                                                                            1000)

    # Draw the figure
    plt.figure()
    plt.plot(Problem1_gd_diff)
    plt.xlabel("Iteration Index")
    plt.ylabel("Objective Error")
    plt.title("The cost of gradient descent for problem 1")

    plt.savefig("Problem1_Gradient_Descent_Cost.jpg")

    plt.figure()
    plt.plot(Problem1_gd_time, Problem1_gd_diff)
    plt.xlabel("CPU time")
    plt.ylabel("Objective Error")
    plt.title("The CPU time of gradient descent for problem 1")

    plt.savefig("Problem1_Gradient_Descent_Time.jpg")
    '''
    # Stochastic Gradient Descent Output
    Problem1_sgd_theta, Problem1_sgd_time, Problem1_sgd_diff = StochasticGD(OriginalData['x'], OriginalData['y'], 1,
                                                                            1000)

    # Draw the figure
    plt.figure()
    plt.plot(Problem1_sgd_diff)
    plt.xlabel("Iteration Index")
    plt.ylabel("Objective Error")
    plt.title("The cost of stochmatic gradient descent for problem 1")

    plt.savefig("Problem1_Stochastic_Gradient_Descent_Cost.jpg")

    plt.figure()
    plt.plot(Problem1_sgd_time, Problem1_sgd_diff)
    plt.xlabel("CPU time")
    plt.ylabel("Objective Error")
    plt.title("The CPU time of stochmatic gradient descent for problem 1")

    plt.savefig("Problem1_Stochastic_Gradient_Descent_Time.jpg")
    '''
    # Test the validation of the dictionary
    print("Data All ready")
    print(np.array([[0.0]] * OriginalData['x'].shape[0]))
    print(CostFunction(OriginalData['x'], OriginalData['y'], np.array([[1]] * OriginalData['x'].shape[1])))
    '''


def main():
    Problem1Main()

main()
