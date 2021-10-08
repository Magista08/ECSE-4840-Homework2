import pandas as pd
import numpy as np

OUTPUT = 0


def GetData(FileName):
    # Test the file opened is correct or not
    ProcessedData = {
        'CO': [],
        'PT08.S1': [],

    }
    try:
        OpenSignal = open(FileName)
        OpenSignal.close()
    except FileNotFoundError:
        print("File is not found.")
        exit(1)
    except PermissionError:
        print("You don't have permission to access this file.")
        exit(-1)

    # Get the row data from the excel
    RowData = pd.read_excel('AirQualityUCI.xlsx')

    '''
    # Test the row data is correct or not
    PrintData = RowData.values[0]
    print(PrintData)
    '''

    return RowData


def StochasticGD():
    return


if __name__ == "__main__":
    OriginalData = GetData("AirQualityUCI.xlsx")

    print("Data All ready")
