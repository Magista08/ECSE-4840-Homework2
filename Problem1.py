import pandas as pd
import numpy as np

OUTPUT = 0


def GetData(_FileName):
    # Test the file opened is correct or not
    ProcessedData = {
        'CO': [],
        'Tin Oxide': [],
        'Hydro Carbons': [],
        'Benzene': [],
        'Titania':[],
        'NOx': [],
        'Tungsten Oxide3': [],
        'NO2': [],
        'Tungsten Oxide4': [],
        'Indium Oxide': []
    }
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

    # Define the basic parameter
    LineLength = len(RowData.values[0])

    # Add the Row Data into different sector
    for i in RowData.values:
        DataPtr = 2
        for j in ProcessedData.keys():
            ProcessedData[j].append(i[DataPtr])
            DataPtr += 1
            if DataPtr >= LineLength:
                print("The excel's data: \n'{}'\n is out of range, Please check!".format(i))
                exit(-1)

    # Add the index num
    Index = range(len(RowData.values))
    ProcessedData['Index'] = Index
    '''
    # Test the row data is correct or not
    PrintData = RowData.values[0]
    print(PrintData)
    '''

    return ProcessedData


def StochasticGD(_SequenceNum):
    return


if __name__ == "__main__":
    OriginalData = GetData("AirQualityUCI.xlsx")

    print("Data All ready")
    print(OriginalData['CO'])
