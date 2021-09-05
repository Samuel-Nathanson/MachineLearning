import pandas
import math
import numpy as np

# O(1) key lookup function
def keyExists(df, key):
    try:
        df[key]
    except KeyError:
        return False
    return True

# data: Pandas Dataframe containing data
# columnId: Column ID to replace values with
# ordinalValueDict: Mapping of ordinal values to integer ordinal values
def convertToOrdinal(dataObj, columnId, ordinalValueDict, inplace=False):
    data = dataObj if inplace else dataObj.copy(deep=True)
    # Validate arguments
    assert(type(data) == pandas.core.frame.DataFrame)
    assert(type(columnId) == str)
    assert(type(ordinalValueDict) == dict)
    assert(keyExists(data, columnId))

    # Encode ordinal values as integers
    for ordinal, integer in ordinalValueDict.items():
        dataCopy[columnId].replace(ordinal, integer, inplace=True)

    if(not inplace):
        return data

# data: Pandas Dataframe containing data
# columnId: Column ID to replace nominal values with one-hot encoded values
# nominalValueList: List of nominal values which represent all possible categorical values in the column
def convertToNominal(dataObj, columnId, nominalValueList, inplace=False):
    data = dataObj if inplace else dataObj.copy(deep=True)
    assert(type(data) == pandas.core.frame.DataFrame)
    assert(type(columnId) == str)
    assert(type(nominalValueList) == list or type(nominalValueList) == np.ndarray)
    assert(keyExists(data, columnId))

    oneHotWidth = len(nominalValueList)
    oneHotFormat = "#0" + str(oneHotWidth+2) + "b"

    def oneHotEncode(integerValue):
        assert(type(integerValue) == int)
        return format(integerValue, oneHotFormat)

    # Create a mapping of nominal values to one-hot encoded in O(N) time.
    nominalValueDict = {}
    count = 0
    for nominalValue in nominalValueList:
        nominalValueDict[nominalValue] = oneHotEncode(pow(2,count))
        count += 1

    for nominal, oneHot in nominalValueDict.items():
        data[columnId].replace(nominal, oneHot, inplace=True)

    if(not inplace):
        return data

# data: Pandas Dataframe containing data - ** WILL SORT DATA **
# columnId: Column ID to replace nominal values with one-hot encoded values
# xargs: Dictionary of Extra Arguments
#   dMethod - Discretization Method ("frequency" or "equal-width")
#       Notes: This discretization method floors values
#           E.G with four bins over the range [-4,4], the values [-0.7, 0.9, 1.1, 1.9] would transform into [-2, 0, 1, 1]
#   bins - Number of bins to split data into
def discretize(dataObj, columnId, xargs={"dMethod": "frequency", "bins": 10}, inplace=False):
    data = dataObj if inplace else dataObj.copy(deep=True)
    assert(type(data) == pandas.core.frame.DataFrame)
    assert(type(columnId) == str)
    assert(type(xargs) == dict)
    assert(xargs["bins"] > 0)
    assert(keyExists(data, columnId))

    if(xargs["dMethod"] == "frequency"):
        samplesPerBin = math.ceil(len(data[columnId]) / xargs["bins"])
        data.sort_values(by=columnId, inplace=True)
        for index in range (0, len(data[columnId])):
            if(index % samplesPerBin == 0):
                currentBin = data[columnId].iloc[index]
            data[columnId].iat[index] = currentBin
    elif(xargs["dMethod"] == "equal-width"):
        valueRange = np.max(data[columnId]) - np.min(data[columnId])
        binWidth = valueRange / xargs["bins"]
        # Interesting bug occurs
        data[columnId] = data[columnId].apply(lambda x: math.floor((x/binWidth) - x/(2*binWidth)) * binWidth)
    else:
        raise ValueError("PreprocessingTK::discretize - dMethod argument must be one of 'frequency' or 'equal-width'")

    if(not inplace):
        return data


