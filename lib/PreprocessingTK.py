import pandas
import math
import numpy as np

#TODO: Clean up comments

# O(1) key lookup function
def keyExists(df, key):
    try:
        df[key]
    except KeyError:
        return False
    return True

# dataFrame: Pandas Dataframe
# columnId: Column ID to replace values with
# ordinalValueDict: Mapping of ordinal values to integer ordinal values
def convertToOrdinal(dataFrame, columnId, ordinalValueDict, inplace=False):
    data = dataFrame if inplace else dataFrame.copy(deep=True)
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

# dataFrame: Pandas Dataframe
# columnId: Column ID to replace nominal values with one-hot encoded values
# nominalValueList: List of nominal values which represent all possible categorical values in the column
def convertToNominal(dataFrame, columnId, nominalValueList, inplace=False):
    data = dataFrame if inplace else dataFrame.copy(deep=True)
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

# dataFrame: Pandas Dataframe - ** WILL SORT DATA iLoc VALUES **
# columnId: Column ID to discretize values in
# xargs: Dictionary of Extra Arguments
#   dMethod - Discretization Method ("frequency" or "equal-width")
#       Notes: This discretization method floors values
#           E.G with four bins over the range [-4,4], the values [-0.7, 0.9, 1.1, 1.9] would transform into [-2, 0, 1, 1]
#   bins - Number of bins to split data into
# inplace: Set
def discretize(dataFrame, columnId, xargs={"dMethod": "frequency", "bins": 10}, inplace=False):
    data = dataFrame if inplace else dataFrame.copy(deep=True)
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
        # Bug occurs when bins=2 and max of column = valueRange. Subtract by (x/(2*bins)) before flooring,
        # This consistently avoids the bug while keeping the value in the same bin.
        # TODO: Test with negative values
        data[columnId] = data[columnId].apply(lambda x: math.floor((x/binWidth) - x/(2*binWidth)) * binWidth)
    else:
        raise ValueError("PreprocessingTK::discretize - dMethod argument must be one of 'frequency' or 'equal-width'")

    if(not inplace):
        return data

# trainingSetDF - Pandas DataFrame containing training set
# testingSetDF - Pandas DataFrame containing testing set
# columnId: Column ID to replace nominal values with one-hot encoded values
# @returns tuple of values (trainingSet, testingSet) if inplace=False. None otherwise.
def standardize(trainingSetDF, testingSetDF, columnId, inplace=False):
    trainingSet = trainingSetDF if inplace else trainingSetDF.copy(deep=True)
    testingSet = testingSetDF if inplace else testingSetDF.copy(deep=True)
    assert(type(trainingSet) == pandas.core.frame.DataFrame)
    assert(type(testingSet) == pandas.core.frame.DataFrame)
    assert(type(columnId) == str)
    assert(keyExists(trainingSet, columnId))
    assert(keyExists(testingSet, columnId))

    trainingColumnMean = np.mean(trainingSet[columnId])
    trainingColumnStd = np.std(trainingSet[columnId])
    trainingSet[columnId] = trainingSet[columnId].apply(lambda x: (x - trainingColumnMean) / trainingColumnStd)
    testingSet[columnId] = testingSet[columnId].apply(lambda x: (x - trainingColumnMean) / trainingColumnStd)

    if(not inplace):
        return trainingSet, testingSet

# partition: Partitions the dataset into k partitions, each of which
#   contains a train, test, and (optionally) validation set.
# dataFrame: Pandas DataFrame
# k: Number of folds for cross validation
# classificationColumn: ColumnId containing labels - Set to None if not classification task
# includeValidationSet: Boolean True/False - If true, include a validation set within each of the folds
# proportions: 2-Tuple or 3-Tuple determining ratios of data to use for training, testing, and validation
#   Defaults to 60% training, 20% testing, and 20% validation
def partition(dataFrame, k, classificationColumnId=None, includeValidationSet=True, proportions=(0.6,0.2,0.2)):
    assert(type(dataFrame) == pandas.core.frame.DataFrame)
    assert(type(k) == int)
    if(classificationColumnId):
        assert(type(classificationColumnId) == str)
        assert(keyExists(dataFrame, classificationColumnId))
    if(includeValidationSet):
        assert((len(proportions) == 3 and includeValidationSet) or (len(proportions == 2) and not includeValidationSet))
        assert(np.sum(proportions) == 1.0)

    foldSize = math.ceil(dataFrame.shape[0] / k)
    folds = []

    if(classificationColumnId):
        print('Classification Task')
        # first, partition training set based on y-label into sets of size [|Y0|,|Y1|...,|YN|]
    else:
        for foldNum in range(0, k):
            fold = dataFrame[foldNum*foldSize:(foldNum+1)*foldSize]

            # Extract the training set
            testingIndex = math.floor(proportions[0] * len(fold))
            trainingSet = fold[:testingIndex]

            if(includeValidationSet):
                validationIndex = math.floor(proportions[1] * len(fold) + testingIndex)
                # Extract Testing Set
                testingSet = fold[testingIndex:validationIndex]

                # Extract Validation Set
                validationSet = fold[validationIndex:]
                folds.append([trainingSet, testingSet, validationSet])
            else:
                testingSet = fold[testingIndex:]
                folds.append([trainingSet, testingSet])

        # for each k
        #   ensure |Yi|/k >= 1
        #   add first |Yi|/k examples to set Sk
        # Caution: If |Sk| < 2 (or 3, with validation set included) then
        #   we won't be able to distribute values from Sk into training, testing, and validation sets
    return folds

