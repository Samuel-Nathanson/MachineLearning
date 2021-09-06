import pandas
import math
import numpy as np
import sys
from scipy.stats import mode

validListTypes = (pandas.Series, list, np.ndarray)
#TODO: Clean up comments

# O(1) key lookup function
def keyExists(df, key):
    try:
        df[key]
    except KeyError:
        return False
    return True

def imputeData(dataFrame, columnId, nullIndicators=[np.NaN, '?'], imputation={"method": "mean"}, inplace=False):
    data = dataFrame if inplace else dataFrame.copy(deep=True)

    def meanReplacement():
        sum = 0
        count = 0
        for x in range(0,len(data[columnId])):
            try:
                if(not data[columnId].iloc[x] in nullIndicators):
                    sum += float(data[columnId].iloc[x])
                    count += 1
            except TypeError:
                sys.stderr.write(f"TypeError occurred while trying to compute mean of column {columnId}")
                exit(1)

        assert(not(count == 0))
        mean = sum / count
        data[columnId].replace(nullIndicators, mean, inplace=True)

    def constantReplacement():
        assert(keyExists(imputation, "constant"))
        for nullIndicator in nullIndicators:
            data[columnId].replace(nullIndicator, imputation["constant"], inplace=True)

    imputeMethods = {"mean": meanReplacement,
                     "constant": constantReplacement}

    # Validate arguments
    assert (type(data) == pandas.core.frame.DataFrame)
    assert (type(columnId) == str)
    assert (keyExists(data, columnId))
    assert (isinstance(nullIndicators, validListTypes))
    assert (len(nullIndicators) != 0)
    assert (type(imputation) == dict)
    assert (keyExists(imputation, "method"))
    assert (imputation["method"] in imputeMethods.keys())

    imputeMethods[imputation["method"]]()

    if(not inplace):
        return data

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
        data[columnId].replace(ordinal, integer, inplace=True)

    if(not inplace):
        return data

# dataFrame: Pandas Dataframe
# columnId: Column ID to replace nominal values with one-hot encoded values
# nominalValueList: List of nominal values which represent all possible categorical values in the column
def convertNominal(dataFrame, columnId, nominalValueList, inplace=False):
    data = dataFrame if inplace else dataFrame.copy(deep=True)
    assert(type(data) == pandas.core.frame.DataFrame)
    assert(type(columnId) == str)
    assert(isinstance(nominalValueList, validListTypes))
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
                # Extract Testing Set
                testingSet = fold[testingIndex:]
                folds.append([trainingSet, testingSet])

        # for each k
        #   ensure |Yi|/k >= 1
        #   add first |Yi|/k examples to set Sk
        # Caution: If |Sk| < 2 (or 3, with validation set included) then
        #   we won't be able to distribute values from Sk into training, testing, and validation sets
    return folds

# actualValues: list or nparray of values
# expectedValues: list or nparray of values (must be same length as actualValues)
def evaluateError(actualValues, expectedValues, method='MSE'):
    def countTruePositive(actualValues, expectedValues):
        tp = 0
        for index in range(0,len(actualValues)):
            if(actualValues[index] == 1 and expectedValues[index] == 1):
                tp += 1
        return tp

    def countFalsePositive(actualValues, expectedValues):
        fp = 0
        for index in range(0, len(actualValues)):
            if (actualValues[index] == 1 and expectedValues[index] == 0):
                fp += 1
        return fp

    def countTrueNegative(actualValues, expectedValues):
        tn = 0
        for index in range(0, len(actualValues)):
            if (actualValues[index] == 0 and expectedValues[index] == 0):
                tn += 1
        return tn

    def countFalseNegative(actualValues, expectedValues):
        fn = 0
        for index in range(0, len(actualValues)):
            if (actualValues[index] == 0 and expectedValues[index] == 1):
                fn += 1
        return fn

    def precision(actualValues, expectedValues):
        # Precision and Recall definitions from: https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
        tp = countTruePositive(actualValues, expectedValues)
        fp = countFalsePositive(actualValues, expectedValues)
        assert((tp + fp) != 0)
        return (tp/(tp + fp))

    def recall(actualValues, expectedValues):
        # Precision and Recall definitions from: https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
        tp = countTruePositive(actualValues, expectedValues)
        fn = countFalseNegative(actualValues, expectedValues)
        assert(tp+fn != 0)
        return (tp/(tp+fn))

    def MSE(actualValues, expectedValues):
        # diff = np.subtract(actualValues, expectedValues)
        # np.subtract is 50% slower than my method
        se = 0
        for index in range(0, len(actualValues)):
            se += (expectedValues[index] - actualValues[index]) ** 2
        mse = se / len(actualValues)
        return mse

    def MAE(actualValues, expectedValues):
        # diff = np.subtract(actualValues, expectedValues)
        # np.subtract is 50% slower than my method
        ae = 0
        for index in range(0, len(actualValues)):
            ae += math.fabs(expectedValues[index] - actualValues[index])
        mae = ae / len(actualValues)
        return mae

    def R2(actualValues, expectedValues):
        num = 0
        den = 0
        mean = np.mean(expectedValues)
        for index in range(0, len(actualValues)):
            num += (expectedValues[index] - actualValues[index]) ** 2
            den += (expectedValues[index] - mean) ** 2
        assert(den != 0)
        return 1 - (num / den)

    def pearsonCorr(actualValues, expectedValues):
        # Formula from https://study.com/academy/lesson/pearson-correlation-coefficient-formula-example-significance.html
        n = len(expectedValues)
        sumX = 0
        sumY = 0
        sumXsq = 0
        sumYsq = 0
        sumXY = 0

        for index in range(0, len(actualValues)):
            sumX += actualValues[index]
            sumY += expectedValues[index]
            sumXsq += actualValues[index] ** 2
            sumYsq += expectedValues[index] ** 2
            sumXY += actualValues[index] * expectedValues[index]

        num = n * sumXY - sumX * sumY
        den = (n*sumXsq - sumX**2) * (n*sumYsq - sumY**2)
        if(den <= 0):
            # Try to correct for floating point error
            denTerm1 = (n*sumXsq - sumX**2)
            denTerm2 = (n*sumYsq - sumY**2)
            if (n*sumXsq - sumX**2) < 10e-5:
                denTerm1 += 10e-5
            if (n*sumYsq - sumY**2) < 10e-5:
                denTerm2 += 10e-5
            den = denTerm1 * denTerm2

        pearsonCorrCoeff = num / math.sqrt(den)
        return pearsonCorrCoeff

    def f1(actualValues, expectedValues):
        p = precision(actualValues, expectedValues)
        r = recall(actualValues, expectedValues)
        num = p * r
        den = p + r
        assert(den != 0)
        f1Score = 2 * (num / den)
        return f1Score

    methods = { "precision": precision,
                "recall": recall,
                "MSE": MSE,
                "MAE": MAE,
                "R2": R2,
                "pearson": pearsonCorr,
                "f1": f1}

    assert(isinstance(actualValues, validListTypes))
    assert(isinstance(expectedValues, validListTypes))
    assert(len(actualValues) != 0)
    assert(len(actualValues) == len(expectedValues))
    assert(method in methods.keys())

    return methods[method](list(actualValues), list(expectedValues))

def naivePredictor(trainingSet, testingSet, predictorColId, method="regression"):
    assert(type(trainingSet) == pandas.DataFrame)
    assert(type(testingSet) == pandas.DataFrame)
    assert (keyExists(trainingSet, predictorColId))
    assert (keyExists(testingSet, predictorColId))

    def majority(trainingSet):
        return mode(trainingSet[predictorColId])

    def regress(trainingSet):
        return np.mean(trainingSet[predictorColId])

    methods = {
        "classification" : majority,
        "regression" : regress
    }

    assert(method in methods)
    return methods[method](trainingSet)



