import pandas
from lib.SNUtils import keyExists, distanceEuclideanL2, validListTypes
from lib.PreprocessingTK import *
from scipy.stats import mode
from numpy import mean, array, append, inf
import numpy as np
import sys

'''
chooseBestK: Tuning Function to find the best K
@param1 Pandas.DataFrame - validationSet - Validation set to test data on
@param2 str: classPredictionValue - If class, choose the class prediction value to predict on
@param3 str: className - Class name / column
@param4 int maxK: Maximum K to choose
@return int Best K value
'''
def chooseBestK(validationSet: pandas.DataFrame, classPredictionValue: str=None, className: str=None,maxK: int=7):
    print("Tuning K hyperparameter...")
    numFolds = 5
    validationFolds = partition(validationSet, numFolds, classificationColumnId=className)

    bestK = 1
    bestKAccuracy = 0.0
    for k in range(1, maxK+1, 2):
        foldAccuracies = []
        for i in range(0, numFolds):
            # Separate training and testing sets
            testingSet = validationFolds.pop(i)
            trainingSet = pandas.concat(validationFolds, ignore_index=True)
            validationFolds.insert(i, testingSet)

            predicted_scores = []
            for x in range(0, len(testingSet)):
                sys.stdout.write(f'\rk={k}, fold={i}/{numFolds-1}, query={x}/{len(testingSet)-1}')
                prediction = predict(k, trainingSet, testingSet.iloc[x].drop(labels=className), className)
                predicted_scores.append(prediction)

            foldAccuracy = evaluateError(predicted_scores, testingSet[className], method="accuracy", classLabel=classPredictionValue)
            foldAccuracies.append(foldAccuracy)
        meanFoldAccuracy = np.mean(foldAccuracy)
        if(meanFoldAccuracy > bestKAccuracy):
            bestKAccuracy = meanFoldAccuracy
            bestK = k
        print(f" | acc={meanFoldAccuracy}")
    return bestK


'''
findNearestNeighbors: Finds the k nearest neighbors 
Runs in O(N*Log(N)) Time
@param1 int: k - Number of neighbors to return
@param2 pandas.DataFrame : trainData - Training data to work with
@param3 query: (Series, list, ndarray) - Query of feature values to make a prediction for
@param4 str: yColumnId - Column ID of class column in training data
@return list of tuples - First entry in the tuple corresponds with the iLoc of the point in the dataFrame, second entry corresponds to the distance
'''
def findKNearestNeighbors(k: int, trainData: pandas.DataFrame, query : validListTypes, yColumnId: str):
    assert(type(k) == int)
    assert(type(trainData) == pandas.DataFrame)
    assert(isinstance(query, validListTypes))
    assert(type(yColumnId) == str)
    assert(keyExists(trainData, yColumnId))

    distance = distanceEuclideanL2
    numExamples = trainData.shape[0]
    p1 = query
    distances = []

    for i in range(0, numExamples):
        p2 = trainData.iloc[i]
        p2.pop(yColumnId)
        distances.append([i, distance(p1, p2)])

    sortedNeighbors = sorted(distances, key=lambda disTuple: disTuple[1])

    return sortedNeighbors[0:k]


'''
kNNEditTrainingSet - Edits the training set by removing misclassified points
@param1 pandas.DataFrame : trainData - Training Data Set.
@param2 - 
@param2 str: yColumnId - Y Column ID to classify on
@param3 int: k - Number of neighbors for KNN classification
'''
def kNNEditTrainingSet(trainDataFrame: pandas.DataFrame, \
                       validationData: pandas.DataFrame, \
                       yColumnId: str, \
                       classPredictionValue: str=None, \
                       k: int=3, \
                       epsilon: float=0.1,
                       doClassification: bool=True,
                       inplace: bool=False):

    trainData = trainDataFrame if inplace else trainDataFrame.copy(deep=True)

    assert (type(trainData) == pandas.DataFrame)
    assert (type(yColumnId) == str)
    assert (keyExists(trainData, yColumnId))
    assert (type(k) == int)

    prevScore = -1.0
    currScore = 0.0
    prevSize = len(trainData) + 1
    currSize = len(trainData)
    # If the first condition evaluates to false and the second condition evaluates to true,
    # could this degrade performance?
    while(currScore > prevScore or currSize < prevSize):
        idx = 0
        predictions = []
        while idx < len(trainData):
            query = trainData.iloc[idx]
            query.pop(yColumnId)

            sys.stdout.write(f'\rquery={idx}/{len(trainData) - 1}')
            sys.stdout.flush()

            prediction = predict(k, trainData, query, yColumnId, doClassification)
            predictions.append(prediction)

            idx += 1

        indicesToDrop = []
        for i in range(0, len(predictions)):
            shouldDrop = (predictions[i] != trainData.iloc[i][yColumnId]) \
                if doClassification else (np.abs(predictions[i] - trainData.iloc[i][yColumnId]) > epsilon)
            if shouldDrop:
                indicesToDrop.append(i)

        trainData = trainData = trainData.drop(trainData.index[indicesToDrop])

        predictions = []
        for i in range(0, len(validationData)):
            validationPoint = validationData.iloc[i].copy(deep=True)
            validationPoint.pop(yColumnId)
            prediction = predict(3, trainData, validationPoint, yColumnId, doClassification)
            predictions.append(prediction)

        prevSize = currSize
        currSize = len(trainData)

        prevScore = currScore

        if (doClassification):
            currScore = evaluateError(predictions, validationData[yColumnId], method="accuracy",
                                             classLabel=classPredictionValue)
        else:
            currScore = -1 * evaluateMSE(trainData, validationData, yColumnId)
        print(f" | score={currScore}")



    if(not inplace):
        return trainData

'''
EvaluateMSE
@param1 : pandas.DataFrame - trainData
@param2 : pandas.DataFrame - testData
@param3 : str - yColumnId 
@return : float - accuracy 
'''
def evaluateMSE(trainData: pandas.DataFrame, testData: pandas.DataFrame, yColumnId: str):
    assert (type(trainData) == pandas.DataFrame)
    assert (type(testData) == pandas.DataFrame)
    assert(type(yColumnId) == str)
    assert (keyExists(trainData, yColumnId))

    predictions = []
    for i in range(0, len(testData)):
        testPoint = testData.iloc[i].copy(deep=True)
        testPoint.pop(yColumnId)
        prediction = predict(1, trainData, testPoint, yColumnId)
        predictions.append(prediction)

    mse = evaluateError(predictions, testData[yColumnId], method="MSE")
    return mse

'''
kNNCondenseTrainingSet - Condenses the training set based on Hart's Algorithm
@param1 pandas.DataFrame: trainData - Training data to draw from
@param2 str: yColumnId - Class column ID
@param3 bool: doClassification=True - True if this data set represents a classification task
@return pandas.DataFrame corresponding to condensed dataset
'''
def kNNCondenseTrainingSet(trainData: pandas.DataFrame, yColumnId: str, doClassification=True, epsilon=0.1):
    assert (type(trainData) == pandas.DataFrame)
    assert(type(yColumnId) == str)
    assert (keyExists(trainData, yColumnId))

    # Randomize indices of training data.
    # Run KNN Condensation Algorithm
    XTrainData = trainData.sample(frac=1, random_state=0).reset_index(drop=True).copy(deep=True)
    minSetZ = pandas.DataFrame(columns=XTrainData.columns).append(XTrainData.iloc[0])

    prevSizeZ = 0
    newSizeZ = 1

    while(newSizeZ != prevSizeZ):
        indicesToDrop = []
        for i in range(1, XTrainData.shape[0]):
            query = XTrainData.iloc[i]
            queryWithoutClass = query.copy(deep=True)
            queryWithoutClass.pop(yColumnId)
            nearestNeighborList = findKNearestNeighbors(1, minSetZ, queryWithoutClass, yColumnId)
            # findNearestNeighbors returns a list of tuples (neighborIndex, distance)
            nearestNeighborIdx = nearestNeighborList[0][0]
            # Create DataFrame with nearest neighbor
            nearestNeighbor = pandas.DataFrame(columns=XTrainData.columns).append(minSetZ.iloc[nearestNeighborIdx])
            nearestNeighborVal = nearestNeighbor.iloc[0][yColumnId]
            queryVal = query[yColumnId]

            if(doClassification):
                if(not (nearestNeighborVal == queryVal)):
                    minSetZ = minSetZ.append(query)
                    indicesToDrop.append(i)
            else:
                SE = (queryVal - nearestNeighborVal) ** 2
                if (SE > epsilon):
                    minSetZ = minSetZ.append(query)
                    indicesToDrop.append(i)

        prevSizeZ = newSizeZ
        newSizeZ = minSetZ.shape[0]
        XTrainData.drop(XTrainData.index[indicesToDrop], inplace=True)

    return minSetZ

'''
predict: kNN Prediction for a query X using training dataset T
@param1 int: k - Number of neighbors to use for KNN algorithm
@param2 pandas.DataFrame : trainData - Training Set
@param3 (Series, list, ndarray): query - Query to search for class from - Must not include class label
@param4: str: yColumnId - Class column ID
@param5 bool: classify=True - True if classification prediction, False if regression prediction
'''
def predict(k: int, trainData: pandas.DataFrame, query : validListTypes, yColumnId: str, classify:bool=True):
    assert(type(k) == int)
    assert(type(trainData) == pandas.DataFrame)
    assert(isinstance(query, validListTypes))
    assert(type(yColumnId) == str)
    assert(type(classify) == bool)
    assert(keyExists(trainData, yColumnId))

    nearestKNeighbors = findKNearestNeighbors(k, trainData, query, yColumnId)

    if(classify):
        nearestKNeighborClasses = list(map(lambda x: trainData.iloc[x[0]][yColumnId], nearestKNeighbors))
        return mode(nearestKNeighborClasses)[0][0]
    else:
        nearestKNeighborValues = list(map(lambda x: trainData.iloc[x[0]][yColumnId], nearestKNeighbors))
        return mean(nearestKNeighborValues)


''''
plotDataSet: Plots a dataset with KNN class boundaries
@param1 int: k - Number of neighbors to use in KNN algorithm
@param2 pandas.DataFrame: trainData - Training data to plot and use for KNN Algorithm
@param3 yColumnId: str - Class column ID
@param4 list columnNames=[0,1]: Since this visualization only works in 2 dimensions, this list allows you
    to choose which features are visualized    
'''
def plotDataSet(k: int, trainData: pandas.DataFrame, yColumnId: str, columnNums: list=[0,1]):
    import colorsys
    # Inspired by # https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    trainDataDrop = trainData.drop(trainData.columns.difference(
        [trainData.columns[columnNums[0]], trainData.columns[columnNums[1]], yColumnId]), axis=1)

    h = .2  # step size in the mesh

    X1 = trainData[trainData.columns[columnNums[0]]]
    X2 = trainData[trainData.columns[columnNums[1]]]

    x1Min, x1Max = X1.min() - 1, X1.max() + 1
    x2Min, x2Max = X2.min() - 1, X2.max() + 1

    xx, yy = np.meshgrid(np.arange(x1Min, x1Max, h),
                         np.arange(x2Min, x2Max, h))

    # Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = array([])
    examples = np.c_[xx.ravel(), yy.ravel()]
    for example in examples:
        Z = np.append(Z, predict(k, trainDataDrop, example, yColumnId))

    # Quickly replace ordinal features with integer values
    uniqueList = np.unique(Z)
    mapping = {}
    c = list(trainData[yColumnId])
    for i in range(0, len(uniqueList)):
        mapping[uniqueList[i]] = i
    for i in range(0, len(Z)):
        if(not (type(Z[i]) == str)):
            Z[i] = mapping[Z[i]]
    for i in range(0, len(c)):
        c[i] = mapping[c[i]]

    # for example in examples
    Z = Z.reshape(xx.shape)

    # cmap_light = ListedColormap(['#FFAAAA','#AAAAFF', "AAFFAA", "CBC3E3", ])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF', "#00FF00", "#800080"])
    cmap_light = ListedColormap(['#FFAAAA','#AAAAFF', "#AAFFAA", "#CBC3E3" ])

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(list(X1), list(X2), c=c, cmap=cmap_bold)
    plt.xlim(xx.min() - 0.5, xx.max() + 0.5)
    plt.ylim(yy.min() - 0.5, yy.max() + 0.5)

    plt.title(f"{len(uniqueList)}-Class classification (k = {k})")

    plt.xlabel(f"{trainData.columns[columnNums[0]]}")
    plt.ylabel(f"{trainData.columns[columnNums[1]]}")

    plt.show()

