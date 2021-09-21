import pandas
from lib.SNUtils import keyExists, distanceEuclideanL2, validListTypes
from scipy.stats import mode
from numpy import mean, array, append

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
        p2 = trainData.drop(columns=yColumnId).iloc[i]
        distances.append([i, distance(p1, p2)])

    sortedNeighbors = sorted(distances, key=lambda disTuple: disTuple[1])

    return sortedNeighbors[0:k]

'''
kNNCondenseTrainingSet - Condenses the training set based on Hart's Algorithm
@param1 pandas.DataFrame: trainData - Training data to draw from
@param2 str: yColumnId - Class column ID
@return pandas.DataFrame corresponding to condensed dataset
'''
def kNNCondenseTrainingSet(trainData: pandas.DataFrame, yColumnId: str):
    assert (type(trainData) == pandas.DataFrame)
    assert(type(yColumnId) == str)
    assert (keyExists(trainData, yColumnId))

    # Randomize indices of training data.
    XTrainData = trainData.sample(frac=1, random_state=0).reset_index(drop=True).copy(deep=True)
    minSetZ = pandas.DataFrame(columns=XTrainData.columns).append(XTrainData.iloc[0])

    prevSizeZ = 0
    newSizeZ = 1

    while(newSizeZ != prevSizeZ):
        indicesToDrop = []
        for i in range(1, XTrainData.shape[0]):
            query = XTrainData.iloc[i]
            nearestNeighborList = findKNearestNeighbors(1, minSetZ, query.drop(labels=[yColumnId]), yColumnId)
            # findNearestNeighbors returns a list of tuples (neighborIndex, distance)
            nearestNeighborIdx = nearestNeighborList[0][0]
            # Create DataFrame with nearest neighbor
            nearestNeighbor = pandas.DataFrame(columns=XTrainData.columns).append(minSetZ.iloc[nearestNeighborIdx])
            nearestNeighborClass = nearestNeighbor.iloc[0][yColumnId]
            queryClass = query[yColumnId]
            print(f"{i}/{XTrainData.shape[0]}")

            if(not (nearestNeighborClass == queryClass)):
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
    # Inspired by # https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    # from sklearn import neighbors, datasets

    trainDataDrop = trainData.drop(trainData.columns.difference([trainData.columns[columnNums[0]], trainData.columns[columnNums[1]], yColumnId]), 1)
    # n_neighbors = 15

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

    # for example in examples
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA','#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(list(X1), list(X2), c=list(trainData[yColumnId]), cmap=cmap_bold)
    plt.xlim(xx.min() - 1, xx.max() + 1)
    plt.ylim(yy.min() - 1, yy.max() + 1)

    plt.title(f"3-Class classification (k = {k})")

    plt.show()


