import pandas
from lib.SNUtils import keyExists, distanceEuclideanL2, validListTypes
from scipy.stats import mode
from numpy import mean

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
        print("Size of Z=" + str(newSizeZ))
        indicesToDrop = []
        for i in range(1, XTrainData.shape[0]):
            query = pandas.DataFrame(columns=XTrainData.columns).append(XTrainData.iloc[i])
            nearestNeighborList = findKNearestNeighbors(1, minSetZ, query, yColumnId)
            # findNearestNeighbors returns a list of tuples (neighborIndex, distance)
            nearestNeighborIdx = nearestNeighborList[0][0]
            # Create DataFrame with nearest neighbor
            nearestNeighbor = pandas.DataFrame(columns=XTrainData.columns).append(minSetZ.iloc[nearestNeighborIdx])
            nearestNeighborClass = nearestNeighbor.iloc[0][yColumnId]
            queryClass = query.iloc[0][yColumnId]

            if(not (nearestNeighborClass == queryClass)):
                minSetZ = minSetZ.append(query)
                indicesToDrop.append(i)
        prevSizeZ = newSizeZ
        newSizeZ = minSetZ.shape[0]
        XTrainData.drop(XTrainData.index[indicesToDrop], inplace=True)

    return minSetZ

def predict(k: int, trainData: pandas.DataFrame, query : validListTypes, yColumnId: str, classify:bool=True):
    assert(type(k) == int)
    assert(type(trainData) == pandas.DataFrame)
    assert(isinstance(query, validListTypes))
    assert(type(yColumnId) == str)
    assert(type(classify) == bool)
    assert(keyExists(trainData, yColumnId))
    assert(keyExists(query, yColumnId))

    nearestKNeighbors = findKNearestNeighbors(k, trainData, query, yColumnId)

    if(classify):
        nearestKNeighborClasses = list(map(lambda x: trainData.iloc[x[0]][yColumnId], nearestKNeighbors))
        return mode(nearestKNeighborClasses)[0][0]
    else:
        nearestKNeighborValues = list(map(lambda x: trainData.iloc[x[0]][yColumnId], nearestKNeighbors))
        return mean(nearestKNeighborValues)


def plotDataSet(k: int, trainData: pandas.DataFrame, yColumnId: str, columnNums: list=[0,1]):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    # from sklearn import neighbors, datasets

    # n_neighbors = 15

    # import some data to play with
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]  # we only take the first two features. We could
    # # avoid this ugly slicing by using a two-dim dataset
    # y = iris.target

    h = .02  # step size in the mesh

    X1 = trainData[trainData.columns[columnNums[0]]]
    X2 = trainData[trainData.columns[columnNums[1]]]

    x1Min, x1Max = X1.min() - 1, X1.max() + 1
    x2Min, x2Max = X2.min() - 1, X2.max() + 1

    xx, yy = np.meshgrid(np.arange(x1Min, x1Max, h),
                         np.arange(x2Min, x2Max, h))

    # Z = predict(np.c_[xx.ravel(), yy.ravel()])
    examples = np.c_[xx.ravel(), yy.ravel()]

    # for example in examples
    # Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(list(X1), list(X2), list(trainData[trainData.columns[yColumnId]]), cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title("3-Class classification (k = %i)")

    plt.show()


# https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib