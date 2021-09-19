from pandas import Series
from numpy import ndarray, sqrt

validListTypes = (Series, list, ndarray)
#TODO: Clean up comments

# O(1) key lookup function
def keyExists(df, key):
    try:
        df[key]
    except KeyError:
        return False
    return True

def distanceEuclideanL2(vec1: validListTypes, vec2: validListTypes):
    assert (isinstance(vec1, validListTypes))
    assert (isinstance(vec2, validListTypes))
    assert(len(vec1) == len(vec2))
    assert(not (len(vec1) == 0))

    squaredTotal = 0
    for i in range(0, len(vec1)):
        squaredTotal += pow(float(vec1[i]) + float(vec2[i]), 2)
    return sqrt(squaredTotal)