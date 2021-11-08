from pandas import Series
from numpy import ndarray, sqrt, zeros, random, exp

validListTypes = (Series, list, ndarray)

'''
keyExists: O(1) key lookup function
@param df - Dictionary or DataFrame
@param key - Key to lookup
@returns boolean, True if the key is in the data structure
'''
def keyExists(df, key):
    try:
        df[key]
    except KeyError:
        return False
    return True

'''
distanceEuclideanL2: Returns Euclidean Distance between two vectors
For one-hot coded values, returns 0/1 loss 
@param1: (Series, list, ndarray) : vec1 - First Vector
@param2: (Series, list, ndarray) : vec2 - Second Vector
@returns floating point value 
'''
def distanceEuclideanL2(vec1: validListTypes, vec2: validListTypes):
    assert (isinstance(vec1, validListTypes))
    assert (isinstance(vec2, validListTypes))
    assert(len(vec1) == len(vec2))
    assert(not (len(vec1) == 0))


    squaredTotal = 0
    for i in range(0, len(vec1)):
        if(type(vec1[i]) == str and type(vec2[i] == str)):
            if((vec1[i][0:2] == '0b') and (vec2[i][0:2] == '0b')):
                if(vec1[i] == vec2[i]):
                    squaredTotal +=0
                else:
                    squaredTotal +=1
        else:
            squaredTotal += pow(float(vec1[i]) - float(vec2[i]), 2)
    return sqrt(squaredTotal)


def zero_ish(shape: list=[]):
    '''
    Compute random np.array with weights between -0.1 and 0.1
    :param shape: list
    :return: multi-dimensional array of shape : shape
    '''
    return 0.02* (random.rand(*shape) - 0.5)


def sigmoid(x):
    '''
    Sigmoid basis function
    :param x: scalar value
    :return: sigmoid computation of scalar value
    '''
    return 1 / ( 1 + exp(-1*x))

def softmax(arr: list):
    '''
    Softmax to treat all classes uniformly
    :param arr:
    :return:
    '''
    exp_sum = sum([exp(x) for x in arr])
    return [exp(x) / exp_sum for x in arr]

def sigmoid_derivative(x):
    '''
    Sigmoid Derivattive
    Proof can be found in this answer:
    https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    :param x: value
    :return: derivative of sigmoid function at value x
    '''
    return sigmoid(x)*(1-sigmoid(x))