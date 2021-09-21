# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

'''
preprocessAbalone: All preprocessing for the Abalone dataset condensed into one function
This process..
1. Replaces nominal values with one-hot coded values
2. Separates data into N folds
@param : numFolds=5 - The number of folds to return
@returns: List of Pandas DataFrames
'''
def preprocessAbalone(numFolds=5):
    # Read Data with Features
    '''
        Name		Data Type	Meas.	Description
        ----		---------	-----	-----------
        Sex		nominal			M, F, and I (infant)
        Length		continuous	mm	Longest shell measurement
        Diameter	continuous	mm	perpendicular to length
        Height		continuous	mm	with meat in shell
        Whole weight	continuous	grams	whole abalone
        Shucked weight	continuous	grams	weight of meat
        Viscera weight	continuous	grams	gut weight (after bleeding)
        Shell weight	continuous	grams	after being dried
        Rings		integer			+1.5 gives the age in years
    '''

    featureNames = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings"
    ]
    data = pandas.read_csv("../../data/Abalone/abalone.data",
                           names=featureNames)

    # Convert nominal data to categorical using one-hot encoding
    nominalFeatures = ["Sex"]
    for nominalFeature in nominalFeatures:
        uniqueValues = np.unique(data[nominalFeature])
        convertNominal(data, nominalFeature, uniqueValues, inplace=True)

    # Partition data into folds
    numFolds = 5
    folds = partition(data, numFolds, classificationColumnId=None)

    return folds

if __name__ == "__main__":

    numFolds = 5
    k=3
    folds = preprocessAbalone(numFolds)
    # Test our learner
    className = "Rings"
    foldEvaluations = []
    for i in range(0,k):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)
        # Make a prediction
        prediction = naivePredictor(trainingSet, className, method="regression")
        predicted_scores = [prediction for x in range(0,len(testingSet))]
        # Compose a performance evaluation, based on multiple metrics
        mse = evaluateError(predicted_scores, testingSet[className], method="MSE")
        # mae = evaluateError(predicted_scores, testingSet[className], method="MAE")
        # r2 = evaluateError(predicted_scores, testingSet[className], method="R2")
        # pearson = evaluateError(predicted_scores, testingSet[className], method="pearson")
        foldEvaluation = {
            'MSE' : mse
            # 'MAE' : mae,
            # 'R2': r2,
            # 'Pearson': pearson
        }
        foldEvaluations.append(foldEvaluation)

    print("\nLearning Performance Evaluation")
    evalDf = pandas.DataFrame(foldEvaluations)
    # evalDf.index.name = 'Fold'
    evalDf = evalDf.rename_axis(index=None, columns='Fold')
    print(evalDf.round(2))