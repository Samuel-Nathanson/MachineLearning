# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

'''
preprocessCars: Preprocesses the cars dataset and returns N folds
This process..
1. Replaces ordinal values with integer values
3. Partitions data into N stratified folds
@param numFolds : The number of folds to produce
@return List of pandas DataFrames
'''
def preprocessCars(numFolds: int=5):
    # Read Data with Features
    '''
       Buying       v-high, high, med, low
       Maint        v-high, high, med, low
       Doors        2, 3, 4, 5-more
       Persons      2, 4, more
       Lug Boot     small, med, big
       Safety       low, med, high
       Evaluation   unacc, acc, good, vgood
    '''
    featureNames=["Buying", "Maint", "Doors", "Persons", "Lug Boot", "Safety", "Evaluation"]
    data = pandas.read_csv("../../data/Cars/car.data",
                      names=featureNames)
    # Show original data frame

    # Convert ordinal data to integer
    ordinalValueDict = {
        "Buying": {
            "v-high": 1,
            "vhigh": 1,
            "high": 2,
            "med": 3,
            "low": 4
        },
        "Maint": {
            "v-high": 1,
            "vhigh": 1,
            "high": 2,
            "med": 3,
            "low": 4
        },
        "Doors": {
            "5-more": 5,
            "5more": 5
        },
        "Persons": {
            "more" : 4
        },
        "Lug Boot": {
            "small": 1,
            "med": 2,
            "big": 3
        },
        "Safety": {
            "low": 1,
            "med": 2,
            "high": 3
        }
    }
    for featureName in ordinalValueDict.keys():
        convertOrdinal(data, featureName, ordinalValueDict[featureName], inplace=True)

    # Partition data into folds
    # Stratify by Y class label
    classColName = "Evaluation"
    folds = partition(data, numFolds, classificationColumnId=classColName)

    return folds

if __name__ == "__main__":
    folds = preprocessCars()
    numFolds = 5
    # Comment: Could be improved to O(1) by assigning values directly, but this is more general
    # e.g. classLabels = [y0, y1, y2, e.t.c.]
    # Test our learner
    classColName = "Evaluation"
    classLabels = np.unique(folds[0][classColName])
    className = "Evaluation"
    foldEvaluations = []
    for i in range(0,numFolds):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)

        foldEvaluation = {}
        for classLabel in classLabels:
            prediction = naivePredictor(trainingSet, className, method="classification")

            predicted_scores = [prediction for x in range(0,len(testingSet))] # Using first mode only

            accuracy = evaluateError(predicted_scores, testingSet[className], method="accuracy", classLabel=classLabel)
            # precision = evaluateError(predicted_scores, testingSet["class"], method="precision", classLabel=classLabel)
            # recall = evaluateError(predicted_scores, testingSet["class"], method="recall", classLabel=classLabel)
            # f1 = evaluateError(predicted_scores, testingSet["class"], method="f1", classLabel=classLabel)

            foldEvaluation[f'accuracy-{classLabel}'] = accuracy
        foldEvaluations.append(foldEvaluation)

    print("\nLearning Performance Evaluation")
    evalDf = pandas.DataFrame(foldEvaluations)
    # evalDf.index.name = 'Fold'
    evalDf = evalDf.rename_axis(index=None, columns='Fold')
    print(evalDf.round(2))
