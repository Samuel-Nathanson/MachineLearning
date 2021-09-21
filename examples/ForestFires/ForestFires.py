# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

'''
preprocessForestFires: Preprocesses the forest fires dataset and returns N folds
This process..
1. Replaces nominal values with one-hot coded values
3. Partitions data into N folds
@param numFolds : The number of folds to produce
@return List of pandas DataFrames
'''
def preprocessForestFires(numFolds = 5):
    # Read Data with Features
    '''
       1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
       2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
       3. month - month of the year: "jan" to "dec"
       4. day - day of the week: "mon" to "sun"
       5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
       6. DMC - DMC index from the FWI system: 1.1 to 291.3
       7. DC - DC index from the FWI system: 7.9 to 860.6
       8. ISI - ISI index from the FWI system: 0.0 to 56.10
       9. temp - temperature in Celsius degrees: 2.2 to 33.30
       10. RH - relative humidity in %: 15.0 to 100
       11. wind - wind speed in km/h: 0.40 to 9.40
       12. rain - outside rain in mm/m2 : 0.0 to 6.4
       13. area - the burned area of the forest (in ha): 0.00 to 1090.84
       (this output variable is very skewed towards 0.0, thus it may make
        sense to model with the logarithm transform).
        '''

    featureNames = [
        "X",
        "Y",
        "Month",
        "Day",
        "FFMC",
        "DMC",
        "DC",
        "ISI",
        "temp",
        "RH",
        "Wind",
        "Rain",
        "Area"
    ]

    data = pandas.read_csv("../../data/ForestFires/forestfires.data",
                           names=featureNames,
                           skiprows=[0])

    # Convert nominal data to categorical using one-hot encoding
    nominalFeatures = ["Day", "Month"]
    for nominalFeature in nominalFeatures:
        uniqueValues = np.unique(data[nominalFeature])
        convertNominal(data, nominalFeature, uniqueValues, inplace=True)

    # Partition data into folds
    folds = partition(data, numFolds, classificationColumnId=None)

    return folds

if __name__ == "__main__":
    # Test our learner
    className = "Area"
    foldEvaluations = []
    numFolds = 5
    folds = preprocessForestFires(numFolds)

    for i in range(0,numFolds):
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