# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

'''
preprocessMachine: Preprocesses the machine dataset and returns N folds
This process..
1. Replaces nominal values with one-hot coded values
3. Partitions data into N folds
@param numFolds : The number of folds to produce
@return List of pandas DataFrames
'''
def preprocessMachine(numFolds=5):
    # Read Data with Features
    '''
       1. vendor name: 30
          (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec,
           dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson,
           microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry,
           sratus, wang)
       2. Model Name: many unique symbols
       3. MYCT: machine cycle time in nanoseconds (integer)
       4. MMIN: minimum main memory in kilobytes (integer)
       5. MMAX: maximum main memory in kilobytes (integer)
       6. CACH: cache memory in kilobytes (integer)
       7. CHMIN: minimum channels in units (integer)
       8. CHMAX: maximum channels in units (integer)
       9. PRP: published relative performance (integer)
      10. ERP: estimated relative performance from the original article (integer)
      '''

    featureNames = [
        "VendorName",
        "ModelName",
        "MYCT",
        "MMIN",
        "MMAX",
        "CACH",
        "CHMIN",
        "CHMAX",
        "PRP",
        "ERP"
    ]

    data = pandas.read_csv("../../data/Machine/machine.data",
                           names=featureNames)

    dropColumns = ["ModelName", "VendorName"]

    for column in dropColumns:
        dropColumn(data, column, inplace=True)

    # Convert nominal data to categorical using one-hot encoding
    # nominalFeatures = ["VendorName"]
    # for nominalFeature in nominalFeatures:
    #     uniqueValues = np.unique(data[nominalFeature])
    #     convertNominal(data, nominalFeature, uniqueValues, inplace=True)

    # Partition data into folds
    folds = partition(data, numFolds, classificationColumnId=None)

    return folds

if __name__ == "__main__":
    # Test our learner
    className = "ERP"
    foldEvaluations = []
    numFolds = 5

    folds = preprocessMachine(numFolds)

    for i in range(0,numFolds):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)
        prediction = naivePredictor(trainingSet, className, method="regression")
        predicted_scores = [prediction for x in range(0,len(testingSet))]
        mse = evaluateError(predicted_scores, testingSet[className], method="MSE")
        mae = evaluateError(predicted_scores, testingSet[className], method="MAE")
        r2 = evaluateError(predicted_scores, testingSet[className], method="R2")
        pearson = evaluateError(predicted_scores, testingSet[className], method="pearson")
        foldEvaluation = {
            'MSE' : mse,
            'MAE' : mae,
            'R2': r2,
            'Pearson': pearson
        }
        foldEvaluations.append(foldEvaluation)

    print("\nLearning Performance Evaluation")
    evalDf = pandas.DataFrame(foldEvaluations)
    # evalDf.index.name = 'Fold'
    evalDf = evalDf.rename_axis(index=None, columns='Fold')
    print(evalDf.round(2))

