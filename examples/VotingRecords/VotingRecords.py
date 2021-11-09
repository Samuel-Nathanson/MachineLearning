# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np
from lib.KNN import *
import os

'''
preprocessVotingRecords: Preprocesses the voting records dataset and returns N folds
This process..
1. Partitions data into N stratified folds
@param numFolds : The number of folds to produce
@return List of pandas DataFrames
'''
def preprocessVotingRecords(numFolds):
    # Read Data with Features
    '''
      1. Class Name: 2 (democrat, republican)
       2. handicapped-infants: 2 (y,n)
       3. water-project-cost-sharing: 2 (y,n)
       4. adoption-of-the-budget-resolution: 2 (y,n)
       5. physician-fee-freeze: 2 (y,n)
       6. el-salvador-aid: 2 (y,n)
       7. religious-groups-in-schools: 2 (y,n)
       8. anti-satellite-test-ban: 2 (y,n)
       9. aid-to-nicaraguan-contras: 2 (y,n)
      10. mx-missile: 2 (y,n)
      11. immigration: 2 (y,n)
      12. synfuels-corporation-cutback: 2 (y,n)
      13. education-spending: 2 (y,n)
      14. superfund-right-to-sue: 2 (y,n)
      15. crime: 2 (y,n)
      16. duty-free-exports: 2 (y,n)
      17. export-administration-act-south-africa: 2 (y,n)
      '''
    featureNames = [
        "affiliation",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa"
    ]
    file_path = os.path.join(os.path.dirname(__file__), "../../data/VotingRecords/house-votes-84.data")
    data = pandas.read_csv(file_path, names=featureNames)

    # Partition data into folds

    classColName = "affiliation"
    folds = partition(data, numFolds, classificationColumnId=classColName)

    return folds

if __name__ == "__main__":
    # Comment: Could be improved to O(1) by assigning values directly, but this is more general
    # e.g. classLabels = [y0, y1, y2, e.t.c.]
    # Test our learner
    numFolds = 5
    k=3
    className = "affiliation"
    folds = preprocessVotingRecords(numFolds)
    classLabels = np.unique(folds[0][className])
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

            # Translate Class Label
            foldEvaluation[f'accuracy-{classLabel}'] = accuracy
        foldEvaluations.append(foldEvaluation)

    print("\nLearning Performance Evaluation")
    evalDf = pandas.DataFrame(foldEvaluations)
    # evalDf.index.name = 'Fold'
    evalDf = evalDf.rename_axis(index=None, columns='Fold')
    print(evalDf.round(3))
