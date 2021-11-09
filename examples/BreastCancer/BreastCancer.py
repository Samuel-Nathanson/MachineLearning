# Import Libraries
from lib.PreprocessingTK import *
import pandas
import os
import numpy as np

'''
preprocessBreastCancer: Preprocesses the breast cancer dataset and returns N folds
This process..
1. Drops the ID column 
2. Imputes missing data
3. Partitions data into N stratified folds
@param numFolds : The number of folds to produce
@return List of pandas DataFrames
'''
def preprocessBreastCancer(numFolds: int=5):
    assert(type(numFolds) == int)
    # Read Data with Features
    '''
       #  Attribute                     Domain
       -- -----------------------------------------
       1. Sample code number            id number
       2. Clump Thickness               1 - 10
       3. Uniformity of Cell Size       1 - 10
       4. Uniformity of Cell Shape      1 - 10
       5. Marginal Adhesion             1 - 10
       6. Single Epithelial Cell Size   1 - 10
       7. Bare Nuclei                   1 - 10
       8. Bland Chromatin               1 - 10 # DNA stain color 
       9. Normal Nucleoli               1 - 10
      10. Mitoses                       1 - 10
      11. Class:                        (2 for benign, 4 for malignant)
      '''
    featureNames=["id", "clumpThickness", "cellSizeUniformity", "cellShapeUniformity",
                  "maginalAdhesion", "epithelialCellSize", "bareNuclei", "blandChromatin",
                  "normalNucleoli", "mitoses", "class"]

    file_path = os.path.join(os.path.dirname(__file__), "../../data/BreastCancer/breast-cancer-wisconsin.data")
    data = pandas.read_csv(file_path, names=featureNames)

    # Delete ID Column
    data.drop("id", axis=1, inplace=True)

    # Impute Missing Data
    missingDataColumns = ["maginalAdhesion", "epithelialCellSize", "bareNuclei"]
    for column in missingDataColumns:
           imputeData(data, column, nullIndicators=['?'], imputation={"method":"mean"}, inplace=True)

    # Partition data into folds
    k = 5
    classColName = "class"
    folds = partition(data, k, classificationColumnId=classColName)

    return folds

# Test our learner
# Comment: Could be improved to O(1) by assigning values directly, but this is more general
# e.g. classLabels = [y0, y1, y2, e.t.c.]

if __name__ == "__main__":
    k=5
    folds = preprocessBreastCancer(k)
    classLabels = [2, 4]
    className = "class"

    foldEvaluations = []
    for i in range(0,k):
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

            # Translate Class Label
            foldEvaluation[f'accuracy-{"benign" if classLabel == 2 else "malignant"}'] = accuracy
        foldEvaluations.append(foldEvaluation)

    print("\nLearning Performance Evaluation")
    evalDf = pandas.DataFrame(foldEvaluations)
    # evalDf.index.name = 'Fold'
    evalDf = evalDf.rename_axis(index=None, columns='Fold')
    print(evalDf.round(3))


