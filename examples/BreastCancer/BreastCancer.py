# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

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
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
  '''
featureNames=["id", "clumpThickness", "cellSizeUniformity", "cellShapeUniformity",
              "maginalAdhesion", "epithelialCellSize", "bareNuclei", "blandChromatin",
              "normalNucleoli", "mitoses", "class"]

data = pandas.read_csv("../../data/BreastCancer/breast-cancer-wisconsin.data",
                       names=featureNames)
# Show original data frame
print("Original Data Frame")
print(data)

# Delete ID Column
data.drop("id", axis=1, inplace=True)
columnNames = list(data.columns)

# Demonstration: Check for fields with missing data
print("Check for fields with missing data ['?', NaN]")
for name in columnNames:
       missingData = data.loc[data[name] == '?']
       if(not missingData.empty):
              print(missingData[[name]])

# Impute Missing Data
missingDataColumns = ["maginalAdhesion", "epithelialCellSize", "bareNuclei"]
for column in missingDataColumns:
       imputeData(data, column, nullIndicators=['?'], imputation={"method":"mean"}, inplace=True)
print("\nImputed data using mean imputation method")

# Demonstration: Check for fields with missing data
print("Check again for fields with missing data ['?', NaN]")
haveMissingData = False
for name in columnNames:
       missingData = data.loc[data[name] == '?']
       if(not missingData.empty):
              haveMissingData = True
              print(missingData[[name]])
if(haveMissingData):
    print("Imputation failed! Still have missing data.")
    exit(1)
else:
    print("Imputation succeeded! Filled missing data with mean.")

# Partition data into folds
k = 5
classColName = "class"
print(f"\nPartition data into {k} folds with train, test, and (Optional) validation sets")
print(f"Stratifying by values in column: {classColName}")
folds = partition(data, k, classificationColumnId=classColName)
for i in range(0, len(folds)):
    print(f"Fold {i}, size={len(folds[i])}")

# Test our learner
# Comment: Could be improved to O(1) by assigning values directly, but this is more general
# e.g. classLabels = [y0, y1, y2, e.t.c.]
classLabels = np.unique(data[classColName])
className = "class"
foldEvaluations = []
for i in range(0,k):
    testingSet = folds.pop(i)
    trainingSet = pandas.concat(folds, ignore_index=True)
    folds.insert(i, testingSet)
    foldEvaluation = {}
    for classLabel in classLabels:
        prediction = naivePredictor(trainingSet, testingSet, classificationColId=className, method="classification")
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


