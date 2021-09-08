from lib.PreprocessingTK import *
import pandas
import numpy as np

columnNames=["id", "clumpThickness", "cellSizeUniformity", "cellShapeUniformity",
              "maginalAdhesion", "epithelialCellSize", "bareNuclei", "blandChromatin",
              "normalNucleoli", "mitoses", "class"]

data = pandas.read_csv("../../data/BreastCancer/breast-cancer-wisconsin.data",
                       names=columnNames)

# Show original data frame
print("Original Data Frame")
print(data)

# Delete ID Column
data.drop("id", axis=1, inplace=True)
columnNames = list(data.columns)

# Highlight fields with missing data
print("Check for fields with missing data ['?', NaN]")
for name in columnNames:
       missingData = data.loc[data[name] == '?']
       if(not missingData.empty):
              print(missingData[[name]])

# Impute Missing Data
missingDataColumns = ["maginalAdhesion", "epithelialCellSize", "bareNuclei"]
for column in missingDataColumns:
       imputeData(data, column, nullIndicators=['?'], imputation={"method":"mean"}, inplace=True)

print("Imputed data using mean imputation method")

print("Check again for fields with missing data ['?', NaN]")
haveMissingData = False
for name in columnNames:
       missingData = data.loc[data[name] == '?']
       if(not missingData.empty):
              haveMissingData = True
              print(missingData[[name]])
if(haveMissingData):
       print("Imputation failed! Still have missing data!")

# Partition Data into 1 fold with training and testing set: ratios 0.75 and 0.25
k=5
folds = partition(data, k, classificationColumnId="class", includeValidationSet=False, proportions=(0.75, 0.25))
print(f"Partition data into {k} folds with equally sized train and test sets (no validation set.)")

for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

yCol="class"
classLabel = 2
evalRows = []

for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    prediction = naivePredictor(trainingSet, testingSet, classificationColId=yCol, method="classification")
    predicted_scores = [prediction for x in range(0,len(testingSet))] # Using first mode only

    accuracy = evaluateError(predicted_scores, testingSet["class"], method="accuracy", classLabel=classLabel)
    # precision = evaluateError(predicted_scores, testingSet["class"], method="precision", classLabel=classLabel)
    # recall = evaluateError(predicted_scores, testingSet["class"], method="recall", classLabel=classLabel)
    # f1 = evaluateError(predicted_scores, testingSet["class"], method="f1", classLabel=classLabel)

    evalRow = {
        f'accuracy-{"benign" if classLabel == 2 else "malignant" }': accuracy
    }
    evalRows.append(evalRow)

evalDf = pandas.DataFrame(evalRows)
evalDf = evalDf.rename_axis(index=None, columns='Fold')

print(evalDf.round(2))

