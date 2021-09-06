from lib.PreprocessingTK import *
import pandas
import numpy as np

columnNames=["id", "clumpThickness", "cellSizeUniformity", "cellShapeUniformity",
              "maginalAdhesion", "epithelialCellSize", "bareNuclei", "blandChromatin",
              "normalNucleoli", "mitoses", "class"]
data = pandas.read_csv("../data/BreastCancer/breast-cancer-wisconsin.data",
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

for name in columnNames:
       missingData = data.loc[data[name] == '?']

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
k=1
fold = partition(data, k, classificationColumnId="class", includeValidationSet=False, proportions=(0.75, 0.25))
print(f"Partition data into {k} folds with equally sized train and test sets (no validation set.)")
print(f"Fold 1, testSize={len(fold[0])}, trainSize={len(fold[1])}")

yCol="class"

evalRows = []
trainingSet = fold[0]
testingSet = fold[1]
prediction = naivePredictor(trainingSet, testingSet, predictorColId=yCol, method="classification")
predicted_scores = [prediction for x in range(0,len(testingSet))]
mse = evaluateError(predicted_scores, testingSet["ERP"], method="MSE")
mae = evaluateError(predicted_scores, testingSet["ERP"], method="MAE")
r2 = evaluateError(predicted_scores, testingSet["ERP"], method="R2")
pearson = evaluateError(predicted_scores, testingSet["ERP"], method="pearson")

evalRow = {
 'MSE' : mse,
 'MAE' : mae,
 'R2': r2,
 'Pearson': pearson
}
evalRows.append(evalRow)

evalDf = pandas.DataFrame(evalRows)

print(evalDf.round(1))

