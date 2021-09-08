from lib.PreprocessingTK import *
import pandas
import numpy as np

# Read Data
data = pandas.read_csv("../../data/Cars/car.data",
                  names=["Buying", "Maint", "Doors", "Persons", "Lug Boot", "Safety", "Evaluation"])

# Show original data frame
print("Original Data Frame")
print(data)

ordinalValueDict = {
    "Buying": {
        "v-high": 1,
        "high": 2,
        "med": 3,
        "low": 4
    },
    "Maint": {
        "v-high": 1,
        "high": 2,
        "med": 3,
        "low": 4
    },
    "Doors": {
        "5-more": 5
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

for colId in ordinalValueDict.keys():
    convertToOrdinal(data, colId, ordinalValueDict[colId], inplace=True)

# Show updated data frame
print("Data Frame after converting ordinal values to integer")
print(data)

# Partition data into 5 folds with equally sized train and test sets (no validation set.)
folds = partition(data, 5, classificationColumnId="Evaluation", includeValidationSet=False, proportions=(0.8,0.2))
print("Partition data into 5 folds with equally sized train and test sets (no validation set.)")
for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

classLabels = ["unacc", "acc", "good", "vgood"]
evalRows = []
yCol = "Evaluation"

for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    evalRow = {}
    for classLabel in classLabels:
        prediction = naivePredictor(trainingSet, testingSet, classificationColId=yCol, method="classification")
        predicted_scores = [prediction for x in range(0,len(testingSet))] # Using first mode only

        accuracy = evaluateError(predicted_scores, testingSet[yCol], method="accuracy", classLabel=classLabel)
    # precision = evaluateError(predicted_scores, testingSet["class"], method="precision", classLabel=classLabel)
    # recall = evaluateError(predicted_scores, testingSet["class"], method="recall", classLabel=classLabel)
    # f1 = evaluateError(predicted_scores, testingSet["class"], method="f1", classLabel=classLabel)

        evalRow[f'accuracy-{classLabel}'] = accuracy
    evalRows.append(evalRow)

evalDf = pandas.DataFrame(evalRows)
evalDf = evalDf.rename_axis(index=None, columns='Fold')

print(evalDf.round(3))

