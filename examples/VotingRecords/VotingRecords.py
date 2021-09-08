from lib.PreprocessingTK import *
import pandas
import numpy as np

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

data = pandas.read_csv("../../data/VotingRecords/house-votes-84.data",
                       names=featureNames)

# Show original data frame
print("Original Data Frame")
print(data)

for feature in featureNames:
    convertNominal(data,feature,['y','n','?'], inplace=True)

k=5
folds = partition(data, k, classificationColumnId="affiliation", includeValidationSet=False)

print(f"Partition data into {k} folds with equally sized train and test sets (no validation set.)")

for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

yCol="affiliation"
classLabels = ["republican", "democrat"]
evalRows = []

for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    prediction = naivePredictor(trainingSet, testingSet, classificationColId=yCol, method="classification")
    predicted_scores = [prediction for x in range(0,len(testingSet))] # Using first mode only

    evalRow = {}
    for classLabel in classLabels:
        accuracy = evaluateError(predicted_scores, testingSet[yCol], method="accuracy", classLabel=classLabel)
        evalRow[f'accuracy-{classLabel}'] = accuracy
    evalRows.append(evalRow)

evalDf = pandas.DataFrame(evalRows)
evalDf = evalDf.rename_axis(index=None, columns='Fold')

print(evalDf.round(2))