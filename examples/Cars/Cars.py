# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

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
print("Original Data Frame")
print(data)

# Convert ordinal data to integer
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
for featureName in ordinalValueDict.keys():
    convertToOrdinal(data, featureName, ordinalValueDict[featureName], inplace=True)
# Show updated data frame
print("\nData Frame after converting ordinal values to integer")
print(data)

# Partition data into folds
# Stratify by Y class label
k = 5
proportions = (0.75, 0.25) # Train / Test proportions
classColName = "Evaluation"
print(f"\nPartition data into {k} folds with train, test, and (Optional) validation sets: Proportions are {str(proportions)})")
print(f"Stratifying by values in column: {classColName}")
folds = partition(data, k, classificationColumnId=classColName, includeValidationSet=False, proportions=proportions)
for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

# Comment: Could be improved to O(1) by assigning values directly, but this is more general
# e.g. classLabels = [y0, y1, y2, e.t.c.]
# Test our learner
classLabels = np.unique(data[classColName])
className = "Evaluation"
foldEvaluations = []
for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    foldEvaluation = {}
    for classLabel in classLabels:
        prediction = naivePredictor(trainingSet, testingSet, classificationColId=className, method="classification")
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
