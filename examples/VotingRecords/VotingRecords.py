# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

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
data = pandas.read_csv("../../data/VotingRecords/house-votes-84.data",
                       names=featureNames)
# Show original data frame
print("Original Data Frame")
print(data)

# Partition data into folds
k = 5
proportions = (0.75, 0.25) # Train / Test proportions
classColName = "affiliation"
print(f"\nPartition data into {k} folds with train, test, and (Optional) validation sets: Proportions are {str(proportions)})")
print(f"Stratifying by values in column: {classColName}")
folds = partition(data, k, classificationColumnId=classColName, includeValidationSet=False, proportions=proportions)
for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

# Comment: Could be improved to O(1) by assigning values directly, but this is more general
# e.g. classLabels = [y0, y1, y2, e.t.c.]
# Test our learner
classLabels = np.unique(data[classColName])
className = "affiliation"
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

        # Translate Class Label
        foldEvaluation[f'accuracy-{classLabel}'] = accuracy
    foldEvaluations.append(foldEvaluation)

print("\nLearning Performance Evaluation")
evalDf = pandas.DataFrame(foldEvaluations)
# evalDf.index.name = 'Fold'
evalDf = evalDf.rename_axis(index=None, columns='Fold')
print(evalDf.round(3))
