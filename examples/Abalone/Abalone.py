# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

# Read Data with Features
'''
	Name		Data Type	Meas.	Description
	----		---------	-----	-----------
	Sex		nominal			M, F, and I (infant)
	Length		continuous	mm	Longest shell measurement
	Diameter	continuous	mm	perpendicular to length
	Height		continuous	mm	with meat in shell
	Whole weight	continuous	grams	whole abalone
	Shucked weight	continuous	grams	weight of meat
	Viscera weight	continuous	grams	gut weight (after bleeding)
	Shell weight	continuous	grams	after being dried
	Rings		integer			+1.5 gives the age in years
'''

featureNames = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings"
]
data = pandas.read_csv("../../data/Abalone/abalone.data",
                       names=featureNames)
# Show original data frame
print("\nOriginal Data Frame")
print(data)

# Convert nominal data to categorical using one-hot encoding
nominalFeatures = ["Sex"]
for nominalFeature in nominalFeatures:
    uniqueValues = np.unique(data[nominalFeature])
    convertNominal(data, nominalFeature, uniqueValues, inplace=True)
# Show updated data frame
print("\nData Frame after converting nominal values to categorical using one-hot encoding")
print(data)

# Partition data into folds
k = 5
proportions = (0.75, 0.25) # Train / Test proportions
print(f"\nPartition data into {k} folds with train, test, and (Optional) validation sets: Proportions are {str(proportions)})")
folds = partition(data, k, classificationColumnId=None, includeValidationSet=False, proportions=proportions)
for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

# Test our learner
className = "Rings"
foldEvaluations = []
for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    # Make a prediction
    prediction = naivePredictor(trainingSet, testingSet, classificationColId=className, method="regression")
    predicted_scores = [prediction for x in range(0,len(testingSet))]
    # Compose a performance evaluation, based on multiple metrics
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