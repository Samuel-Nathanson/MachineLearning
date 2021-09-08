# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np

# Read Data with Features
'''
   1. vendor name: 30 
      (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, 
       dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
       microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, 
       sratus, wang)
   2. Model Name: many unique symbols
   3. MYCT: machine cycle time in nanoseconds (integer)
   4. MMIN: minimum main memory in kilobytes (integer)
   5. MMAX: maximum main memory in kilobytes (integer)
   6. CACH: cache memory in kilobytes (integer)
   7. CHMIN: minimum channels in units (integer)
   8. CHMAX: maximum channels in units (integer)
   9. PRP: published relative performance (integer)
  10. ERP: estimated relative performance from the original article (integer)
  '''

featureNames = [
    "VendorName",
    "ModelName",
    "MYCT",
    "MMIN",
    "MMAX",
    "CACH",
    "CHMIN",
    "CHMAX",
    "PRP",
    "ERP"
]

data = pandas.read_csv("../../data/Machine/machine.data",
                       names=featureNames)

# Show original data frame
print("\nOriginal Data Frame")
print(data)

# Convert nominal data to categorical using one-hot encoding
# Convert nominal data to categorical using one-hot encoding
nominalFeatures = ["VendorName", "ModelName"]
for nominalFeature in nominalFeatures:
    uniqueValues = np.unique(data[nominalFeature])
    convertNominal(data, nominalFeature, uniqueValues, inplace=True)
# Show updated data frame
print("Data Frame after converting nominal values to categorical using one-hot encoding")
print(data)

# For demonstration purposes: Discretize PRP column into 10 bins based on frequency
print("\nFor demonstration purposes: Discretize PRP column into 10 bins based on frequency")
discretize(data, "PRP", xargs={"dMethod": "frequency", "bins": 10}, inplace=True)
# Show updated data frame
print("Data Frame after discretizing the PRP field into 10 bins of equal frequency (Showing PRP column.)")
print(data)

# Partition data into folds
k = 5
proportions = (0.75, 0.25) # Train / Test proportions
print(f"\nPartition data into {k} folds with train, test, and (Optional) validation sets: Proportions are {str(proportions)})")
folds = partition(data, k, classificationColumnId=None, includeValidationSet=False, proportions=proportions)
for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

# Test our learner
className = "ERP"
foldEvaluations = []
for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    prediction = naivePredictor(trainingSet, testingSet, classificationColId="ERP", method="regression")
    predicted_scores = [prediction for x in range(0,len(testingSet))]
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

