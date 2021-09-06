from lib.PreprocessingTK import *
import pandas
import numpy as np
data = pandas.read_csv("../data/Machine/machine.data",
                  names=["VendorName", "ModelName", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])

# Show original data frame
print("Original Data Frame")
print(data)

# Convert nominal data to categorical using one-hot encoding
vendorNames = np.unique(data["VendorName"])
modelNames = np.unique(data["ModelName"])
convertNominal(data, "VendorName", vendorNames, inplace=True)
convertNominal(data, "ModelName", modelNames, inplace=True)

# Show updated data frame
print("Data Frame after converting nominal values to categorical using one-hot encoding")
print(data)

# discretize(data, "PRP", xargs={"dMethod": "equal-width", "bins": 10}, inplace=True)
# zeroIndex = np.min(data["PRP"])
# oneIndex = np.max(data["PRP"])
# print("Equal-Width Discretized - PRP Bin 0: " + str(np.count_nonzero(data["PRP"] == zeroIndex)))
# print("Equal-Width Discretized - PRP Bin 1: " + str(np.count_nonzero(data["PRP"] == oneIndex)))

# Discretize PRP column into 10 bins based on frequency
discretize(data, "PRP", xargs={"dMethod": "frequency", "bins": 10}, inplace=True)

# Show updated data frame
print("Data Frame after discretizing the PRP field into 10 bins of equal frequency (Showing PRP column.)")
print(data)

# Partition data into 5 folds with equally sized train and test sets (no validation set.)
folds = partition(data, 5, classificationColumnId=None, includeValidationSet=False, proportions=(0.5,0.5))
print("Partition data into 5 folds with equally sized train and test sets (no validation set.)")
for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

# Demonstration: Partition data into 10 folds with train, test, and validation sets with ratio (0.75, 0.15, 0.1)
# folds = partition(data, 10, classificationColumnId=None, includeValidationSet=True, proportions=(0.75,0.15, 0.1))
# print("Demonstration: Partition data into 10 folds with train, test, and validation sets with ratio (0.75, 0.15, 0.1)")
# for i in range(0, len(folds)):
#     print(f"Fold {i}, trainSize={len(folds[i][0])}, testSize={len(folds[i][1])}, validationSize={len(folds[i][2])}")


yCol = "ERP"

evalRows = []
for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    prediction = naivePredictor(trainingSet, testingSet, predictorColId="ERP", method="regression")
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


