from lib.PreprocessingTK import *
import pandas
import numpy as np

'''   
   1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
   2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
   3. month - month of the year: "jan" to "dec" 
   4. day - day of the week: "mon" to "sun"
   5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
   6. DMC - DMC index from the FWI system: 1.1 to 291.3 
   7. DC - DC index from the FWI system: 7.9 to 860.6 
   8. ISI - ISI index from the FWI system: 0.0 to 56.10
   9. temp - temperature in Celsius degrees: 2.2 to 33.30
   10. RH - relative humidity in %: 15.0 to 100
   11. wind - wind speed in km/h: 0.40 to 9.40 
   12. rain - outside rain in mm/m2 : 0.0 to 6.4 
   13. area - the burned area of the forest (in ha): 0.00 to 1090.84 
   (this output variable is very skewed towards 0.0, thus it may make
    sense to model with the logarithm transform). 
    '''

featureNames = [
    "X",
    "Y",
    "Month",
    "Day",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "Wind",
    "Rain",
    "Area"
]

data = pandas.read_csv("../../data/ForestFires/forestfires.data",
                       names=featureNames,
                       skiprows=[0])

# Show original data frame
print("Original Data Frame")
print(data)

# Convert nominal data to categorical using one-hot encoding
nominalFeatures = ["Day", "Month"]
for nominalFeature in nominalFeatures:
    uniqueValues = np.unique(data[nominalFeature])
    convertNominal(data, nominalFeature, uniqueValues, inplace=True)

# Show updated data frame
print("Data Frame after converting nominal values to categorical using one-hot encoding")
print(data)

def computePearsonModeSkewedness(arr):
    arrMean = np.mean(arr)
    arrMedian = np.median(arr)
    arrStddev = np.std(arr)

    return ((arrMean - arrMedian) / arrStddev)

# Apply a log transformation to the area column.
beforeSkewedness = computePearsonModeSkewedness(list(data["Area"]))
print(f"Skewedness before applying log transformation: {beforeSkewedness}")

data["Area"] = data["Area"].map(lambda x: x+2) # Add one to prevent math domain errors (e.g. log2(0))
data["Area"] = data["Area"].map(lambda x: math.log(x,2))

afterSkewedness = computePearsonModeSkewedness(list(data["Area"]))
print(f"Skewedness after applying log transformation: {afterSkewedness}")

# Show updated data frame
print("Data Frame after applying a logarithm transformation")
print(data)

# Partition data into 5 folds with equally sized train and test sets (no validation set.)
folds = partition(data, 5, classificationColumnId=None, includeValidationSet=False, proportions=(0.75,0.25))
print("Partition data into 5 folds with equally sized train and test sets (no validation set.)")
for i in range(0, len(folds)):
    print(f"Fold {i}, testSize={len(folds[i][0])}, trainSize={len(folds[i][1])}")

yCol = "Area"
evalRows = []
for fold in folds:
    trainingSet = fold[0]
    testingSet = fold[1]
    prediction = naivePredictor(trainingSet, testingSet, classificationColId=yCol, method="regression")
    predicted_scores = [prediction for x in range(0,len(testingSet))]
    mse = evaluateError(predicted_scores, testingSet[yCol], method="MSE")
    mae = evaluateError(predicted_scores, testingSet[yCol], method="MAE")
    r2 = evaluateError(predicted_scores, testingSet[yCol], method="R2")
    pearson = evaluateError(predicted_scores, testingSet[yCol], method="pearson")

    evalRow = {
        'MSE' : mse,
        'MAE' : mae,
        'R2': r2,
        'Pearson': pearson
    }
    evalRows.append(evalRow)

print("Performance Evaluation")
evalDf = pandas.DataFrame(evalRows)
# evalDf.index.name = 'Fold'
evalDf = evalDf.rename_axis(index=None, columns='Fold')
print(evalDf.round(1))
