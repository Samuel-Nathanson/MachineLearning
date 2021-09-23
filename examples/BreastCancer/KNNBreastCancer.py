# CVT to Jupyter

from BreastCancer import preprocessBreastCancer
import lib.KNN as kNN
from lib.PreprocessingTK import *
import pandas
import time

numFolds = 5
folds = preprocessBreastCancer(numFolds)

yColumnId = "class"

'''
Tuning Function to find the best K
'''
def chooseBestK(validationSet: pandas.DataFrame):
    numFolds = 5
    classColName = "class"
    validationFolds = partition(validationSet, numFolds, classificationColumnId=classColName)
    predicted_scores = [kNN.predict(k, trainingSet, testingSet.drop(columns=yColumnId).iloc[x], yColumnId) for x in range(0, len(testingSet))]  # Using first mode only


# prediction = predict(3, folds[0], folds[1].drop(columns=yColumnId).iloc[1], yColumnId)
# print(prediction)z

k = 3
numFolds = 5
folds = preprocessBreastCancer(numFolds)
classLabel = 2
className = "class"

foldEvaluations = []
for i in range(0, numFolds):
    testingSet = folds.pop(i)
    trainingSet = pandas.concat(folds, ignore_index=True)
    folds.insert(i, testingSet)
    foldEvaluation = {}
    t0 = time.time_ns()
    predicted_scores = [kNN.predict(k, trainingSet, testingSet.drop(columns=yColumnId).iloc[x], yColumnId) for x in range(0, len(testingSet))]  # Using first mode only
    duration = time.time_ns() - t0

    accuracy = evaluateError(predicted_scores, testingSet[className], method="accuracy", classLabel=classLabel)
    # precision = evaluateError(predicted_scores, testingSet["class"], method="precision", classLabel=classLabel)
    # recall = evaluateError(predicted_scores, testingSet["class"], method="recall", classLabel=classLabel)
    # f1 = evaluateError(predicted_scores, testingSet["class"], method="f1", classLabel=classLabel)

    # Translate Class Label
    foldEvaluation[f'accuracy-{"benign" if classLabel == 2 else "malignant"}'] = \
        {
            "accuracy": accuracy,
            "runtime": duration,
            "TestingSize" : len(testingSet),
            "TrainingSize" : len(trainingSet)
        }

    foldEvaluations.append(foldEvaluation)


for i in range(0, numFolds):
    testingSet = folds.pop(i)
    trainingSet = pandas.concat(folds, ignore_index=True)
    t0 = time.time_ns()
    folds.insert(i, testingSet)
    trainingSet = kNN.kNNCondenseTrainingSet(trainingSet, className)
    duration_condense = time.time_ns() - t0

    foldEvaluation = {}
    t0 = time.time_ns()
    predicted_scores = [kNN.predict(k, trainingSet, testingSet.drop(columns=yColumnId).iloc[x], yColumnId) for x in range(0, len(testingSet))]  # Using first mode only
    duration = time.time_ns() - t0

    accuracy = evaluateError(predicted_scores, testingSet[className], method="accuracy", classLabel=classLabel)
    # precision = evaluateError(predicted_scores, testingSet["class"], method="precision", classLabel=classLabel)
    # recall = evaluateError(predicted_scores, testingSet["class"], method="recall", classLabel=classLabel)
    # f1 = evaluateError(predicted_scores, testingSet["class"], method="f1", classLabel=classLabel)

    # Translate Class Label
    foldEvaluation[f'accuracy-{"benign" if classLabel == 2 else "malignant"}'] = \
        {
            "accuracy": accuracy,
            "runtime": duration,
            "duration_condense": duration_condense,
            "TestingSize" : len(testingSet),
            "TrainingSize" : len(trainingSet)
        }

    foldEvaluations.append(foldEvaluation)

print("\nLearning Performance Evaluation")
evalDf = pandas.DataFrame(foldEvaluations)
# evalDf.index.name = 'Fold'
evalDf = evalDf.rename_axis(index=None, columns='Fold')
print(evalDf.round(3).to_string())