# CVT to Jupyter

from BreastCancer import preprocessBreastCancer
import lib.KNN as kNN
from lib.PreprocessingTK import *
import pandas

numFolds = 5
folds = preprocessBreastCancer(numFolds)

yColumnId = "class"

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
    predicted_scores = [kNN.predict(k, trainingSet, testingSet.drop(columns=yColumnId).iloc[x], yColumnId) for x in range(0, len(testingSet))]  # Using first mode only

    accuracy = evaluateError(predicted_scores, testingSet[className], method="accuracy", classLabel=classLabel)
    # precision = evaluateError(predicted_scores, testingSet["class"], method="precision", classLabel=classLabel)
    # recall = evaluateError(predicted_scores, testingSet["class"], method="recall", classLabel=classLabel)
    # f1 = evaluateError(predicted_scores, testingSet["class"], method="f1", classLabel=classLabel)

    # Translate Class Label
    foldEvaluation[f'accuracy-{"benign" if classLabel == 2 else "malignant"}'] = accuracy
    foldEvaluations.append(foldEvaluation)

print("\nLearning Performance Evaluation")
evalDf = pandas.DataFrame(foldEvaluations)
# evalDf.index.name = 'Fold'
evalDf = evalDf.rename_axis(index=None, columns='Fold')
print(evalDf.round(3))