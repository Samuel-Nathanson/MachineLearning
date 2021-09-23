# Import Libraries
from lib.PreprocessingTK import *
from examples.Cars.Cars import *
import lib.KNN as kNN
import pandas
import numpy as np

if __name__ == "__main__":
    folds = preprocessCars()
    numFolds = 5
    k=3
    # Comment: Could be improved to O(1) by assigning values directly, but this is more general
    # e.g. classLabels = [y0, y1, y2, e.t.c.]
    # Test our learner
    className = "Evaluation"
    classLabels = np.unique(folds[0][className])
    foldEvaluations = []
    for i in range(0,numFolds):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)

        foldEvaluation = {}
        for classLabel in [classLabels[0]]:
            predicted_scores = [print(x) or kNN.predict(k, trainingSet, testingSet.drop(columns=className).iloc[x], className) for x in range(0, len(testingSet))]  # Using first mode only

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
