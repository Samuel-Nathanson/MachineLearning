# Import Libraries
from lib.PreprocessingTK import *
from examples.ForestFires.ForestFires import *
import lib.KNN as kNN
import pandas
import numpy as np

if __name__ == "__main__":
    # Test our learner
    className = "Area"
    foldEvaluations = []
    numFolds = 5
    k=3
    folds = preprocessForestFires(numFolds)

    for i in range(0,numFolds):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)

        predicted_scores = [kNN.predict(k, trainingSet, testingSet.drop(columns=className).iloc[x], className) for x
                            in range(0, len(testingSet))]  # Using first mode only

        # Make a prediction
        predicted_scores = [kNN.predict(k, trainingSet, testingSet.drop(columns=className).iloc[x], className, classify=False) for x
                            in range(0, len(testingSet))]
        # Compose a performance evaluation, based on multiple metrics
        mse = evaluateError(predicted_scores, testingSet[className], method="MSE")
        # mae = evaluateError(predicted_scores, testingSet[className], method="MAE")
        # r2 = evaluateError(predicted_scores, testingSet[className], method="R2")
        # pearson = evaluateError(predicted_scores, testingSet[className], method="pearson")
        foldEvaluation = {
            'MSE' : mse
            # 'MAE' : mae,
            # 'R2': r2,
            # 'Pearson': pearson
        }
        foldEvaluations.append(foldEvaluation)

    print("\nLearning Performance Evaluation")
    evalDf = pandas.DataFrame(foldEvaluations)
    # evalDf.index.name = 'Fold'
    evalDf = evalDf.rename_axis(index=None, columns='Fold')
    print(evalDf.round(2))