# Import Libraries
from lib.PreprocessingTK import *
from lib.KNN import *
from examples.Machine.Machine import *
import pandas
import numpy as np

if __name__ == "__main__":
    # Test our learner
    className = "ERP"
    foldEvaluations = []
    numFolds = 5
    k=3

    folds = preprocessMachine(numFolds)

    for i in range(0,numFolds):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)

        predicted_scores = [
            print(f"{x}/{testingSet.size}") or predict(k, trainingSet, testingSet.drop(columns=className).iloc[x], className, classify=False) for x
            in range(0, len(testingSet))]
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

