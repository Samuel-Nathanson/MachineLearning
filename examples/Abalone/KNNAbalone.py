# Import Libraries
from lib.PreprocessingTK import *
import pandas
import numpy as np
from lib.KNN import *
from examples.Abalone.Abalone import preprocessAbalone

if __name__ == "__main__":

    numFolds = 5
    k=3
    folds = preprocessAbalone(numFolds)
    # Test our learner
    className = "Rings"
    foldEvaluations = []
    for i in range(0,k):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)
        # Make a prediction
        predicted_scores = [
            predict(k, trainingSet, testingSet.drop(columns=className).iloc[x],
                                                       className) for x in range(0, len(testingSet))]
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