# CVT to Jupyter
import numpy as np
import pandas
import time
import sys

from lib.KNN import *
from lib.PreprocessingTK import *

# Classification
from examples.BreastCancer.BreastCancer import preprocessBreastCancer
from examples.Cars.Cars import preprocessCars
from examples.VotingRecords.VotingRecords import preprocessVotingRecords
# Regression
from examples.ForestFires.ForestFires import preprocessForestFires
from examples.Abalone.Abalone import preprocessAbalone
from examples.Machine.Machine import preprocessMachine

'''
runKNNExperiment - Runs a KNN Experiment to produce accuracy and timing information
@param1 List of pandas.DataFrame : folds - Folds of data to use for k-folds cross validation. Obtain using partition() function.
@param2 str: className - Y Class Label to predict
@param3 bool: doRegression=True - Set to true for regression tasks, false for classification tasks
@param4 any: classPredictionValue - This is the class you're attempting to predict accuracy for.
@return none 
'''
def runKNNExperiment(folds, yColumnId, doRegression=True, classPredictionValue=None, doCondensedAlgorithm=False, doEditedAlgorithm=True):
    doClassification = not doRegression
    # Remove one of the folds for validation.
    validationSet = folds.pop(0)

    # Parameter Values to tune
    k=None
    epsilon = 0.1

    # Tune K Value
    print("Choosing best K value")
    t0 = time.time()
    k = chooseBestK(validationSet, className=yColumnId, doClassification=doClassification, classPredictionValue=classPredictionValue, maxK=7) if k==None else k
    print(f"TIME: Choose Best K: {time.time() - t0}")
    print(f"Chose K Value of {k}")

    # For KNN-E and KNN-C, we choose an Epsilon value and then perform the algorithm
    if(doCondensedAlgorithm or doEditedAlgorithm):
        print("Choosing best Epsilon Value")
        t0 = time.time()
        if(doRegression):
            epsilon, mse, condensed = chooseBestEpsilon(validationSet, pandas.concat(folds, ignore_index=True), yColumnId, condensedAlgorithm=doCondensedAlgorithm)
            print(f"Chose Epsilon Value of {epsilon} in {time.time() - t0} seconds")
        if (doCondensedAlgorithm):
            t0 = time.time()
            print("Condensing training set...")
            for i in range(0, len(folds)):
                folds[i] = kNNCondenseTrainingSet(folds[i], yColumnId, doClassification, epsilon)
                sys.stdout.write(f'fold={i}/{numFolds - 1}, elapsed={(time.time() - t0)}')

        if (doEditedAlgorithm):
            print("Editing training set...")
            t0 = time.time()
            editedData = kNNEditTrainingSet(pandas.concat(folds, ignore_index=True), validationSet, yColumnId, classPredictionValue, k=k, epsilon=epsilon, doClassification=doClassification)
            print(f"Edited Training Set in {time.time() - t0} seconds")
            folds = partition(editedData, numFolds - 1)

    foldAccuracies = []
    print(f"Assessing accuracy of kNN, k={k}...")
    t0 = time.time()
    for i in range(0, len(folds)):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)

        predicted_scores = []
        for x in range(0, len(testingSet)):
            prediction = predict(k, trainingSet, testingSet.iloc[x].drop(labels=yColumnId), yColumnId, doClassification)
            sys.stdout.write(
                f'\rk={k}, fold={i}/{numFolds - 1}, query={x}/{len(testingSet) - 1}, elapsed={(time.time() - t0)}')
            sys.stdout.flush()
            predicted_scores.append(prediction)

        method = "MSE" if doRegression else "accuracy"
        foldScore = evaluateError(predicted_scores, testingSet[yColumnId], method=method,
                                     classLabel=classPredictionValue)
        foldAccuracies.append(foldScore)
        print(f" | {method}={foldScore}")
    meanfoldScore = np.mean(foldAccuracies)
    print(f"\nMean Score: {method}={meanfoldScore}")
    print("====****====****====****====****====****====****====")

    return k

if __name__ == "__main__":
    numFolds = 5
    k=3

    # Classification
    doBreastCancer = False
    doCarEvaluations = True
    do1984VotingRecords = True

    # Regression
    doForestFires = True
    doMachine = True
    doAbalone = True

    # Other Algorithms
    doCondensedAlgorithm = False
    doEditedAlgorithm = False

    # Options
    doDecisionPlots = False

    if(doBreastCancer):
        folds = preprocessBreastCancer(numFolds)
        className = "class"
        classPredictionValue = 2
        experimentName = "Breast Cancer Diagnosis"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        runKNNExperiment(folds, className, False, classPredictionValue=classPredictionValue, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)
        if(doDecisionPlots):
            plotDataSet(k, pandas.concat(folds, ignore_index=True), className, [0, 1])

    if (doCarEvaluations):
        folds = preprocessCars(numFolds)
        className = "Evaluation"
        classPredictionValue = "vgood"
        experimentName = "Car Evaluations"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        runKNNExperiment(folds, className, False, classPredictionValue=classPredictionValue, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)
        if(doDecisionPlots):
            plotDataSet(k, pandas.concat(folds, ignore_index=True), className, [0, 1])

    if (do1984VotingRecords):
        folds = preprocessVotingRecords(numFolds)
        className = "affiliation"
        classPredictionValue = "democrat"
        experimentName = "1984 Congressional Voting Records"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        runKNNExperiment(folds, className, False, classPredictionValue=classPredictionValue, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)
        if(doDecisionPlots):
            plotDataSet(k, pandas.concat(folds, ignore_index=True), className, [0, 1])

    if(doForestFires):
        folds = preprocessForestFires(numFolds)
        className = "Area"
        experimentName = "Forest Fires"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        runKNNExperiment(folds, className, True, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)

    if(doAbalone):
        folds = preprocessAbalone(numFolds)
        className = "Rings"
        experimentName = "Abalone"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        runKNNExperiment(folds, className, True, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)

    if(doMachine):
        folds = preprocessMachine(numFolds)
        className = "ERP"
        experimentName = "Machine Performance"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        runKNNExperiment(folds, className, True, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)


