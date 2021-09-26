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
chooseBestEpsilon - Chooses best epsilon for training set condensation / edit algorithm
@param1: pandas.DataFrame: validationSet - Validation set used for accuracy evaluation
@param2: pandas.DataFrame: trainingSet - Training set to condense or edit
@param3: str: className - Y Column class name
@return bestEpsilon: float, bestMSE: float, bestSet: pandas.DataFrame 
'''
def chooseBestEpsilon(validationSet: pandas.DataFrame, trainingSet: pandas.DataFrame, className: str, condensedAlgorithm=True):
    bestSet = 0
    bestMSE = np.inf
    bestEpsilon = 0

    min = 1/10
    max = 3/10
    increment = 1/10

    classMax = trainingSet[className].max()
    classMin = trainingSet[className].min()
    classRange = classMax - classMin
    idx = 0
    for epsilon in np.arange(min, max, increment):
        subset=None
        if(condensedAlgorithm):
            subset = kNNCondenseTrainingSet(trainingSet, className, doClassification=False, epsilon=epsilon * classRange)
        else:
            subset = kNNEditTrainingSet(trainingSet, validationSet, className, classPredictionValue=None, k=3, doClassification=False, epsilon=epsilon * classRange)
        mse = evaluateMSE(subset, validationSet,yColumnId=className)
        if(mse < bestMSE):
            bestMSE = mse
            bestEpsilon = epsilon * classRange
            bestSet = subset
    return bestEpsilon, bestMSE, bestSet

'''
runKNNExperiment - Runs a KNN Experiment to produce accuracy and timing information
@param1 List of pandas.DataFrame : folds - Folds of data to use for k-folds cross validation. Obtain using partition() function.
@param2 str: className - Y Class Label to predict
@param3 bool: doRegression=True - Set to true for regression tasks, false for classification tasks
@param4 any: classPredictionValue - This is the class you're attempting to predict accuracy for.
@return none 
'''
def runKNNExperiment(folds, className, doRegression=True, classPredictionValue=None, doCondensedAlgorithm=False, doEditedAlgorithm=True):
    k=None
    doClassification = not doRegression
    # Choose first set for k tuning
    validationSet = folds.pop(0)

    # Parameter Tuning
    epsilon = 0.1
    k=3
    if(doClassification):
        print("Choosing best K value")
        k = chooseBestK(validationSet, classPredictionValue, className , maxK=7) if k==None else k
        print(f"Chose K Value of {k}")
    else: # If we are performing regression..
        print("Choosing best Epsilon Value")
        epsilon, mse, condensed = chooseBestEpsilon(validationSet, pandas.concat(folds, ignore_index=True), className, condensedAlgorithm=doCondensedAlgorithm)
        print(f"Chose Epsilon Value of {epsilon}")

    foldAccuracies = []

    if (doCondensedAlgorithm):
        t0 = time.time()
        print("Condensing training set...")
        for i in range(0, len(folds)):
            folds[i] = kNNCondenseTrainingSet(folds[i], className, doClassification, epsilon)
            sys.stdout.write(f'fold={i}/{numFolds - 1}, elapsed={(time.time() - t0)}')

    if (doEditedAlgorithm):
        print("Editing training set...")
        editedData = kNNEditTrainingSet(pandas.concat(folds, ignore_index=True), validationSet, className,
                                        classPredictionValue, k=k)
        folds = partition(editedData, numFolds - 1, classificationColumnId=className)

    print(f"Assessing accuracy of kNN, k={k}...")
    t0 = time.time()
    for i in range(0, len(folds)):
        testingSet = folds.pop(i)
        trainingSet = pandas.concat(folds, ignore_index=True)
        folds.insert(i, testingSet)

        predicted_scores = []
        for x in range(0, len(testingSet)):
            sys.stdout.write(
                f'\rk={k}, fold={i}/{numFolds - 1}, query={x}/{len(testingSet) - 1}, elapsed={(time.time() - t0)}')
            sys.stdout.flush()
            prediction = predict(k, trainingSet, testingSet.iloc[x].drop(labels=className), className, doClassification)
            predicted_scores.append(prediction)

        method = "MSE" if doRegression else "accuracy"
        foldAccuracy = evaluateError(predicted_scores, testingSet[className], method=method,
                                     classLabel=classPredictionValue)
        foldAccuracies.append(foldAccuracy)
        print(f" | {method}={foldAccuracy}")
    meanFoldAccuracy = np.mean(foldAccuracies)
    print(f"\nMean Accuracy{method}={meanFoldAccuracy}")
    print("====****====****====****====****====****====****====")

    return k

if __name__ == "__main__":
    numFolds = 5
    k=None

    # Classification
    doBreastCancer = False
    doCarEvaluations = False
    do1984VotingRecords = False

    # Regression
    doForestFires = False
    doMachine = True
    doAbalone = False

    # Other Algorithms
    doCondensedAlgorithm = False
    doEditedAlgorithm = True

    # Options
    doDecisionPlots = False

    if(doBreastCancer):
        folds = preprocessBreastCancer(numFolds)
        className = "class"
        classPredictionValue = 2
        experimentName = "Breast Cancer Diagnosis"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        k = runKNNExperiment(folds, className, False, classPredictionValue, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)
        if(doDecisionPlots):
            plotDataSet(k, pandas.concat(folds, ignore_index=True), className, [0, 1])

    if (doCarEvaluations):
        folds = preprocessCars(numFolds)
        className = "Evaluation"
        classPredictionValue = "vgood"
        experimentName = "Car Evaluations"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        # k = runKNNExperiment(folds, className, False, classPredictionValue)
        k=1
        if(doDecisionPlots):
            plotDataSet(k, pandas.concat(folds, ignore_index=True), className, [0, 1])

    if (do1984VotingRecords):
        folds = preprocessVotingRecords(numFolds)
        className = "affiliation"
        classPredictionValue = "democrat"
        experimentName = "1984 Congressional Voting Records"
        print("====****====****====****====****====****====****====")
        print(f"Running {experimentName} Experiment - Predict {className}")
        runKNNExperiment(folds, className, False, classPredictionValue, doCondensedAlgorithm=doCondensedAlgorithm, doEditedAlgorithm=doEditedAlgorithm)
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


