'''
– Provide sample outputs from one test set on one fold for a classification tree and a regression tree.
– Show a sample classification tree without pruning and with pruning.
– Show a sample regression tree without early stopping and with early stopping.
– Demonstrate the calculation of information gain, gain ratio, and mean squared error.
– Demonstrate a decision being made to prune a subtree (pruning) and a decision being made to stop growing a subtree (early stopping).
– Demonstrate an example traversing a classification tree and a class label being assigned at the leaf.
– Demonstrate an example traversing a regression tree and a prediction being made at the leaf.
'''

import numpy as np
import pandas
import time
import sys

# Classification
from examples.BreastCancer.BreastCancer import preprocessBreastCancer
from examples.Cars.Cars import preprocessCars
from examples.VotingRecords.VotingRecords import preprocessVotingRecords
# Regression
from examples.ForestFires.ForestFires import preprocessForestFires
from examples.Abalone.Abalone import preprocessAbalone
from examples.Machine.Machine import preprocessMachine

from lib.DecisionTree import ID3ClassificationTree, CARTRegressionTree

if __name__ == "__main__":

    # Classification
    doBreastCancer = True
    doCarEvaluations = False
    do1984VotingRecords = False

    # Regression
    doForestFires = False
    doMachine = False
    doAbalone = False

    '''
    Create a dictionary to store parameters for each experiment
    '''
    # TODO: Set nominal values!
    experiments = {
        "breastCancer": {
            "yCol":"class",
            "task":"classification",
            "predictClass": 2,
            "experimentName":"Breast Cancer Diagnosis Prediction",
            "preprocessFunc":preprocessBreastCancer,
            "runExperiment": doBreastCancer,
            "categoricalValues": []
        },
        "carEvaluations": {
            "yCol": "Evaluation",
            "task": "classification",
            "predictClass": "vgood",
            "experimentName": "Car Evaluation Prediction",
            "preprocessFunc": preprocessCars,
            "runExperiment": doCarEvaluations,
            "categoricalValues": []
        },
        "votingRecords": {
            "yCol": "affiliation",
            "task": "classification",
            "predictClass": "democrat",
            "experimentName": "Political Affiliation Prediction",
            "preprocessFunc": preprocessVotingRecords,
            "runExperiment": do1984VotingRecords,
            "categoricalValues": [
                "handicapped-infants",
                "water-project-cost-sharing",
                "adoption-of-the-budget-resolution",
                "physician-fee-freeze",
                "el-salvador-aid",
                "religious-groups-in-schools",
                "anti-satellite-test-ban",
                "aid-to-nicaraguan-contras",
                "mx-missile",
                "immigration",
                "synfuels-corporation-cutback",
                "education-spending",
                "superfund-right-to-sue",
                "crime",
                "duty-free-exports",
                "export-administration-act-south-africa"
            ]
        },
        "forestFires": {
            "yCol": "Area",
            "task": "regression",
            "experimentName": "Forest Fires Burned Area Prediction",
            "preprocessFunc": preprocessForestFires,
            "runExperiment": doForestFires,
            "categoricalValues": ["Day", "Month"]
        },
        "abalone": {
            "yCol": "Rings",
            "task": "regression",
            "experimentName": "Abalone Age Prediction",
            "preprocessFunc": preprocessAbalone,
            "runExperiment": doAbalone,
            "categoricalValues": ["Sex"]
        },
        "machine": {
            "yCol": "ERP",
            "task": "regression",
            "experimentName": "Machine Performance",
            "preprocessFunc": preprocessMachine,
            "runExperiment": doMachine,
            "categoricalValues": []
        }
    }

    '''
    Set shared values for all experiments
    '''
    numFolds = 6
    doPostPruning = True
    doThresholdTuning = True

    ''' 
    Iterate through each experiment to either skip or run
    '''
    for experiment in experiments.values():
        print("====****====****====****====****====****====****====")
        if(not experiment["runExperiment"]):
            # Skip experiment
            print(f"Skipping experiment: {experiment['experimentName']}")
            continue
            print("====****====****====****====****====****====****====\n\n")
        else:
            # Run experiment
            print(f"Running experiment: {experiment['experimentName']}")
            # Preprocess Data into folds
            folds = experiment["preprocessFunc"](numFolds)
            print(f"Separated data into {len(folds)} folds, sizes {[x.shape for x in folds]}")

        if(experiment["task"] == "classification"):
            foldAccuracies = []
            for i in range(0, len(folds)-1):
                # Pop Tuning and Pruning set from the folds
                pruningSet = folds.pop(i)  # Pruning Set, unused for experiments without pruning
                testingSet = folds.pop(i)  # Testing Set
                trainingSet = pandas.concat(folds, ignore_index=True)
                folds.insert(i, testingSet)
                folds.insert(i, pruningSet)
                xargs = {
                    "PruningSet": pruningSet if doPostPruning else None,
                    "CategoricalValues": experiment["categoricalValues"]
                }
                clf = ID3ClassificationTree()
                clf.train(trainingSet=trainingSet, yCol=experiment["yCol"], xargs=xargs)

                print(f"ID3 Decision Tree constructed {'and pruned' if doPostPruning else ''}")
                testingAccuracy = clf.score(testingSet)
                print(f"Fold {i-1} : ID3 Decision Tree accuracy on testing set = {testingAccuracy}")

                foldAccuracies.append(testingAccuracy)
            meanfoldScore = np.mean(foldAccuracies)
            print(f"Fold Mean Score: {meanfoldScore}")


        elif(experiment["task"] == "regression"):
            foldMSEs = []
            for i in range(0, len(folds)-1):
                # Pop Tuning and Pruning set from the folds
                tuningSet = folds.pop(i)  # Pruning Set, unused for experiments without pruning
                testingSet = folds.pop(i)  # Testing Set
                trainingSet = pandas.concat(folds, ignore_index=True)
                folds.insert(i, testingSet)
                folds.insert(i, tuningSet)

                predicted_scores = []

                xargs = {
                    "TuningSet": tuningSet if doThresholdTuning else None,
                    "CategoricalValues": experiment["categoricalValues"]
                }
                clf = CARTRegressionTree()
                clf.train(trainingSet=trainingSet, yCol=experiment["yCol"], xargs=xargs)

                print(f"CART Decision Tree constructed {'and tuned' if doThresholdTuning else ''}")
                testingMSE = clf.score(testingSet)
                print(f"Fold {i-1} : CART Decision Tree MSE on testing set = {testingMSE}")

                foldMSEs.append(testingMSE)
            meanfoldScore = np.mean(foldMSEs)
            print(f"Fold Mean Score : {meanfoldScore}")



        print(f"Successfully ran {experiment['experimentName']}")
        print("====****====****====****====****====****====****====\n\n")
