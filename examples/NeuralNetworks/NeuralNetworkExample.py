import numpy as np
import pandas

# Classification
from examples.BreastCancer.BreastCancer import preprocessBreastCancer
from examples.Cars.Cars import preprocessCars
from examples.VotingRecords.VotingRecords import preprocessVotingRecords
# Regression
from examples.ForestFires.ForestFires import preprocessForestFires
from examples.Abalone.Abalone import preprocessAbalone
from examples.Machine.Machine import preprocessMachine

from lib.AutoencoderNeuralNetwork import AutoencoderNeuralNetwork
from lib.NeuralNetwork import NeuralNetwork
from lib.SimpleLinearNetwork import SimpleLinearNetwork
from lib.LogisticClassifier import LogisticClassifier

if __name__ == "__main__":

    numFolds = 5

    # Classification
    doBreastCancer = True
    doCarEvaluations = False
    do1984VotingRecords = False

    # Regression
    doForestFires = False
    doMachine = False
    doAbalone = False

    # Additional Options
    doLinearPrediction = True
    doNeuralNetwork = False
    doAutoencoderNetwork = False

    if([doLinearPrediction, doNeuralNetwork, doAutoencoderNetwork].count(True) != 1):
        print("Please select a single algorithm to run: Linear Prediction, Neural Network, or Autoencoder Network")
        exit(299)

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
    Iterate through each experiment to either skip or run
    '''
    for experiment in experiments.values():
        print("====****====****====****====****====****====****====")
        if(not experiment["runExperiment"]):
            # Skip experiment
            print(f"Skipping experiment: {experiment['experimentName']}")
            continue
        else:
            # Run experiment
            print(f"Running experiment: {experiment['experimentName']}")
            # Preprocess Data into folds
            folds = experiment["preprocessFunc"](numFolds)
            print(f"Separated data into {len(folds)} folds, sizes {[x.shape for x in folds]}")

        # Classification
        if(experiment["task"] == "classification"):
            foldScores = []
            for i in range(0, len(folds)-1):
                # Pop Tuning set from the folds
                tuningSet = folds.pop(i)  # Pruning Set, unused for experiments without pruning
                testingSet = folds.pop(i)  # Testing Set
                trainingSet = pandas.concat(folds, ignore_index=True)
                folds.insert(i, testingSet)
                folds.insert(i, tuningSet)

                xargs = {}

                clf = None
                name = ""
                score_name = "cross-entropy"
                if(doLinearPrediction):
                    clf = LogisticClassifier()
                    name = "Logistic Regression"
                elif(doNeuralNetwork):
                    clf = NeuralNetwork()
                    name = "Neural Network"
                elif(doAutoencoderNetwork):
                    clf = AutoencoderNeuralNetwork()
                    name = "Autoencoder Network"
                else:
                    exit(285)

                print(f"Training {name}")
                clf.train(trainData=trainingSet, yCol=experiment["yCol"])
                foldScore = clf.score(testingSet)
                print(f"Fold {i-1} : {score_name} on testing set = {foldScore}")

                foldScores.append(foldScore)

            meanfoldScore = np.mean(foldScores)
            print(f"Fold Mean Score: {meanfoldScore}")

        # Regression
        elif(experiment["task"] == "regression"):
            foldMSEs = []
            for i in range(0, len(folds)-1):
                # Pop Tuning set from the folds
                tuningSet = folds.pop(i)  # Pruning Set, unused for experiments without pruning
                testingSet = folds.pop(i)  # Testing Set
                trainingSet = pandas.concat(folds, ignore_index=True)
                folds.insert(i, testingSet)
                folds.insert(i, tuningSet)

                xargs = {}

                clf = None
                name = ""
                score_name = "MSE"
                if(doLinearPrediction):
                    clf = SimpleLinearNetwork()
                    name = "Simple Linear Network"
                elif(doNeuralNetwork):
                    clf = NeuralNetwork()
                    name = "Neural Network"
                elif(doAutoencoderNetwork):
                    clf = AutoencoderNeuralNetwork()
                    name = "Autoencoder Network"
                else:
                    exit(285)

                print(f"Training {name}")
                clf.train(trainData=trainingSet, yCol=experiment["yCol"])
                foldScore = clf.score(testingSet)
                print(f"Fold {i-1} : {score_name} on testing set = {foldScore}")

                foldScores.append(foldScore)

            meanfoldScore = np.mean(foldScores)
            print(f"Fold Mean Score: {meanfoldScore}")



        print(f"Successfully ran {experiment['experimentName']}")
        print("====****====****====****====****====****====****====\n\n")
