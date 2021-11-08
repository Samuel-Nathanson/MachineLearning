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
    doLinearPrediction = False
    doNeuralNetwork = False
    doAutoencoderNetwork = True

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
            "categoricalValues": [],
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

            # Standardize!
            from pandas.api.types import is_numeric_dtype
            from lib.PreprocessingTK import standardize
            for column_name in folds[0].drop(columns=experiment['yCol']).columns:
                if(is_numeric_dtype(folds[0][column_name])):
                    for i in range(1, len(folds)):
                        tmp, folds[i] = standardize(folds[0], folds[i], column_name, inplace=False)
                    folds[0], tmp = standardize(folds[0], folds[0], column_name, inplace=False)

            print(f"Separated data into {len(folds)} folds, sizes {[x.shape for x in folds]}")

        # Classification
        if(experiment["task"] == "classification"):
            foldScores = []
            for i in range(0, len(folds)-1):
                # Pop Tuning set from the folds
                tuningSet = folds.pop(i)  # Tuning Set
                testingSet = folds.pop(i)  # Testing Set
                trainingSet = pandas.concat(folds, ignore_index=True)
                folds.insert(i, testingSet)
                folds.insert(i, tuningSet)

                xargs = {}

                clf = None
                name = ""
                xargs = {}
                score_name = "cross-entropy"
                if(doLinearPrediction):
                    clf = LogisticClassifier()
                    xargs = {
                        "learning_rate": 0.01,
                        "stochastic_gradient_descent": False,  # Stochastic G.D. does not appear to converge.
                        "convergence_threshold": 1
                    }
                    name = "Logistic Regression"
                elif (doNeuralNetwork):
                    clf = NeuralNetwork()
                    name = "Neural Network"
                    xargs = {
                        "learning_rate": 0.01,
                        "minibatch_learning": True,
                        "convergence_threshold": 0.01,
                        "hidden_layer_dims": [8, 8],
                    }
                elif(doAutoencoderNetwork):
                    clf = NeuralNetwork()
                    name = "Autoencoder Neural Network"
                    xargs = {
                        "learning_rate": 0.001,
                        "minibatch_learning": True,
                        "convergence_threshold": 0.01,
                        "hidden_layer_dims": [len(trainingSet.columns ) - 3],
                    }
                else:
                    exit(285)

                print(f"Training {name}")
                if(doAutoencoderNetwork):
                    clf.train_autoencoder(trainData=trainingSet, yCol=experiment["yCol"], xargs=xargs)
                else:
                    clf.train(trainData=trainingSet, yCol=experiment["yCol"], xargs=xargs)



                foldScore = clf.score(testingSet)
                print(f"Fold {i} : {score_name} on testing set = {foldScore}")

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
                    xargs = {
                        "learning_rate": 0.001,
                        "stochastic_gradient_descent": False,  # Stochastic G.D. does not appear to converge.
                        "convergence_threshold": 0.01
                    }
                elif(doNeuralNetwork):
                    clf = NeuralNetwork()
                    name = "Neural Network"
                    xargs = {
                        "learning_rate": 0.001,
                        "minibatch_learning": True,
                        "convergence_threshold": 0.01,
                    }
                elif(doAutoencoderNetwork):
                    clf = AutoencoderNeuralNetwork()
                    name = "Autoencoder Network"
                else:
                    exit(285)

                print(f"Training {name}")
                clf.train(trainData=trainingSet, yCol=experiment["yCol"], xargs=xargs)
                foldScore = clf.score(testingSet)
                print(f"Fold {i} : {score_name} on testing set = {foldScore}")

                foldMSEs.append(foldScore)

            meanfoldScore = np.mean(foldMSEs)
            print(f"Fold Mean Score: {meanfoldScore}")



        print(f"Successfully ran {experiment['experimentName']}")
        print("====****====****====****====****====****====****====\n\n")
