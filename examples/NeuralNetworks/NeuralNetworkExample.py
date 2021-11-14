import numpy as np
import pandas
import argparse
import multiprocessing
import copy

# Classification
from examples.BreastCancer.BreastCancer import preprocessBreastCancer
from examples.Cars.Cars import preprocessCars
from examples.VotingRecords.VotingRecords import preprocessVotingRecords
# Regression
from examples.ForestFires.ForestFires import preprocessForestFires
from examples.Abalone.Abalone import preprocessAbalone
from examples.Machine.Machine import preprocessMachine

from lib.NeuralNetwork import NeuralNetwork, train_network
from lib.SimpleLinearNetwork import SimpleLinearNetwork
from lib.LogisticClassifier import LogisticClassifier

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="Dataset", type=str, required=False)
    parser.add_argument("-model", help="Model to run", type=str, required=False)
    parser.add_argument("-learning_rate", help="Learning rate", type=float, required=False)
    parser.add_argument("-convergence_threshold", help="Convergence Threshold", type=float, required=False)
    args = parser.parse_args()

    numFolds = 5

    # Classification
    doBreastCancer = args.dataset == "breast_cancer" or False
    doCarEvaluations = args.dataset == "car_evaluations" or False
    do1984VotingRecords = args.dataset == "voting_records" or False

    # Regression
    doForestFires = args.dataset == "forest_fires" or False
    doMachine = args.dataset == "machine_performance" or False
    doAbalone = args.dataset == "abalone_age" or False

    # Additional Options
    print(args.model)
    doLinearPrediction = args.model == "linear" or False
    doNeuralNetwork = args.model == "neural_network" or False
    doAutoencoderNetwork = args.model == "autoencoder" or False

    common_convergence_threshold = args.convergence_threshold or 0.1
    common_learning_rate = args.learning_rate or 0.1

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
            "tuned_hidden_dims": [9, 4]
        },
        "carEvaluations": {
            "yCol": "Evaluation",
            "task": "classification",
            "predictClass": "vgood",
            "experimentName": "Car Evaluation Prediction",
            "preprocessFunc": preprocessCars,
            "runExperiment": doCarEvaluations,
            "categoricalValues": [],
            "tuned_hidden_dims": [5,6]
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
            ],
            "tuned_hidden_dims": [46, 46]
        },
        "forestFires": {
            "yCol": "Area",
            "task": "regression",
            "experimentName": "Forest Fires Burned Area Prediction",
            "preprocessFunc": preprocessForestFires,
            "runExperiment": doForestFires,
            "categoricalValues": ["Day", "Month"],
            "tuned_hidden_dims": [15,38]
        },
        "abalone": {
            "yCol": "Rings",
            "task": "regression",
            "experimentName": "Abalone Age Prediction",
            "preprocessFunc": preprocessAbalone,
            "runExperiment": doAbalone,
            "categoricalValues": ["Sex"],
            "tuned_hidden_dims": [5, 14]
        },
        "machine": {
            "yCol": "ERP",
            "task": "regression",
            "experimentName": "Machine Performance",
            "preprocessFunc": preprocessMachine,
            "runExperiment": doMachine,
            "categoricalValues": [],
            "tuned_hidden_dims": [6,9]
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

            # Standardize and one-hot-code values!
            from pandas.api.types import is_numeric_dtype
            from lib.PreprocessingTK import standardize, convertNominal, partition
            all_folds = pandas.concat(folds)
            for column_name in all_folds.drop(columns=experiment['yCol']).columns:
                if(is_numeric_dtype(all_folds[column_name])):
                    all_folds = standardize(trainingSetDF=all_folds, testingSetDF=None, columnId=column_name, inplace=False)
                else:
                    all_folds = convertNominal(all_folds, column_name, experiment['categoricalValues'], inplace=False)
            folds = partition(all_folds, numFolds, classificationColumnId=experiment['yCol'])

            print(f"Separated data into {len(folds)} folds, sizes {[x.shape for x in folds]}")

        # Classification
        if experiment["task"] == "classification":
            foldScores = []
            for i in range(0, len(folds)-1):

                # Pop Tuning set from the folds

                tuningSet = folds.pop(i)  # Tuning Set
                testingSet = folds.pop(i)  # Testing Set
                trainingSet = pandas.concat(folds, ignore_index=True)
                folds.insert(i, testingSet)
                folds.insert(i, tuningSet)

                # Declare common variables

                xargs = {}
                clf = None
                name = ""
                xargs = {}
                score_name = "cross-entropy"

                # Set experiment-specific variables

                # Linear Regression

                if(doLinearPrediction):
                    clf = LogisticClassifier()
                    name = "Logistic Regression"
                    xargs = {
                        "learning_rate": 0.01,  # TODO: Tune
                        "convergence_threshold": common_convergence_threshold,
                    }

                # Neural Networks (Two Hidden Layers)

                elif (doNeuralNetwork):
                    clf = NeuralNetwork()
                    name = "Neural Network"
                    xargs = {
                        "learning_rate": 0.1,
                        "minibatch_learning": True, # Not implemented yet
                        "convergence_threshold": common_convergence_threshold,
                        "hidden_layer_dims": [8, 8], # TODO: Tune
                        "task": "classification"
                    }

                # Autoencoder Neural Networks (Two Hidden Layers)
                elif(doAutoencoderNetwork):
                    clf = NeuralNetwork()
                    name = "Autoencoder Neural Network"
                    xargs = {
                        "learning_rate": 0.1, # Increasing to 0.01 seems to cause divergence
                        "minibatch_learning": True, # Not implemented yet
                        "convergence_threshold": common_convergence_threshold,
                        "hidden_layer_dims": experiment["tuned_hidden_dims"], # TODO: Tune
                        "task": "classification"
                    }
                else:
                    exit(285)

                # Train the model


                print(f"Training {name}")

                if(doAutoencoderNetwork):
                    clf.train_autoencoder(trainData=trainingSet, yCol=experiment["yCol"], xargs=xargs)
                if(doNeuralNetwork):
                    import time
                    t0 = time.time()
                    # Tune hidden layer sizes from 0.5 - 1.5 times the number of inputs
                    num_inputs = len(trainingSet.columns) - 1

                    ten_percent_less_units = int(np.floor(num_inputs -2))
                    ten_percent_more_units = int(np.ceil(num_inputs +1))
                    layer1_range = range(ten_percent_less_units, ten_percent_more_units)
                    layer2_range = range(ten_percent_less_units, ten_percent_more_units)

                    # fifty_percent_less_units = int(np.floor(num_inputs / 2))
                    # fifty_percent_more_units = int(num_inputs + fifty_percent_less_units)
                    #
                    # layer1_range = range(fifty_percent_less_units, fifty_percent_more_units)
                    # layer2_range = range(fifty_percent_less_units, fifty_percent_more_units)

                    combinations = []
                    networks = []
                    for j in layer1_range:
                        for k in layer2_range:
                            networks.append(NeuralNetwork())
                            combinations.append([j, k])

                    '''
                    Multi-threading
                    '''
                    num_procs = len(combinations)
                    jobs = []
                    queue = None
                    lock = multiprocessing.Lock()
                    manager = multiprocessing.Manager()
                    scores = manager.dict()

                    for proc_num in range(0, num_procs):
                        combination = combinations[proc_num]
                        tuning_xargs = copy.deepcopy(xargs)
                        tuning_xargs["hidden_layer_dims"] = combination
                        process = multiprocessing.Process(target=train_network,
                                                          args=(trainingSet,
                                                                experiment["yCol"],
                                                                tuning_xargs,
                                                                lock,
                                                                scores,
                                                                tuningSet,
                                                                proc_num))

                        jobs.append(process)

                    # Time to train 81 networks: 111.82631874084473
                    # Start the threads (i.e. calculate the random number lists)
                    for proc_num, j in enumerate(jobs):
                        print(f"Starting thread {proc_num}")
                        j.start()

                    # Ensure all of the threads have finished
                    for j in jobs:
                        j.join()

                    t1 = time.time()

                    print(f"Time to train {num_procs} neural networks: {t1-t0}")

                    bestScore = np.max([x for x in scores.keys()])
                    bestDims = scores[bestScore] # Get dims of best network
                    print(f"best dims={bestDims}")
                    tuning_xargs["hidden_layer_dims"] = bestDims
                    clf = NeuralNetwork()
                    clf.train(trainData=trainingSet, yCol=experiment["yCol"], xargs=tuning_xargs)
                else:
                    clf.train(trainData=trainingSet, yCol=experiment["yCol"], xargs=xargs)

                # Find cross-entropy at each fold

                foldScore = clf.score(testingSet)
                print(f"Fold {i} : {score_name} on testing set = {foldScore}")
                foldScores.append(foldScore)

            # Print average cross-entropy score across all folds
            meanfoldScore = np.mean(foldScores)
            print(f"Fold Mean Score: {meanfoldScore}")

        # Experiment on Regression Data Sets

        elif (experiment["task"] == "regression"):

            foldScores = []
            for i in range(0, len(folds)-1):

                # Pop Tuning set from the folds

                tuningSet = folds.pop(i)  # Tuning Set
                testingSet = folds.pop(i)  # Testing Set
                trainingSet = pandas.concat(folds, ignore_index=True)
                folds.insert(i, testingSet)
                folds.insert(i, tuningSet)

                # Declare common variables

                xargs = {}
                clf = None
                name = ""
                xargs = {}
                score_name = "MSE"

                # Set experiment-specific variables

                # Linear Regression

                if (doLinearPrediction):
                    clf = SimpleLinearNetwork()
                    name = "Linear Network"
                    xargs = {
                        "learning_rate": 0.1,  # TODO: Tune
                        "stochastic_gradient_descent": False, # Not implemented yet
                        "convergence_threshold": common_convergence_threshold,
                    }

                # Neural Networks (Two Hidden Layers)

                elif (doNeuralNetwork):
                    clf = NeuralNetwork()
                    name = "Neural Network"
                    xargs = {
                        "learning_rate": 0.1,
                        "minibatch_learning": True,  # Not implemented yet
                        "convergence_threshold": common_convergence_threshold,
                        "hidden_layer_dims": [8, 8],  # TODO: Tune
                        "task": "regression"
                    }

                # Autoencoder Neural Networks (Two Hidden Layers)

                elif (doAutoencoderNetwork):
                    clf = NeuralNetwork()
                    name = "Autoencoder Neural Network"
                    xargs = {
                        "learning_rate": 0.1,  # Increasing to 0.01 seems to cause divergence
                        "minibatch_learning": True,  # Not implemented yet
                        "convergence_threshold": common_convergence_threshold,
                        "hidden_layer_dims": experiment["tuned_hidden_dims"],  # TODO: Tune
                        "task": "regression"
                    }
                else:
                    exit(285)

                # Train the model

                print(f"Training {name}")

                if (doAutoencoderNetwork):
                    clf.train_autoencoder(trainData=trainingSet, yCol=experiment["yCol"], xargs=xargs)
                elif doNeuralNetwork:
                    import time

                    t0 = time.time()
                    # Tune hidden layer sizes from 0.5 - 1.5 times the number of inputs
                    num_inputs = len(trainingSet.columns) - 1
                    # fifty_percent_less_units = int(np.floor(num_inputs / 2))
                    # fifty_percent_more_units = int(num_inputs + fifty_percent_less_units)
                    #
                    # layer1_range = range(fifty_percent_less_units, fifty_percent_more_units)
                    # layer2_range = range(fifty_percent_less_units, fifty_percent_more_units)

                    layer1_range = [15]
                    layer2_range = [38
                                    ]
                    combinations = []
                    networks = []
                    for j in layer1_range:
                        for k in layer2_range:
                            networks.append(NeuralNetwork())
                            combinations.append([j, k])

                        '''
                        Multi-threading
                        '''
                        num_procs = len(combinations)
                        jobs = []
                        queue = None
                        lock = multiprocessing.Lock()
                        manager = multiprocessing.Manager()
                        scores = manager.dict()

                        for proc_num in range(0, num_procs):
                            combination = combinations[proc_num]
                            tuning_xargs = copy.deepcopy(xargs)
                            tuning_xargs["hidden_layer_dims"] = combination
                            process = multiprocessing.Process(target=train_network,
                                                              args=(trainingSet,
                                                                    experiment["yCol"],
                                                                    tuning_xargs,
                                                                    lock,
                                                                    scores,
                                                                    tuningSet,
                                                                    proc_num))

                            jobs.append(process)

                        # Start the threads (i.e. calculate the random number lists)
                        for proc_num, j in enumerate(jobs):
                            print(f"Starting thread {proc_num}")
                            j.start()

                        # Ensure all of the threads have finished
                        for j in jobs:
                            j.join()

                        t1 = time.time()

                        print(f"Time to train {num_procs} neural networks: {t1 - t0}")

                        bestScore = np.max([x for x in scores.keys()])
                        bestDims = scores[bestScore]  # Get dims of best network
                        print(f"best dims={bestDims}")
                        tuning_xargs["hidden_layer_dims"] = bestDims
                        clf = NeuralNetwork()
                        clf.train(trainData=trainingSet, yCol=experiment["yCol"], xargs=tuning_xargs)
                else:
                    clf.train(trainData=trainingSet, yCol=experiment["yCol"], xargs=xargs)

                # Find cross-entropy at each fold

                foldScore = clf.score(testingSet)
                print(f"Fold {i} : {score_name} on testing set = {foldScore}")
                foldScores.append(foldScore)

            # Print average cross-entropy score across all folds
            meanfoldScore = np.mean(foldScores)
            print(f"Fold Mean Score: {meanfoldScore}")

        print(f"Successfully ran {experiment['experimentName']}")
        print("====****====****====****====****====****====****====\n\n")
