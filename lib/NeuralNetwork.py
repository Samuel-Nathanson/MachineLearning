import numpy as np
import pandas
from lib.PreprocessingTK import evaluateError,
from lib.SNUtils import zero_ish

class NeuralNetwork:

    def __init__(self):
        '''
        Constructor
        '''
        self.yCol = ""


    def __str__(self):
        '''
        String conversion
        '''
        pass


    def __repr__(self):
        '''
        String conversion for print
        '''
        pass


    def train(self, trainData: pandas.DataFrame, yCol: str, xargs: dict={}):
        '''
        Train logistic classifier based on training data
        '''
        self.yCol = yCol
        self.activation_function = xargs["activation_function"] if "activation_function" in xargs else "sigmoid"
        self.hidden_layer_dims = xargs["hidden_layer_dims"] if "hidden_layer_dims" in xargs else [10]
        self.convergence_threshold = xargs["convergence_threshold"] if "convergence_threshold" in xargs else 0.01
        self.minibatch_learning = xargs["minibatch_learning"] if "minibatch_learning" in xargs else False
        self.learning_rate = xargs["learning_rate"] if "learning_rate" in xargs else 0.01
        self.task = xargs["task"] if "task" in xargs else "classification"
        self.weights = None

        # Set weights
        # hidden_layer_dims contains an array containing the dimensions of each hidden layer
        # e.g. [8,8] means two hidden layers with 8 nodes each

        self.num_inputs = len(trainData.drop(columns=[yCol]).columns)

        if(self.task == "classification"):
            self.unique_vals = np.unique(trainData[yCol])
            vectorized_one_hot_coder = np.vectorize(lambda val, u_val: 1 if val == u_val else 0)
            self.one_hot_code = lambda x: vectorized_one_hot_coder(x, self.unique_vals)
            self.num_outputs = len(np.unique(trainData[yCol]))


        layer1_weights = zero_ish([self.num_inputs, self.hidden_layer_dims[0]])
        layer2_weights = zero_ish([self.hidden_layer_dims[0], self.hidden_layer_dims[1]])
        output_dim = self.num_outputs if self.task == "classification" else self.hidden_layer_dims[1]
        output_layer_weights = zero_ish([self.hidden_layer_dims[1], output_dim])

        self.layers = [layer1_weights, layer2_weights, output_layer_weights]


    def score(self, testingSet: pandas.DataFrame):
        '''
        Score logistic classifier based on testing data
        '''
        predictedScores = []
        for x in range(0, len(testingSet)):
            prediction = self.predict(testingSet.iloc[x])
            predictedScores.append(prediction)
            # print(f"Actual Value= {testingSet[self.yCol].iloc[x]}, Predicted Score= {prediction}")
        method = "MSE"
        mse = evaluateError(predictedScores, testingSet[self.yCol], method=method)
        return mse

    def predict(self, example: pandas.Series):
        '''
        Predict a class label using the logistic classifier
        '''
        inputs = example.drop(columns=[self.yCol])
        for layer in self.layers:
            # Compute outputs of layer and use as inputs to next layer
            # If this is the final layer, we can either return the distribution (for classification)
            # Or compute the average (for regression)
