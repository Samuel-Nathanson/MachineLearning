import numpy as np
import pandas
from lib.PreprocessingTK import evaluateError
from alive_progress import alive_bar

class LogisticClassifier:

    def __init__(self):
        '''
        Constructor
        '''
        self.yCol = ""
        self.weights = []
        self.num_outputs = 0
        self.num_inputs = 0
        self.unique_vals = None
        self.one_hot_code = None
        self.num_outputs = 0
        self.num_inputs = 0
        self.xargs = None
        self.learning_rate = 0
        self.stochastic_gradient_descent = False
        self.convergence_threshold = 0


    def __str__(self):
        '''
        String conversion
        '''
        print(self.weights)


    def __repr__(self):
        '''
        String conversion for print
        '''
        print(self.__str__())


    def train(self, trainData: pandas.DataFrame, yCol: str, xargs: dict={}):
        '''
        Train logistic classifier based on training data
        '''
        self.yCol = yCol
        vectorized_one_hot_coder = np.vectorize(lambda val, u_val: 1 if val == u_val else 0)
        self.unique_vals = np.unique(trainData[yCol])
        self.one_hot_code = lambda x: vectorized_one_hot_coder(x, self.unique_vals)
        self.num_outputs = len(np.unique(trainData[yCol]))
        self.num_inputs = len(trainData.drop(columns=[yCol]).columns) + 1 # add 1 for bias unit
        self.xargs = xargs
        self.learning_rate = self.xargs["learning_rate"] if ("learning_rate" in xargs.keys()) else 0.001
        self.stochastic_gradient_descent = self.xargs["stochastic_gradient_descent"] if ("stochastic_gradient_descent" in self.xargs.keys()) else False
        self.convergence_threshold = self.xargs["convergence_threshold"] if "convergence_threshold" in self.xargs.keys() else 0.001

        # Initialize weights with random values between -0.01 and 0.01
        self.weights = (np.random.rand(self.num_inputs, self.num_outputs) * 0.02 ) - 0.01

        weights_delta = None

        def convergent(delta):
            if(not hasattr(convergent, "prev_delta")):
                convergent.prev_delta = delta
                return False
            else:
                difference = np.abs(np.sum(np.abs(delta)) - np.sum(np.abs(convergent.prev_delta)))
                convergent.prev_delta = delta
                if(difference < self.convergence_threshold):
                    return True
            return False

        epoch_num = 0
        weights_delta = np.zeros(self.weights.shape)

        while(True): # loop until convergence
            epoch_num +=1

            with alive_bar(len(trainData),
                           title=f"Epoch #{epoch_num} - Delta: {np.sum(np.abs(weights_delta))}",
                           bar="notes") as bar:
                weights_delta = np.zeros(self.weights.shape)

                iter = (trainData.sample(frac=1) if self.stochastic_gradient_descent else trainData).iterrows()
                for exampleIter in iter:
                    example = exampleIter[1]
                    actual_value = self.one_hot_code(example[yCol])
                    # Append bias unit and convert to list
                    example = list(pandas.concat([pandas.Series([1]), example]))
                    # This will become a valid probability distribution after applying the softmax func.
                    class_probabilities = [0] * self.num_outputs
                    for i in range(0, len(class_probabilities)):
                        # Sum the current predicted weighted values
                        for j in range(0, self.num_inputs):
                            class_probabilities[i] += self.weights[j][i] * example[j]

                    # Make predictions for distribution
                    softmax_den = np.sum([np.exp(x) for x in class_probabilities])
                    for i in range(0, len(class_probabilities)):
                        class_probabilities[i] = np.exp(class_probabilities[i]) / softmax_den

                    # Update Rule
                    for i in range(0, self.num_outputs):
                        for j in range(0, self.num_inputs):
                            weights_delta[j][i] += (actual_value[i] - class_probabilities[i]) * example[j]
                    if(self.stochastic_gradient_descent):
                        self.weights += self.learning_rate * weights_delta
                    bar()

            if(not self.stochastic_gradient_descent):
                self.weights += (self.learning_rate * weights_delta) / len(trainData)
            # Batch Update

            if(convergent(weights_delta)):
                break


    def score(self, testingSet: pandas.DataFrame):
        '''
        Score logistic classifier based on testing data
        '''
        predictedScores = []
        for x in range(0, len(testingSet)):
            prediction = self.predict(testingSet.iloc[x])
            predictedScores.append(prediction)

        method = "cross-entropy"

        one_hot_coded = [self.one_hot_code(r) for r in testingSet[self.yCol]]
        cross_ent = evaluateError(predictedScores, one_hot_coded, method=method)
        return cross_ent

    def predict(self, example: pandas.Series):
        '''
        Predict a class label using the logistic classifier
        '''
        example = list(pandas.concat([pandas.Series([1]), example]))
        class_probabilities = [0] * self.num_outputs
        for i in range(0, len(class_probabilities)):
            # Sum the current predicted weighted values
            for j in range(0, self.num_inputs):
                class_probabilities[i] += self.weights[j][i] * example[j]

        # Make predictions for distribution
        softmax_den = np.sum([np.exp(x) for x in class_probabilities])
        for i in range(0, len(class_probabilities)):
            class_probabilities[i] = np.exp(class_probabilities[i]) / softmax_den

        return class_probabilities
