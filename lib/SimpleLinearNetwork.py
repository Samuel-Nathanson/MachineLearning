import numpy as np
import pandas
from lib.PreprocessingTK import evaluateError
from alive_progress import alive_bar

class SimpleLinearNetwork:

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
        self.num_inputs = len(trainData.drop(columns=[yCol]).columns)
        self.xargs = xargs
        self.learning_rate = self.xargs["learning_rate"] if ("learning_rate" in xargs.keys()) else 0.001
        self.stochastic_gradient_descent = self.xargs["stochastic_gradient_descent"] if ("stochastic_gradient_descent" in self.xargs.keys()) else False
        self.convergence_threshold = self.xargs["convergence_threshold"] if "convergence_threshold" in self.xargs.keys() else 0.001

        # Initialize weights with random values between -0.01 and 0.01
        self.weights = (np.random.rand(self.num_inputs) * 0.02 ) - 0.01

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
                           title=f"Epoch #{epoch_num} - Delta: {np.sum(np.abs(weights_delta))}") as bar:
                weights_delta = np.zeros(self.weights.shape)

                iter = (trainData.sample(frac=1) if self.stochastic_gradient_descent else trainData).iterrows()
                for exampleIter in iter:
                    example = exampleIter[1].drop(labels=self.yCol)
                    actual_value = exampleIter[1][yCol]

                    prediction = self.predict(example)

                    # Update Rule
                    for j in range(0, self.num_inputs):
                        if(type(example[j]) == str):
                            #One-hot Encoded
                            weights_delta[j] += (actual_value - prediction) * sum([int(x)*self.weights[j] for x in example[j][2:]])
                        else:
                            weights_delta[j] += (actual_value - prediction) * example[j]

                    if(self.stochastic_gradient_descent):
                        self.weights += self.learning_rate * weights_delta
                        print(self.weights)

                    bar()

            if(not self.stochastic_gradient_descent):
                self.weights += ( self.learning_rate * weights_delta ) / len(trainData)
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
            # print(f"Actual Value= {testingSet[self.yCol].iloc[x]}, Predicted Score= {prediction}")
        method = "MSE"
        mse = evaluateError(predictedScores, testingSet[self.yCol], method=method)
        return mse

    def predict(self, example: pandas.Series):
        '''
        Predict a class label using the simple linear network
        '''
        prediction = 0
        x = example.drop(columns=[self.yCol])
        for j in range(0, self.num_inputs):
            if (type(example[j]) == str):
                # One-hot Encoded
                prediction += sum([int(x) * self.weights[j] for x in example[j][2:]])
            else:
                prediction += example[j] * self.weights[j]
        return prediction