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
        self.previous_error = np.inf


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


    def adapt_learning_rate(self, error):
        a = .01
        b = .01
        if (self.previous_error - error) > 0:
            # increase learning rate
            self.learning_rate += a
        else:
            self.learning_rate -= self.learning_rate * b


    def train(self, trainData: pandas.DataFrame, yCol: str, xargs: dict={}):
        '''
        Train logistic classifier based on training data
        '''
        self.yCol = yCol
        self.num_inputs = len(trainData.drop(columns=[yCol]).columns) + 1
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

        def add_bias(x):
            return pandas.concat([pandas.Series([1]), x])

        def remove_output(x):
            return x.drop(labels=self.yCol)


        while(True): # loop until convergence
            epoch_num +=1
            if(epoch_num >114):
                print("hi")

            with alive_bar(len(trainData),
                           title=f"Epoch {epoch_num}, E: {self.previous_error:.2f}, \u03B7={self.learning_rate:.4f}") as bar:
                weights_delta = np.zeros(self.weights.shape)

                iter = (trainData.sample(frac=1) if self.stochastic_gradient_descent else trainData).iterrows()
                for exampleIter in iter:

                    example_with_bias = add_bias(exampleIter[1])
                    example = list(remove_output(example_with_bias))

                    actual_value = example_with_bias[yCol]
                    prediction = self.predict(exampleIter[1])

                    # Update Rule
                    for j in range(0, self.num_inputs):
                        if(type(example[j]) == str):
                            #One-hot Encoded
                            weights_delta[j] += (actual_value - prediction) * sum([int(x)*self.weights[j] for x in example[j][2:]])
                        else:
                            weights_delta[j] += (actual_value - prediction) * example[j]

                    if(self.stochastic_gradient_descent):
                        self.weights += self.learning_rate * weights_delta
                    bar()

            # Adaptive Learning Rate
            error = self.score(trainData)
            self.adapt_learning_rate(error)

            if not self.stochastic_gradient_descent:
                # Batch Update
                self.weights += (self.learning_rate * weights_delta) / len(trainData)

            # Check for convergence
            difference = np.abs(self.previous_error - error)
            if difference < self.convergence_threshold:
                break

            self.previous_error = error


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
        example = list(pandas.concat([pandas.Series([1]), example]).drop(columns=self.yCol))
        for j in range(0, self.num_inputs):
            if (type(example[j]) == str):
                # One-hot Encoded
                prediction += sum([int(x) * self.weights[j] for x in example[j][2:]])
            else:
                prediction += example[j] * self.weights[j]
        return prediction