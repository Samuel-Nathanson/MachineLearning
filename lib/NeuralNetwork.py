import numpy as np
import pandas
from lib.PreprocessingTK import evaluateError
from lib.SNUtils import zero_ish, sigmoid, softmax, sigmoid_derivative
import copy
from alive_progress import alive_bar

class NeuralNetwork:

    def __init__(self):
        '''
        Constructor
        '''
        self.yCol = ""
        self.initialized = False

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


    def initialize(self, trainData: pandas.DataFrame, yCol: str, xargs: dict={}):
        self.yCol = yCol
        self.activation_function = xargs["basis_function"] if "basis_function" in xargs else sigmoid
        self.hidden_layer_dims = xargs["hidden_layer_dims"] if "hidden_layer_dims" in xargs else []
        self.convergence_threshold = xargs["convergence_threshold"] if "convergence_threshold" in xargs else 0.01
        self.minibatch_learning = xargs["minibatch_learning"] if "minibatch_learning" in xargs else False
        self.learning_rate = xargs["learning_rate"] if "learning_rate" in xargs else 0.01
        self.task = xargs["task"] if "task" in xargs else "classification"

        # Set weights
        # hidden_layer_dims contains an array containing the dimensions of each hidden layer
        # e.g. [8,8] means two hidden layers with 8 nodes each
        self.num_inputs = len(trainData.drop(columns=[yCol]).columns)

        if (self.task == "classification"):
            self.unique_vals = np.unique(trainData[yCol])
            vectorized_one_hot_coder = np.vectorize(lambda val, u_val: 1 if val == u_val else 0)
            self.one_hot_code = lambda x: vectorized_one_hot_coder(x, self.unique_vals)
            self.num_outputs = len(np.unique(trainData[yCol]))
        elif (self.task == "regression"):
            self.num_outputs = 1
        elif (self.task == "autoencode"):
            self.num_outputs = len(trainData[yCol].iloc[0])

        # We'll use this to generate the number of nodes for each hidden unit.
        # + Add one node to the input layer and each hidden layer for the bias unit.
        self.hidden_layer_dims = [x + 1 for x in self.hidden_layer_dims]

        layer_dims = [self.num_inputs + 1] + self.hidden_layer_dims + [self.num_outputs]

        self.layers = []
        self.errors = []
        self.layer_inputs = []
        self.previous_error = np.inf
        self.stochastic_gradient_descent = False

        # Each layer has a set of weights. Set the weights for each of the layers
        for i, layer_dim in enumerate(layer_dims):
            if (i + 1 == len(layer_dims)):  # Output layer has no additional outputs
                break
            else:
                self.layers.append(zero_ish([layer_dims[i], layer_dims[i + 1]]))
                self.errors.append(np.zeros(layer_dims[i + 1]))
                self.layer_inputs.append(np.zeros(layer_dims[i]))

        self.initialized = True


    def train_autoencoder(self, trainData: pandas.DataFrame, yCol: str, xargs: dict={}):
        '''
        Train Autoencoder Neural Network based on training data
        '''
        autoencoder_train_data = copy.deepcopy(trainData)
        autoencoder_train_data.drop(columns=[yCol], inplace=True)
        new_xargs = copy.deepcopy(xargs)
        new_xargs["task"] = "autoencode"

        # Train autoencoder with output layer set to replicated inputs
        self.yCol = "AUTOENCODER_REPLICATED_INPUTS"
        autoencoder_train_data[self.yCol] = autoencoder_train_data.values.tolist()
        self.train(autoencoder_train_data, self.yCol, xargs=new_xargs)

        # Reset some indices
        self.yCol = yCol
        self.task = xargs["task"] if "task" in xargs else ""

        if(self.task == "classification"):
            self.unique_vals = np.unique(trainData[yCol])
            vectorized_one_hot_coder = np.vectorize(lambda val, u_val: 1 if val == u_val else 0)
            self.one_hot_code = lambda x: vectorized_one_hot_coder(x, self.unique_vals)

        # Remove Replicated Output Layer
        self.layers.pop(-1)
        self.num_outputs = self.layers[-1].shape[1]
        self.errors.pop(-1)

        # insert two new layers
        # Reset y column
        self.yCol = yCol
        first_hidden_layer_size = self.hidden_layer_dims[-1]
        new_hidden_layer_size = first_hidden_layer_size
        new_output_layer_size = len(np.unique(trainData[yCol])) if self.task == "classification" else 1
        new_hidden_layer = zero_ish([first_hidden_layer_size, new_hidden_layer_size])
        new_output_layer = zero_ish([new_hidden_layer_size, new_output_layer_size])
        new_output_error = np.zeros(new_output_layer_size)
        new_hidden_error = np.zeros(new_hidden_layer_size)
        self.layers.append(new_hidden_layer)
        self.layers.append(new_output_layer)
        self.layer_inputs.append(np.zeros(new_hidden_layer_size))
        self.errors.append(new_hidden_error)
        self.errors.append(new_output_error)

        self.train(trainData=trainData, yCol=yCol, xargs=xargs)


    def train(self, trainData: pandas.DataFrame, yCol: str, xargs: dict={}):
        '''
        Train Neural Network based on training data
        '''
        if(not self.initialized):
            self.initialize(trainData, yCol, xargs)

        # Backpropagation
        epoch = 0
        while True:
            epoch += 1

            with alive_bar(len(trainData),
                           title=f"Epoch {epoch}, E: {self.previous_error:.2f}, \u03B7={self.learning_rate:.4f}") as bar:
                weight_updates = []
                '''
                Set weight updates to zero
                '''
                for layer in self.layers:
                    weight_updates.append(np.zeros(layer.shape))

                for t in range(0, len(trainData)):
                    xt = trainData.iloc[t]
                    predicted_value = self.predict(xt)
                    actual_value = None

                    # use correct data type
                    if(self.task == "classification"):
                        actual_value = self.one_hot_code(xt[yCol])
                    elif(self.task == "regression"):
                        actual_value = [xt[yCol]]
                    elif(self.task == "autoencode"):
                        actual_value = xt[yCol]
                    else:
                        exit(286)

                    # Append bias unit and convert to list
                    xt = list(pandas.concat([pandas.Series([1]), xt]))

                    '''
                    Compute Backpropagated Error at Each of the Hidden Units
                    '''

                    # Update final layer error
                    self.errors[-1] = np.subtract(actual_value, predicted_value)

                    # Compute backpropagated error at each set of hidden units
                    for layer_num in reversed(range(1, len(self.layers))):

                        # Helpful variables to improve clarity
                        num_input_units = self.layers[layer_num].shape[0]
                        num_output_units = self.layers[layer_num].shape[1]
                        error_unit_idx = layer_num - 1

                        # Compute backpropagated error for each hidden unit in this layer
                        for j in range(0, num_input_units):
                            self.errors[error_unit_idx][j] = 0
                            for i in range(0, num_output_units):
                                error_weight = self.layers[layer_num][j][i]
                                backpropagated_error = self.errors[error_unit_idx + 1]
                                self.errors[error_unit_idx][j] += error_weight * backpropagated_error[i]

                    '''
                    Compute Updates
                    '''

                    # Iterate over each layer to update the weights
                    for layer_num in range(0, len(self.layers)):

                        # Initialize helpful variables
                        is_final_layer = layer_num == len(self.layers) - 1
                        layer = self.layers[layer_num]
                        error = self.errors[layer_num]
                        layer_inputs = self.layer_inputs[layer_num]
                        num_input_units = layer.shape[0]
                        num_output_units = layer.shape[1]

                        # Initialize weight deltas to a zeroized dxK matrix
                        layer_delta = np.zeros((num_input_units,num_output_units))

                        # Set weight updates
                        for j in range(0, num_input_units):
                            for i in range(0, num_output_units):
                                backpropagated_error = error[i]
                                input_scalar = layer_inputs[j]

                                # The final layer is a special case, since it does not require the chain rule for updating weights
                                if(is_final_layer):
                                    layer_delta[j][i] = backpropagated_error * input_scalar
                                else:
                                    layer_delta[j][i] = backpropagated_error * sigmoid_derivative(input_scalar) * input_scalar

                        # Update layer weight deltas
                        weight_updates[layer_num] = np.add(weight_updates[layer_num], layer_delta)
                    bar()


                # Adaptive Learning Rate
                error = self.score(trainData)
                self.adapt_learning_rate(error)

                '''
                Apply updates after all training examples
                '''
                if not self.stochastic_gradient_descent:
                    for layer_num in range(0, len(self.layers)):
                        # Helpful variables to improve clarity
                        layer = self.layers[layer_num]
                        layer_weight_deltas = weight_updates[layer_num] / len(trainData)
                        self.layers[layer_num] = np.add(layer, layer_weight_deltas * self.learning_rate)

                '''
                Compute Convergence
                '''
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
        method = "cross-entropy" if (self.task == "classification" or self.task == "autoencode") else "MSE"

        score = 0
        if(self.task == "classification"):
            one_hot_coded = [self.one_hot_code(r) for r in testingSet[self.yCol]]
            score = evaluateError(predictedScores, one_hot_coded, method=method)
        else:
            score = evaluateError(predictedScores, testingSet[self.yCol], method=method)

        return score


    def predict(self, example: pandas.Series):
        '''
        Predict a class label using the neural network
        '''
        prediction = None
        inputs = example.drop(labels=[self.yCol])
        # Append bias unit and convert to list
        inputs = list(pandas.concat([pandas.Series([1]), inputs]))
        for layer_num, layer in enumerate(self.layers):
            nInputs = layer.shape[0]
            nOutputs = layer.shape[1]
            is_final_layer = layer_num == (len(self.layers) - 1)

            # We set inputs to either the previous layer's outputs or the sample inputs
            outputs = [0] * nOutputs
            for i in range(0, nOutputs):
                output = 0
                for j in range(0, nInputs):
                    # Not working now?
                    output += layer[j][i] * inputs[j]
                # Don't squish the final layer with the activation function
                if not is_final_layer:
                    output = self.activation_function(output)
                outputs[i] = output
            # Set inputs of next layer equal to outputs of this layer
            self.layer_inputs[layer_num] = inputs
            inputs = outputs

        if self.task == "classification":
            # Use softmax to set the class probabilities
            prediction = softmax(outputs)
        elif self.task == "regression":
            # Set output equal to sum of final layer
            prediction = sum(outputs)
        elif self.task == "autoencode":
            prediction = outputs

        return prediction
