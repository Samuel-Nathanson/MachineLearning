import numpy as np
import pandas
from lib.PreprocessingTK import evaluateError
from lib.SNUtils import zero_ish, sigmoid, softmax, sigmoid_derivative
from alive_progress import alive_bar

class AutoencoderNetwork:

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
        self.activation_function = xargs["basis_function"] if "basis_function" in xargs else sigmoid
        self.hidden_layer_dims = xargs["hidden_layer_dims"] if "hidden_layer_dims" in xargs else []
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
        else:
            self.num_outputs = 1

        # We'll use this to generate the number of nodes for each hidden unit.
        # + Add one node to the input layer and each hidden layer for the bias unit.
        self.hidden_layer_dims = [x + 1 for x in self.hidden_layer_dims]

        layer_dims = [self.num_inputs+1] + self.hidden_layer_dims + [self.num_outputs]

        self.layers = []
        self.errors = []
        self.layer_inputs = []
        self.weight_updates = []

        # Each layer has a set of weights. Set the weights for each of the layers
        for i, layer_dim in enumerate(layer_dims):
            if(i+1 == len(layer_dims)): # Output layer has no additional outputs
                break
            else:
                self.layers.append(zero_ish([layer_dims[i], layer_dims[i+1]]))
                self.errors.append(np.zeros(layer_dims[i+1]))
                self.layer_inputs.append(np.zeros(layer_dims[i]))
                self.weight_updates.append(np.zeros([layer_dims[i], layer_dims[i+1]]))


        # backpropagation

        epoch = 0
        update_delta = 0

        while(True):
            epoch += 1

            weight_updates = []
            for layer in self.layers:
                weight_updates.append(np.zeros(layer.shape))

            for t in range(0, len(trainData)):
                xt = trainData.iloc[t]
                predicted_value = self.predict(xt)
                actual_value = self.one_hot_code(xt[yCol]) if self.task == "classification" else xt[yCol]
                # Append bias unit and convert to list
                xt = list(pandas.concat([pandas.Series([1]), xt]))

                '''
                Compute Backpropagated Error at Each of the Hidden Units
                '''

                # Update final layer error
                self.errors[-1] = (actual_value - predicted_value)

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

                    weight_updates[layer_num] = np.add(weight_updates[layer_num], layer_delta)

            '''
            Apply updates after all training examples
            '''
            for layer_num in range(0, len(self.layers)):
                # Helpful variables to improve clarity
                layer = self.layers[layer_num]
                layer_weight_deltas = weight_updates[layer_num]
                self.layers[layer_num] = np.add(layer, layer_weight_deltas * self.learning_rate)

            '''
            Compute Convergence
            '''
            new_update_delta = 0
            for layer_num in range(0, len(self.layers)):
                layer_weight_deltas = weight_updates[layer_num]
                new_update_delta += np.sum(layer_weight_deltas)
            if(np.abs((new_update_delta - update_delta + 1e-8) / (update_delta + 1e-6)) < self.convergence_threshold):
                # Convergence reached
                break
            else:
                self.learning_rate = 9 * self.learning_rate / 10 # Gradually reduce learning rate in time for convergence
                update_delta = new_update_delta

            # Score, just for fun
            print(f"Score = {self.score(trainData)}")



    def score(self, testingSet: pandas.DataFrame):
        '''
        Score logistic classifier based on testing data
        '''
        predictedScores = []
        for x in range(0, len(testingSet)):
            prediction = self.predict(testingSet.iloc[x])
            predictedScores.append(prediction)
            # print(f"Actual Value= {testingSet[self.yCol].iloc[x]}, Predicted Score= {prediction}")
        method = "cross-entropy" if self.task == "classification" else "MSE"

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
                if(not is_final_layer):
                    output = self.activation_function(output)
                outputs[i] = output
            # Set inputs of next layer equal to outputs of this layer
            self.layer_inputs[layer_num] = inputs
            inputs = outputs

        if(self.task == "classification"):
            # Use softmax to set the class probabilities
            prediction = softmax(outputs)
        else:
            # Set output equal to sum of final layer
            prediction = sum(outputs)

        return prediction
