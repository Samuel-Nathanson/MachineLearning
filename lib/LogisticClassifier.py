import pandas
from lib.PreprocessingTK import evaluateError

class LogisticClassifier:

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


    def train(self, trainData: pandas.DataFrame, yCol: str):
        '''
        Train logistic classifier based on training data
        '''
        self.yCol = yCol

    def score(self, testingSet: pandas.DataFrame):
        '''
        Score logistic classifier based on testing data
        '''
        predictedScores = []
        for x in range(0, len(testingSet)):
            prediction = self.predict(testingSet.iloc[x])
            predictedScores.append(prediction)
        print("Cross-Entropy not implemented yet - Results not valid")
        method = "cross-entropy"
        mse = evaluateError(predictedScores, testingSet[self.yCol], method=method)
        return mse

    def predict(self, example: pandas.Series):
        '''
        Predict a class label using the logistic classifier
        '''
        pass