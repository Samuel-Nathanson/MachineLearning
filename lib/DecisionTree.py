import copy

import numpy as np
import pandas
import scipy.stats
import re


class DecisionNode(object):
    def __init__(self, parent: object=None, data: dict={}):
        self.parent = parent
        self.data = data
        self.children = []

    def __repr__(self):
        data = self.data

        if(data["type"]=="leaf"):
            numExamples = data["numExamples"]
            returnStr = f"Leaf: Value={data['value']} | Examples: {numExamples}\n"
            return returnStr
        if(data["type"] == "rule"):
            numExamples = data["numExamples"]
            splitAttribute = data["splitAttribute"]
            splitValue = data["splitValue"] if data["splitValue"] else "Nominal Categorical"
            returnStr = f"Decision: Split Attribute={splitAttribute} | SplitValue={splitValue} | Examples: {numExamples}\n"
            for child in self.children:
                childVal = "--" + re.sub("\n", "\n--", str(child))
                returnStr += childVal
            return returnStr
        else:
            return ""

    def __str__(self):
        '''Convert tree to string'''
        return self.__repr__()


class DecisionTree(object):
    '''Decision Tree Class'''
    def __init__(self):
        '''Construct Tree'''
        self.trainingSet = None
        self.yCol = None
        self.xargs = None
        self.tree = None

    def __repr__(self):
        '''Print Tree'''
        return self.tree.__repr__()

    def __str__(self):
        '''Convert tree to string'''
        return self.__repr__()

    def train(  self,
                trainingSet: pandas.DataFrame, \
                yCol: str,
                xargs: dict=None):
        self.trainingSet = trainingSet
        self.yCol = yCol
        self.xargs = xargs
        self.tree = None

    def predict(self,
                examples: pandas.DataFrame):
        '''Virtual method: Make a prediction for examples E'''
        raise NotImplementedError()


class ID3ClassificationTree(DecisionTree):
    pruningSet = None
    entropyThreshold = 0.0

    def __init__(self):
        super().__init__()
    def __repr__(self):
        return super().__repr__()
    def __str__(self):
        return super().__str__()

    def train(  self,
                trainingSet: pandas.DataFrame, \
                yCol: str, \
                xargs: dict={"ReducedErrorPruning":False, "PruningSet": None, "NominalValues": []}):
        super().train(trainingSet=trainingSet, yCol=yCol, xargs=xargs)

        # Generate the decision tree
        self.tree = self.generateTree(trainingSet, None)

    def predict(self,
                examples: pandas.DataFrame):
        '''Make a prediction for examples E'''

    def generateTree(self,
                     examples: pandas.DataFrame,
                     node: DecisionNode=None):

        parentNode = node
        # If the group has entropy lower than the threshold, create a leaf node.
        currtEntropy = self.computeEntropy([examples])
        if(currtEntropy <= self.entropyThreshold):
            leafNode = DecisionNode()
            leafNode.children = None
            leafNode.data = {"type": "leaf", "value": scipy.stats.mode(examples[self.yCol])[0][0], "numExamples": len(examples)}
            leafNode.parent = parentNode
            return leafNode
        else:
            maxInformationGainRatio = 0
            bestAttribute = None
            bestSplitVal = None

            attributes = examples.columns.drop(self.yCol)
            for attribute in attributes:
                if(attribute in self.xargs["NominalValues"]):
                    uniqueValues = np.unique(examples[attribute])
                    '''Compute Information Gain Ratio of splitting on this categorical attribute'''
                    for uniqueValue in uniqueValues:
                        splits = [examples[examples[attribute] == uniqueValue] for uniqueValue in \
                                  np.unique(examples[attribute])]
                        totalEntropy = self.computeEntropy(splits)

                        '''Compute Information Gain Ratio'''
                        informationGain = currtEntropy - totalEntropy
                        intrinsicValue = self.computeIntrinsicValue(examples, attribute)
                        gainRatio = self.computeInfoGainRatio(informationGain, intrinsicValue)

                        '''If this is the best Information Gain Ratio we've seen so far, note it'''
                        if(gainRatio > maxInformationGainRatio):
                            maxInformationGainRatio = gainRatio
                            bestAttribute = attribute
                            bestSplitVal = None
                else:
                    # Sort examples by attribute value
                    e = np.unique(examples.sort_values(by=[attribute])[attribute])
                    # Find a set of unique split points to test
                    splitPoints = np.unique([(e[i] + e[i+1]) / 2 for i in range(0, len(e)-1)])
                    # Iterate over each possible split
                    for splitPoint in splitPoints:
                        '''Split on each point and compute total entropy'''
                        split1 = examples[examples[attribute] > splitPoint]
                        split2 = examples[examples[attribute] <= splitPoint]
                        splits = [split1, split2]
                        totalEntropy = self.computeEntropy(splits)

                        '''Compute Information Gain Ratio'''
                        informationGain = currtEntropy - totalEntropy
                        intrinsicValue = -1 * (len(split1)/len(examples)) * np.log2(len(split1) / len(examples)) \
                                         - (len(split2)/len(examples)) * np.log2(len(split2) / len(examples))
                        gainRatio = self.computeInfoGainRatio(informationGain, intrinsicValue)
                        '''If this is the best Information Gain Ratio we've seen so far, note it'''
                        if(gainRatio > maxInformationGainRatio):
                            maxInformationGainRatio = gainRatio
                            bestAttribute = attribute
                            bestSplitVal = splitPoint

            decisionNode = DecisionNode()
            decisionNode.data = {"type": "rule", "splitAttribute": bestAttribute, "splitValue": bestSplitVal, "numExamples": len(examples)}
            decisionNode.parent = parentNode

            if(bestSplitVal == None):
                # Nominal Feature
                branchValues = np.unique(examples[bestAttribute])
                for value in branchValues:
                    branchExamples = examples[examples[attribute] == value]
                    # Now, generate the tree recursively
                    childNode = self.generateTree(branchExamples, decisionNode)
                    # Append new child nodes onto this node
                    decisionNode.children.append(childNode)
            else:
                split1 = examples[examples[bestAttribute] > bestSplitVal]
                split2 = examples[examples[bestAttribute] <= bestSplitVal]
                # Now, generate the tree recursively
                childNode1 = self.generateTree(split1, decisionNode)
                childNode2 = self.generateTree(split2, decisionNode)
                # Append new child nodes onto this node
                decisionNode.children.append(childNode1)
                decisionNode.children.append(childNode2)

            d = 0
            p = parentNode
            while(p !=None):
                d +=1
                p = p.parent

            print(f"Depth={d}, Length of Children: {len(decisionNode.children)}")
            return decisionNode

    def computeIntrinsicValue(self,
                                examples: pandas.DataFrame,
                                attribute: str,
                                splitPoint=None):
        # Find the number of total examples
        nExamples = len(examples)
        branches = None

        if(splitPoint):
            # Here, we will only have two groups
            branches = np.unique(examples[attribute])
        else:
            uniqueValues = np.unique(examples[attribute])

        intrinsicValue = 0
        for value in branches:
            # subset of examples with value V
            sV = len(examples[examples[attribute] == value])
            intrinsicValue -= (sV/nExamples) * np.log2(sV/nExamples)
        return intrinsicValue

    def computeEntropy(self, partitions: list[pandas.DataFrame]):
        # Find the number of total examples
        totalExamples = np.sum([len(x) for x in partitions])

        # Variable to store the total entropy
        totalEntropy = 0

        # Compute total entropy
        for partition in partitions:
            # Get the number of examples in this partition and the unique classes
            nPartitionExamples = len(partition)
            classes = np.unique(partition[self.yCol])

            # Find the entropy for this partition
            partitionEntropy = 0
            for c in classes:
                ci = len(partition[partition[self.yCol] == c])
                partitionEntropy -= (ci/nPartitionExamples) * np.log2(ci/nPartitionExamples)

            # Add a proportion of the partition entropy to the total entropy
            totalEntropy += partitionEntropy * ( nPartitionExamples / totalExamples)

        return totalEntropy

    def computeInfoGainRatio(self, infoGain: float, intrinsicValue: float):
        return infoGain / intrinsicValue

    def computeIntrinsicValue(self, examples: pandas.DataFrame):
        totalIV = 0
        nExamples = len(examples)
        classes = np.unique(examples[self.yCol])
        for c in classes:
            ci = len(examples[examples[self.yCol] == c])
            totalIV += -1 * (ci/nExamples) * (np.log2(ci/nExamples))



class CARTRegressionTree(DecisionTree):
    def __init__(self):
        super().__init__()
    def __repr__(self):
        return super().__repr__()
    def __str__(self):
        return super().__str__()

    @classmethod
    def train(  self,
                trainingSet: pandas.DataFrame, \
                yCol: str,
                xargs: dict= {"EarlyStopping":True}):
        super().train(trainingSet=trainingSet, yCol=yCol, xargs=xargs)

        pass

    @classmethod
    def predict(self,
                examples: pandas.DataFrame):
        '''Virtual method: Make a prediction for examples E'''
        pass

    @classmethod
    def tuneEarlyStoppingThreshold(self):
        pass


