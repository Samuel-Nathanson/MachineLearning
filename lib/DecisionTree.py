import copy

import numpy as np
import pandas
import scipy.stats
from lib.PreprocessingTK import *
from uuid import uuid4


class DecisionNode(object):
    def __init__(self, parent: object=None, data: dict={}):
        '''
        Constructor for
        :param parent: Set parent
        :param data: Set data
        :param type: "leaf" or "rule"
        '''
        self.parent = parent
        self.data = data
        self.children = []

    def __repr__(self):
        '''
        String representation of the Decision Tree Node
        :return: string
        '''
        def recursiveRepr(self, level=0):
            retVal = ""
            if (self.data["type"] == "leaf"):
                # If the node is a leaf, set the return value to the leaf value and #examples covered
                retVal = f"Leaf: value={self.data['value']}. N={self.data['numExamples']}\n"
            elif (self.data["type"] == "rule"):
                # Else if the node is a rule, set the return value to "split on attribute=value, N=#examples"
                numExamples = self.data['numExamples']
                splitAttribute = self.data['splitAttribute']
                splitValue = self.data['splitValue'] if self.data['splitValue'] else 'Nominal Categorical'
                retVal = f"Rule: split on {splitAttribute}={splitValue}. N={numExamples}\n"
            else:
                # Otherwise, this node is bad, raise an exception
                raise Exception("Can not print node without leaf or rule type!")

            # Now, retVal is set. Prefix each return string with the depth
            retVal = "---" * level + str(level) + ": " + retVal
            # If the node has no children
            if (self.data["type"] == "leaf"):
                return retVal
            for child in self.children:
                # Append nodes to the string from the tree recursively
                retVal += recursiveRepr(child, level + 1)
            return retVal
        # Return recursive representation
        return recursiveRepr(self)

    def __str__(self):
        '''Convert tree to string'''
        return self.__repr__()

    def findById(self, id: str=None):
        '''
        Find a node by UUID4: ID is set by the markNodes function
        :param id: UUID4,
        :return: The DecisionNode with id=id. Otherwise None, if no match found
        '''
        nodeId = self.data["id"]
        if(nodeId == id):
            return self
        elif(self.children):
            for child in self.children:
                foundNode = child.findById(id)
                if(foundNode != None):
                    return foundNode
                else:
                    continue
        return None

    def markNodes(self, key="__annotation__", value="marked", demark=False, nodeList=[]):
        '''
        Preorder traversal for marking tree nodes
        :param key: Key to mark in DecisionNode.data
        :param value: Value to mark in DecisionNode.data
        :param demark: This function has the ability to demark a node
        :param nodeList: MUTABLE list of marked nodes.
        :return: None
        '''

        if (not demark):
            self.data[key] = value
            self.data['id'] = str(uuid4())
        else:
            # De-mark option allows us to purge marks from the tree
            if(key in self.data):
                self.data.pop(key)
                self.data.pop('id')

        nodeList.append(self)

        children = self.children

        if (self.children):
            # Recursively mark all children
            for i in range(0, len(children)):
                self.children[i].markNodes(key=key, value=value, nodeList=nodeList)
        else:
            return

        return nodeList

    def getSubtreeClassCounts(self):
        type = self.data["type"]
        if(type == "leaf"):
            if("subtreeClassCounts" in self.data.keys()):
                return self.data["subtreeClassCounts"]
            else:
                return {
                    self.data["value"] : self.data["numExamples"]
                }
        else:
            subtreeCounts = {}
            for child in self.children:
                countsDict = child.getSubtreeClassCounts()
                for key in countsDict:
                    if(key in subtreeCounts.keys()):
                        subtreeCounts[key] += countsDict[key]
                    else:
                        subtreeCounts[key] = countsDict[key]
            return subtreeCounts


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
                xargs: dict={"PruningSet": None, "NominalValues": []}):
        super().train(trainingSet=trainingSet, yCol=yCol, xargs=xargs)
        self.pruningSet = xargs["PruningSet"]

        # Generate the decision tree
        self.tree = self.generateTree(trainingSet, None)

        if(not self.xargs["PruningSet"].empty):
            self.postPruneTree()

    def predict(self,
                example: pandas.Series):
        '''
        Predict the values of a set of examples using the ID3DecisionTree
        :param example: Example to predict.
        :return: data frame of predictions
        '''

        def dfsTraversal(node: DecisionNode, example: pandas.DataFrame, verbose=False):
            try:
                data = node.data
            except:
                print(node)
            nodeType = data["type"]
            if(nodeType == "rule"):
                # Branch further if this node is a rule
                attribute = data["splitAttribute"]
                splitValue = data["splitValue"]
                isNominal = splitValue == None

                if(isNominal):
                    raise NotImplementedError
                    # TODO: Implement
                else:
                    if(example[attribute] < splitValue):
                        if(verbose):
                            print(f"test[\"{attribute}\"] < {splitValue}: True")
                        return dfsTraversal(node.children[0], example) # TODO: Modify param to DF
                    else:
                        if(verbose):
                            print(f"test[\"{attribute}\"] < {splitValue}: False")
                        return dfsTraversal(node.children[1], example)
            elif(nodeType == "leaf"):
                if(verbose):
                    print(f"Reached leaf with value: {data['value']}")
                # Return data value if this node is a leaf
                return data["value"]
            else:
                # Otherwise, this node is bad, raise an exception
                raise Exception("Node without leaf or rule type!!")

        if(self.tree == None):
            raise Exception("Tree not built - Please train with DecisionTree.train(...)")

        return dfsTraversal(self.tree, example)

    def postPruneTree(self, d: int=0):
        if(self.pruningSet.empty or self.tree == None):
            raise Exception("Please call DecisionTree.train with a pruning set in xargs first")

        markedNodes = []
        self.tree.markNodes("pruneInfo", {"tested": False}, nodeList=markedNodes)

        currAccuracy = self.score(testingSet=self.pruningSet)
        bestAccuracy = currAccuracy
        print(f"Current accuracy on pruning set= {bestAccuracy}")
        bestCandidateTree = None
        bestNodeId = None

        for node in markedNodes:
            if(node.parent == None):
                pass
                # Skip the root of the tree
            if(node.data["type"] == "leaf"):
                pass # Skip leaves
            else:
                treeCopy = copy.deepcopy(self.tree)
                nodeToPrune = treeCopy.findById(node.data["id"])

                nodeToPrune.data["type"] = "leaf"
                nodeToPrune.data["value"] = 0.0 # placeholder

                # Accumulate class examples in sub-tree
                subtreeClassCounts = {}
                for child in nodeToPrune.children:
                    classCounts = child.getSubtreeClassCounts()
                    for cls in classCounts.keys():
                        subtreeClassCounts[cls] = classCounts[cls]

                # Compute First Mode:
                n = 0
                mode = 0
                for cls in subtreeClassCounts.keys():
                    m = subtreeClassCounts[cls]
                    if m > n:
                        n = m
                        mode = cls
                nodeToPrune.data["value"] = mode
                # Add class counts, since our tree now has some impurity
                nodeToPrune.data["classCounts"] = subtreeClassCounts

                # Prune Children
                nodeToPrune.children = []

                candidateTree = ID3ClassificationTree()
                candidateTree.tree = treeCopy
                candidateTree.yCol = self.yCol

                # treeCopy now contains pruned node
                accuracy = candidateTree.score(testingSet=self.pruningSet)
                if(accuracy > bestAccuracy):
                    bestAccuracy = accuracy
                    bestCandidateTree = candidateTree
                    bestNodeId=node.data["id"]
        # Prune Node
        if(bestCandidateTree == None):
            print(f"Pruning any more branches would decrease accuracy. Returning best candidate tree with {d} pruned branches")
            return self.tree
        else:
            self.tree = bestCandidateTree.tree
            bestAccuracy = self.score(testingSet=self.pruningSet)
            print(f"Candidate tree accuracy > current accuracy: ({bestAccuracy} > {currAccuracy}). Deciding to prune node {bestNodeId}")
            print(f"Attempting to improve accuracy further with another iteration of pruning...")
            return self.postPruneTree(d+1)

    def score(self, testingSet):
        '''
        Scores the accuracy of the ID3 Decision Tree
        :param testingSet: Testing set to work with
        :return: returns accuracy measure
        '''
        accuracies = []
        predictedScores = []
        classLabel = np.unique(testingSet[self.yCol])[1]
        # print(f"Testing accuracy for class {classLabel}")
        for x in range(0, len(testingSet)):
            prediction = self.predict(testingSet.iloc[x])
            predictedScores.append(prediction)
        method = "accuracy"
        accuracy = evaluateError(predictedScores, testingSet[self.yCol], method=method,
                                  classLabel=classLabel)
        return accuracy


    def generateTree(self,
                     examples: pandas.DataFrame,
                     node: DecisionNode=None):
        '''
        Generate the ID3 Decision Tree
        :param examples: Example training data
        :param node: Parent Node, which this tree will append children to.
        :return: DecisionNode
        '''

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
                        split1 = examples[examples[attribute] <= splitPoint]
                        split2 = examples[examples[attribute] > splitPoint]
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
                split1 = examples[examples[bestAttribute] <= bestSplitVal]
                split2 = examples[examples[bestAttribute] > bestSplitVal]
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

            return decisionNode

    def computeIntrinsicValue(self,
                                examples: pandas.DataFrame,
                                attribute: str):
        '''
        Computes the intrinsic value of an attribute at a split point
        :param examples: Examples to be partitioned
        :param attribute: Categorical Attribute to split on
        :return: Numeric - Entropy of the splits
        '''
        # Find the number of total examples
        nExamples = len(examples)
        branches = None

        uniqueValues = np.unique(examples[attribute])

        intrinsicValue = 0
        for value in branches:
            # subset of examples with value V
            sV = len(examples[examples[attribute] == value])
            intrinsicValue -= (sV/nExamples) * np.log2(sV/nExamples)
        return intrinsicValue

    def computeEntropy(self,
                       partitions: list[pandas.DataFrame]):
        '''
        Computes Entropy of multiple partitions containing class examples
        :param partitions: List of pandas.DataFrame
        :return: Entropy of partitions
        '''
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
        '''
        Computes information gain ratio, proposed by Quinlan
        :param infoGain: Information Gain
        :param intrinsicValue: Intrinsic Value
        :return: Information Gain divided by Intrinsic Value
        '''
        return infoGain / intrinsicValue

