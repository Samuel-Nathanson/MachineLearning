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
                retVal = f"Leaf: value={self.data['value']}. #Examples={len(self.data['examples'])}\n"
            elif (self.data["type"] == "rule"):
                # Else if the node is a rule, set the return value to "split on attribute=value"
                splitAttribute = self.data['splitAttribute']
                splitValue = self.data['splitValue'] if self.data['splitValue'] else 'Nominal Categorical'
                retVal = f"Rule: split on {splitAttribute}={splitValue}. #Examples={len(self.data['examples'])}\n"
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
        # Recursively Mark all subtrees
        if (self.children):
            # Recursively mark all children
            for i in range(0, len(self.children)):
                self.children[i].markNodes(key=key, value=value, nodeList=nodeList)
        return


    def getSubtreeClassCounts(self):
        '''
        This function should only be used for ID3 Decision Trees
        :return: Class Counts
        '''
        type = self.data["type"]
        if(type == "leaf"):
            # TODO: Investigate whether the counts for pruned / merged branches are taken into account
            if("subtreeClassCounts" in self.data.keys()):
                return self.data["subtreeClassCounts"]
            else:
                return {
                    self.data["value"] : len(self.data["examples"])
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
    tuningSet = None
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
                xargs: dict={"PruningSet": None, "CategoricalValues": []}):
        super().train(trainingSet=trainingSet, yCol=yCol, xargs=xargs)
        self.pruningSet = xargs["PruningSet"]

        # Generate the decision tree
        self.tree = self.generateTree(trainingSet, None)

        if(type(self.xargs["PruningSet"]) == pandas.DataFrame):
            self.postPruneTree()

    def predict(self,
                example: pandas.Series):
        '''
        Predict the values of a set of examples using the ID3DecisionTree
        :param example: Example to predict.
        :return: data frame of predictions
        '''

        def dfsTraversal(node: DecisionNode, example: pandas.DataFrame, verbose=True):
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
                    for child in node.children:
                        childAttributeValue = child.data["examples"][attribute].iloc[0]
                        if (example[attribute] == childAttributeValue):
                            if (verbose):
                                print(f"test[\"{attribute}\"] == {childAttributeValue}: True")
                            return dfsTraversal(child, example)
                        # In this case, none of them matched.
                        # TODO: More intelligent logic than picking one at random
                    leftmostAttributeValue = node.children[0].data["examples"][attribute].iloc[0]
                    if (verbose):
                        print(f"test[\"{attribute}\"] didn't match any rules - Using leftmost branching rule")
                        print(f"\tRule: {attribute} == {leftmostAttributeValue}")
                    return dfsTraversal(node.children[0], example)
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
            print(f"POST-PRUNING: Pruned node {bestNodeId} to construct tree with higher accuracy on pruning set ({bestAccuracy} > {currAccuracy}).")
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

        method = "cross-entropy"
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

        def makeLeafNode():
            leafNode = DecisionNode()
            leafNode.children = None
            leafNode.data = {"type": "leaf", "value": scipy.stats.mode(examples[self.yCol])[0][0], "examples": examples}
            leafNode.parent = parentNode
            return leafNode
        parentNode = node
        # If the group has entropy lower than the threshold, create a leaf node.
        currtEntropy = self.computeEntropy([examples])
        if(currtEntropy <= self.entropyThreshold):
            return makeLeafNode()
        else:
            maxInformationGainRatio = 0
            bestAttribute = None
            bestSplitVal = None

            attributes = examples.columns.drop(self.yCol)
            for attribute in attributes:
                if(attribute in self.xargs["CategoricalValues"]):
                    uniqueValues = np.unique(examples[attribute])
                    '''Compute Information Gain Ratio of splitting on this categorical attribute'''
                    for uniqueValue in uniqueValues:
                        splits = [examples[examples[attribute] == uniqueValue] for uniqueValue in \
                                  np.unique(examples[attribute])]
                        '''Compute Information Gain Ratio'''
                        totalEntropy = self.computeEntropy(splits)
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
                    if(len(e) == 1):
                        continue
                    splitPoints = np.unique([(e[i] + e[i+1]) / 2 for i in range(0, len(e)-1)])
                    # Iterate over each possible split
                    for splitPoint in splitPoints:
                        '''Split on each point and compute total entropy'''
                        split1 = examples[examples[attribute] <= splitPoint]
                        split2 = examples[examples[attribute] > splitPoint]
                        splits = [split1, split2]

                        '''Compute Information Gain Ratio'''
                        totalEntropy = self.computeEntropy(splits)
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
            decisionNode.data = {"type": "rule", "splitAttribute": bestAttribute, "splitValue": bestSplitVal, "examples": examples}
            decisionNode.parent = parentNode

            if(bestAttribute == None):
                return makeLeafNode()
            if(bestSplitVal == None):
                # Nominal Feature
                branchValues = np.unique(examples[bestAttribute])
                if(len(branchValues) == 1 ):
                    return makeLeafNode()
                for value in branchValues:
                    branchExamples = examples[examples[bestAttribute] == value]
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

        branches = np.unique(examples[attribute])

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


class CARTRegressionTree(DecisionTree):


    def __init__(self):
        super().__init__()
        self.pruningSet = None
        self.mseThreshold = 0.0
    def __repr__(self):
        return super().__repr__()
    def __str__(self):
        return super().__str__()

    def train(  self,
                trainingSet: pandas.DataFrame, \
                yCol: str, \
                xargs: dict={"TuningSet": None, "CategoricalValues": []}):
        super().train(trainingSet=trainingSet, yCol=yCol, xargs=xargs)
        self.tuningSet = xargs["TuningSet"]

        # Generate the decision tree
        #
        if(type(self.xargs["TuningSet"]) != pandas.DataFrame):
            self.tree = self.generateTree(trainingSet, None)
        else:
            print("Tuning MSE Threshold for CART Regression Tree..")
            bestMSEThreshold = 0.0
            bestTreeMSE = np.inf
            bestCandidateTree = None
            # Compute average difference of points
            dataRange = trainingSet[self.yCol].max() -  trainingSet[self.yCol].min()
            avgDiff = dataRange / len(trainingSet[self.yCol])
            min = 50
            max = 51

            t2 = copy.deepcopy(self) # Used only for scoring

            for mseThreshold in np.arange(min, max)*avgDiff:
                print(f"Building CART Regression Tree with MSE Threshold={mseThreshold}")
                self.mseThreshold = mseThreshold
                candidateTree = self.generateTree(trainingSet, None)

                # Compute Score by building a CART Tree
                t2.tree = candidateTree
                treeMSE = t2.score(self.xargs["TuningSet"])

                if(treeMSE < bestTreeMSE):
                    print(f"Candidate tree MSE ({treeMSE}) < previous best candidate tree MSE ({bestTreeMSE}) - New Best Candidate")
                    bestTreeMSE = treeMSE
                    bestCandidateTree = candidateTree
                    bestMSEThreshold = mseThreshold
                else:
                    print(f"Passing this candidate: tree MSE ({treeMSE}) > best candidate tree MSE ({bestTreeMSE})")

            self.tree = bestCandidateTree

    def predict(self,
                example: pandas.Series):
        '''
        Predict the values of a set of examples using the ID3DecisionTree
        :param example: Example to predict.
        :return: data frame of predictions
        '''
        def dfsTraversal(node: DecisionNode, example: pandas.DataFrame, verbose=True):
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
                    for child in node.children:
                        childAttributeValue = child.data["examples"][attribute].iloc[0]
                        if(example[attribute] == childAttributeValue):
                            if (verbose):
                                print(f"test[\"{attribute}\"] == {childAttributeValue}: True")
                            return dfsTraversal(child, example)
                        # In this case, none of them matched.
                        # TODO: More intelligent logic than picking one at random
                    leftmostAttributeValue = node.children[0].data["examples"][attribute].iloc[0]
                    if (verbose):
                        print(f"test[\"{attribute}\"] didn't match any rules - Using leftmost branching rule")
                        print(f"\tRule: {attribute} == {leftmostAttributeValue}")
                    return dfsTraversal(node.children[0], example)
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


    def score(self, testingSet):
        '''
        Scores the MSE of the CART Decision Tree
        :param testingSet: Testing set to work with
        :return: returns MSE
        '''
        predictedScores = []
        for x in range(0, len(testingSet)):
            prediction = self.predict(testingSet.iloc[x])
            predictedScores.append(prediction)
            print(f"Actual Value= {testingSet[self.yCol].iloc[x]}, Predicted Score= {prediction}")
        method = "MSE"
        mse = evaluateError(predictedScores, testingSet[self.yCol], method=method)
        return mse


    def generateTree(self,
                     examples: pandas.DataFrame,
                     node: DecisionNode=None):
        '''
        Generate the ID3 Decision Tree
        :param examples: Example training data
        :param node: Parent Node, which this tree will append children to.
        :return: DecisionNode
        '''
        def makeLeafNode():
            leafNode = DecisionNode()
            leafNode.children = None
            leafNode.data = {"type": "leaf", "value": np.mean(examples[self.yCol]), "examples": examples}
            leafNode.parent = parentNode
            return leafNode

        parentNode = node
        # If the group has MSE lower than the MSE threshold, create a leaf node.
        currtMSE = self.computeMSE([examples])
        if(currtMSE <= self.mseThreshold):
            print(f"EARLY STOPPING: MSE of these {len(examples)} examples (MSE={currtMSE}) is less than MSE Threshold ({self.mseThreshold})")
            print(f"Making Leaf Node with value={np.mean(examples[self.yCol])}")
            return makeLeafNode()
        else:
            maxMSEReduction = -1
            bestAttribute = None
            bestSplitVal = None

            attributes = examples.columns.drop(self.yCol)
            for attribute in attributes:
                if(attribute in self.xargs["CategoricalValues"]):
                    '''Compute Information Gain Ratio of splitting on this categorical attribute'''
                    splits = [examples[examples[attribute] == uniqueValue] for uniqueValue in \
                              np.unique(examples[attribute])]
                    totalMSE = self.computeMSE(splits)

                    '''Compute MSE Reduction'''
                    mseReduction = currtMSE - totalMSE

                    '''If this is the best MSE Reduction we've seen so far, note it'''
                    if(mseReduction > maxMSEReduction):
                        maxMSEReduction = mseReduction
                        bestAttribute = attribute
                        bestSplitVal = None
                else:
                    # Sort examples by attribute value
                    e = np.unique(examples.sort_values(by=[attribute])[attribute])
                    # Find a set of unique split points to test
                    splitPoints = np.unique([(e[i] + e[i+1]) / 2 for i in range(0, len(e)-1)])
                    # Iterate over each possible split
                    for splitPoint in splitPoints:
                        '''Split on each point and compute total MSE'''
                        split1 = examples[examples[attribute] <= splitPoint]
                        split2 = examples[examples[attribute] > splitPoint]
                        splits = [split1, split2]
                        totalMSE = self.computeMSE(splits)

                        '''Compute MSE Reduction'''
                        mseReduction = currtMSE - totalMSE

                        '''If this is the best MSE Reduction we've seen so far, note it'''
                        if(mseReduction > maxMSEReduction):
                            maxMSEReduction = mseReduction
                            bestAttribute = attribute
                            bestSplitVal = splitPoint

            decisionNode = DecisionNode()
            decisionNode.data = {"type": "rule", "splitAttribute": bestAttribute, "splitValue": bestSplitVal, "examples": examples}
            decisionNode.parent = parentNode

            if(bestSplitVal == None):
                # Nominal Feature
                branchValues = np.unique(examples[bestAttribute])
                if(len(branchValues) == 1):
                    return makeLeafNode()
                for value in branchValues:
                    branchExamples = examples[examples[bestAttribute] == value]
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

            return decisionNode

    def computeMSE(self,
                       partitions: list[pandas.DataFrame]):
        '''
        Computes MSE of multiple partitions containing class examples
        :param partitions: List of pandas.DataFrame
        :return: Entropy of partitions
        '''
        # Find the number of total examples
        totalExamples = np.sum([len(x) for x in partitions])

        # Variable to store the total entropy
        totalSE = 0

        # Compute total entropy
        for partition in partitions:
            partitionSE = 0
            exampleMean = np.mean(partition[self.yCol]) # TODO: Genericize
            for index, example in partition.iterrows():
                partitionSE += (example[self.yCol] - exampleMean) ** 2
            totalSE += partitionSE
        totalMSE = totalSE / totalExamples


        return totalMSE
