#-------------------------------------------------------------------------------
# Name:        id3_gini.py
# Purpose:     Implementation of ID3 algorithm of Decision Tree
#              for binary data with information gain for data split
#              using Gini instead of Entropy
# Reference:   http://en.wikipedia.org/wiki/ID3_algorithm
# Author:      daveti
# Email:       daveti@cs.uoregon.edu
# Blog:        http://daveti.blog.com
# Created:     28/01/2013
# Copyright:   (c) daveti 2013
# Licence:     GNU/GPLv3
#-------------------------------------------------------------------------------

import os
import sys
import math
import ctypes

class Node(object):
    "node representation in the decision tree"
    def __init__(self, n, i):
        "Initializer for the node class"
        self.name = n
        self.index = i
        self.leftNode = None     # Left means '0' branch
        self.rightNode = None    # Right means '1' branch
        self.target = 'NA'
        self.numOfSamples = -1
        self.errorRate = -1
        self.infoGain = -1

    def __str__(self):
        "String representation for the node"
        return('node(%s, %d, %s, %s, %s, %d, %f, %f)' %(
                self.name,
                self.index,
                self.leftNode.name if self.leftNode != None else 'null',
                self.rightNode.name if self.rightNode != None else 'null',
                self.target,
                self.numOfSamples,
                self.errorRate,
                self.infoGain))

    def isLeafNode(self):
        "Retrun true/false if it is a leaf node or not"
        if self.name == '+' or self.name == '-' or self.name == '#':
            return(True)
        else:
            return(False)

    # NOTE: A good programming style should make the data private
    # and give the corresponding 'get' and 'set' methods...Or I am lazy:)

def getTargetCount(trainTarget):
    "Return the number of negative and positive targets"
    funcName = 'getTargetCount'
    numOfBin0 = 0
    numOfBin1 = 0
    for t in trainTarget:
        if t == '0':
            numOfBin0 += 1
        elif t == '1':
            numOfBin1 += 1
        else:
            print(funcName + ' Error: only binary classification is supported')
    return(numOfBin0, numOfBin1)

def getMostCommonTarget(t0, t1):
    "Return the most common target value - 0/1"
    if t0 >= t1:
        return(0)
    else:
        return(1)

def log2Func(x):
    if x == 0:
        return(0)
    else:
        return(math.log(x, 2))

def computeGiniIndex(targetList):
    "Compute the Gini index based on the classification"
    numOfBin0 = 0
    numOfBin1 = 0
    numOfData = len(targetList)
    # Defensive checking
    if numOfData == 0:
        return(0)

    numOfBin0, numOfBin1 = getTargetCount(targetList)
    prob0 = float(numOfBin0) / numOfData
    prob1 = float(numOfBin1) / numOfData
    # GiniIndex = 1 - Sigma(v(a))[p(i|T)]2
    gi = 1 - (prob0*prob0 + prob1*prob1)
    return(gi)

def computeGiniSplit(dataList, targetList, attrIndex):
    "Compute the Gini split for certain attribute"
    funcName = 'computeGiniSplit'
    # Always assume the binary attribute and classification
    # Reference: http://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    # GiniSplit = Sigma(v(a))[GiniIndex(i)*ni/n]

    # Compute Gini index for this attribute
    gi = computeGiniIndex(targetList)

    # G(T|Xa=v) - G(T|Xa=0) and G(T|Xa=1)
    numOfData = len(dataList)
    targetList0 = []
    targetList1 = []
    for i in range(numOfData):
        if dataList[i][attrIndex] == '0':
            targetList0.append(targetList[i])
        elif dataList[i][attrIndex] == '1':
            targetList1.append(targetList[i])

    # Compute Gini index for each sub set after split
    gi0 = computeGiniIndex(targetList0)
    gi1 = computeGiniIndex(targetList1)

    # Compute Gini split
    gs = (len(targetList0)*gi0 + len(targetList1)*gi1)/numOfData
    return(gs)

def id3(trainData, trainTarget, attrList, attrIndexLeft, GiniList=None, debug=False):
    "Construct the tree structure using information gain"
    funcName = 'id3'

    # Create a new root node
    root = Node('root', -1)
    root.numOfSamples = len(trainTarget)

    # Check the data at first
    numOfBin0 = 0
    numOfBin1 = 0
    numOfBin0, numOfBin1 = getTargetCount(trainTarget)
    mostCommonTarget = getMostCommonTarget(numOfBin0, numOfBin1)

    if numOfBin0 == 0:
        # All data are positive
        root.name = '+'
        root.target = '1'
    elif numOfBin1 == 0:
        # All data are negative
        root.name = '-'
        root.target = '0'
    elif len(attrIndexLeft) == 0:
        # No attribute left for branching
        # Choose the most common case
        root.name = '#'
        if mostCommonTarget == 0:
            root.target = '0'
        else:
            root.target = '1'
    else:
        # Branch the tree
        # Select the attribute
        maxIG = -1
        maxIndex = -1
        for i in attrIndexLeft:
            # Get the Gini split value for this attribute
            tmpIG = computeGiniSplit(trainData, trainTarget, i)
            if tmpIG > maxIG:
                # Update the max Gini value and save the index
                maxIG = tmpIG
                maxIndex = i
            # Debug
            if debug == True:
                # Dump all the information gain at the root node
                GiniList.append((attrList[i], tmpIG))

        if debug == True:
            for i in GiniList:
                print(i)

        # Update the root node
        root.name = attrList[maxIndex]
        root.index = maxIndex
        root.infoGain = maxIG

        # Remove this attribute index
        attrIndexLeft.remove(maxIndex)

        # Split the data based on different possible values of this attribute
        trainDataL = []
        trainDataR = []
        trainTargetL = []
        trainTargetR = []
        for i in range(root.numOfSamples):
            if trainData[i][maxIndex] == '0':
                trainDataL.append(trainData[i])
                trainTargetL.append(trainTarget[i])
            elif trainData[i][maxIndex] == '1':
                trainDataR.append(trainData[i])
                trainTargetR.append(trainTarget[i])
            else:
                print(funcName + 'dataSplit' +
                    'Error: only binary attribute/classification is supported')

        # Check data split using this attribute separately
        # Left branch
        numOfBin0 = 0
        numOfBin1 = 0
        numOfBin0, numOfBin1 = getTargetCount(trainTargetL)
        numOfData = numOfBin0 + numOfBin1
        if numOfData == 0:
            # Use the most common target value before data split
            leftNode = Node('#', -1)
            leftNode.numOfSamples = numOfData
            if mostCommonTarget == 0:
                leftNode.target = '0'
            else:
                leftNode.target = '1'
            # Attach the node to the root
            root.leftNode = leftNode
        else:
            # Recursive calling
            attrIdxLeft = attrIndexLeft[:]
            root.leftNode = id3(trainDataL, trainTargetL, attrList, attrIdxLeft)

        # Right brach
        numOfBin0 = 0
        numOfBin1 = 0
        numOfBin0, numOfBin1 = getTargetCount(trainTargetR)
        numOfData = numOfBin0 + numOfBin1
        if numOfData == 0:
            # Use the most common target value before data split
            rightNode = Node('#', -1)
            rightNode.numOfSamples = numOfData
            if mostCommonTarget == 0:
                rightNode.target = '0'
            else:
                rightNode.target = '1'
            # Attach the node to the root
            root.rightNode = rightNode
        else:
            # Recursive calling
            attrIdxLeft = attrIndexLeft[:]
            root.rightNode = id3(trainDataR, trainTargetR, attrList, attrIdxLeft)

    # Return the root
    return(root)

def predict(testData, dTree):
    "Predict the classification based on the learned decision tree and return prediction"
    funcName = 'predict'
    testTarget = []
    for d in testData:
        root = dTree
        while root != None:
            if root.isLeafNode() == True:
                # Found the target
                testTarget.append(root.target)
                break
            else:
                # Find a branch
                if d[root.index] == '0':
                    # Left branch
                    root = root.leftNode
                elif d[root.index] == '1':
                    # Right branch
                    root = root.rightNode
                else:
                    print(funcName + 'Error: only binary data is supported')
                    testTarget.append('X')
                    break

    return(testTarget)


def loadCsvData(fn):
    "Load the csv data from disk into memory and return the processed lists"
    attrList = []
    dataList = []
    targetList = []
    # Verify if the file exists
    if not os.path.exists(fn):
        print('Error: %s does not exists' %fn)
    else:
        try:
            fnObj = open(fn, "r")
            isAttrLine = True
            for line in fnObj:
                line = line.strip()
                # Always assume the first line should be attribute name
                if isAttrLine == False:
                    dataList.append(line)
                else:
                    attrList = line
                    isAttrLine = False
        finally:
            fnObj.close()

        # Handle the comma and reformat the data
        # NOTE: A good way here should be verify the format of this CSV file
        # but I am...lazy

        # Attribute handling - assuming only one element in the list
        attrList = attrList.split(",")
        # Remove the last 'class' attribute
        attrList = attrList[:-1]
        # Debug
        #print('attrList:', attrList)

        # Data handling - multiple lines
        tmpList = dataList[:]
        dataList = []
        for line in tmpList:
            tmpList2 = line.split(",")
            # Save the real data for this sample
            dataList.append(tmpList2[:-1])
            # Save the target value for this sample
            targetList.append(tmpList2[-1])

    return (attrList, dataList, targetList)

def treeWriter(dTree, fn):
    "Write the decision tree to a file"
    # Create a new file at first
    try:
        fnObj = open(fn, "w")
        printTree(dTree, 0, fnObj)
    finally:
        fnObj.close()

def printTree(dTree, depth, fnObj=None):
    "Print the decision tree recursively"
    root = dTree
    # NOTE: Leaf node does not have leftNode or rightNode
    # A non-leaf node does have both leftNode and rightNode!
    if root.isLeafNode() == False:
        # Left branch
        if root.leftNode.isLeafNode() == True:
            output = '| '*depth + root.name + ' = 0 : ' + root.leftNode.target
            if fnObj == None:
                print(output)
            else:
                fnObj.write(output+'\n')
        else:
            output = '| '*depth + root.name + ' = 0 :'
            if fnObj == None:
                print(output)
            else:
                fnObj.write(output+'\n')
            # Recursive calling
            printTree(root.leftNode, (depth+1), fnObj)

        # Right branch
        if root.rightNode.isLeafNode() == True:
            output = '| '*depth + root.name + ' = 1 : ' + root.rightNode.target
            if fnObj == None:
                print(output)
            else:
                fnObj.write(output+'\n')
        else:
            output = '| '*depth + root.name + ' = 1 :'
            if fnObj == None:
                print(output)
            else:
                fnObj.write(output+'\n')
            # Recursive calling
            printTree(root.rightNode, (depth+1), fnObj)

def printTree2(dTree, depth):
    "Print the decision tree recursively"
    print('-'*depth + ': ' + str(dTree))
    if dTree.leftNode != None:
        printTree2(dTree.leftNode, (depth+1))
    if dTree.rightNode != None:
        printTree2(dTree.rightNode, (depth+1))

def main():
    '''
    Main function to run id3 with binary data set
    ./id3 <train> <test> <model>
    '''

    # Process parameters
    if len(sys.argv) != 4:
        print('Error: invalid number of parameters')
        return(1)
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    modelFile = sys.argv[3]
    # Debug
    print(trainFile, testFile, modelFile)

    # Handle the training data
    attrList, dataList, targetList = loadCsvData(trainFile)

    # Construct the attrIndexLeft
    attrIndexLeft = []
    for i in range(len(attrList)):
        attrIndexLeft.append(i)

    # Construct the GiniList
    GiniList = []

    # Get the root node at first
    dTree = id3(dataList,
                targetList,
                attrList,
                attrIndexLeft,
                GiniList,
                True)

    # Print the tree
    printTree2(dTree, 0)
    printTree(dTree,0)

    # Write the tree into model file
    treeWriter(dTree, modelFile)
    print(modelFile + ' file is written!')

    # Handle the testing data
    attrList, dataList, targetList = loadCsvData(testFile)

    # Predict the result
    predList = predict(dataList, dTree)

    # Compute the accuracy
    errorNum = 0
    numOfTest = len(targetList)
    numOfPred = len(predList)
    # Debug
    print('numOfTest=%d, numOfPred=%d' %(numOfTest, numOfPred))
    for i in range(numOfTest):
        if predList[i] != targetList[i]:
            errorNum += 1
    accuracy = float(numOfTest - errorNum) / numOfTest
    print('Accuracy: ', accuracy)


if __name__ == '__main__':
    main()
