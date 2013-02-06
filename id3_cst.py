#-------------------------------------------------------------------------------
# Name:        id3_cst.py
# Purpose:     Implementation of ID3 algorithm of Decision Tree
#              for binary data with information gain for data split
#              and Chi-Square Test for branching stop
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

def computeChiSquareProb(dataList, targetList, attrIndex):
    "Compute the Chi-square probability"
    funcName = 'computeChiSquareProb'
    # Reference: http://www.dmi.unict.it/~apulvirenti/agd/Qui86.pdf
    # X2 = Sigma(v(a))[(pi-p'i)2/p'i+(ni-n'i)2/n'i
    # p'i = p*(pi+ni)/(p+n)
    # n'i = n*(pi+ni)/(p+n)
    numOfBin0 = 0
    numOfBin1 = 0
    numOfData = len(dataList)
    numOfBin0, numOfBin1 = getTargetCount(targetList)
    # p = numOfBin1, n = numOfBin0

    # C(T|Xa=v) - C(T|Xa=0) and C(T|Xa=1)
    numOfBin0Xa0 = 0
    numOfBin1Xa0 = 0
    numOfBin0Xa1 = 0
    numOfBin1Xa1 = 0
    for i in range(numOfData):
        if dataList[i][attrIndex] == '0':
            if targetList[i] == '0':
                numOfBin0Xa0 += 1
            elif targetList[i] == '1':
                numOfBin1Xa0 += 1
            else:
                print(funcName + 'C(T|Xa=0)' +
                    'Error: only binary attribute/classification is supported')
        elif dataList[i][attrIndex] == '1':
            if targetList[i] == '0':
                numOfBin0Xa1 += 1
            elif targetList[i] == '1':
                numOfBin1Xa1 += 1
            else:
                print(funcName + 'C(T|Xa=1)' +
                    'Error: only binary attribute/classification is supported')
    # p0 = numOfBin1Xa0, n0 = numOfBin0Xa0
    # p1 = numOfBin1Xa1, n1 = numOfBin0Xa1

    # Compute expected p/n - p'i, n'i
    expP0 = float(numOfBin1*(numOfBin0Xa0+numOfBin1Xa0))/numOfData
    expN0 = float(numOfBin0*(numOfBin0Xa0+numOfBin1Xa0))/numOfData
    expP1 = float(numOfBin1*(numOfBin0Xa1+numOfBin1Xa1))/numOfData
    expN1 = float(numOfBin0*(numOfBin0Xa1+numOfBin1Xa1))/numOfData

    # Defensive checking
    if expN0 == 0 or expN1 == 0 or expP0 == 0 or expP1 == 0:
        # We do not computer csp for this
        return(-1)

    # Compute the Chi-square prob
    csp = (numOfBin0Xa0-expN0)*(numOfBin0Xa0-expN0)/expN0 + \
            (numOfBin1Xa0-expP0)*(numOfBin1Xa0-expP0)/expP0 + \
            (numOfBin0Xa1-expN1)*(numOfBin0Xa1-expN1)/expN1 + \
            (numOfBin1Xa1-expP1)*(numOfBin1Xa1-expP1)/expP1

    return(csp)

def computeInfoGain(dataList, targetList, attrIndex):
    "Compute the information gain for certain attribute"
    funcName = 'computeInfoGain'
    # Always assume the binary attribute and classification
    # Reference: http://en.wikipedia.org/wiki/Information_gain_in_decision_trees
    # IG(T,a) = H(T) - Sigma(v(a))[P(Xa=v)*H(T|Xa=v)]

    # H(T)
    numOfBin0 = 0
    numOfBin1 = 0
    numOfData = len(dataList)
    numOfBin0, numOfBin1 = getTargetCount(targetList)
    prob0 = float(numOfBin0) / numOfData
    prob1 = float(numOfBin1) / numOfData
    ht = (-1)*(prob0*log2Func(prob0) + prob1*log2Func(prob1))

    # H(T|Xa=v) - H(T|Xa=0) and H(T|Xa=1)
    numOfBin0Xa0 = 0
    numOfBin1Xa0 = 0
    numOfBin0Xa1 = 0
    numOfBin1Xa1 = 0
    for i in range(numOfData):
        if dataList[i][attrIndex] == '0':
            if targetList[i] == '0':
                numOfBin0Xa0 += 1
            elif targetList[i] == '1':
                numOfBin1Xa0 += 1
            else:
                print(funcName + 'H(T|Xa=0)' +
                    'Error: only binary attribute/classification is supported')
        elif dataList[i][attrIndex] == '1':
            if targetList[i] == '0':
                numOfBin0Xa1 += 1
            elif targetList[i] == '1':
                numOfBin1Xa1 += 1
            else:
                print(funcName + 'H(T|Xa=1)' +
                    'Error: only binary attribute/classification is supported')

    prob0Xa0 = 0
    prob1Xa0 = 0
    prob0Xa1 = 0
    prob1Xa1 = 0
    if (numOfBin0Xa0 + numOfBin1Xa0) != 0:
        prob0Xa0 = float(numOfBin0Xa0) / (numOfBin0Xa0 + numOfBin1Xa0)
        prob1Xa0 = float(numOfBin1Xa0) / (numOfBin0Xa0 + numOfBin1Xa0)
    if (numOfBin0Xa1 + numOfBin1Xa1) != 0:
        prob0Xa1 = float(numOfBin0Xa1) / (numOfBin0Xa1 + numOfBin1Xa1)
        prob1Xa1 = float(numOfBin1Xa1) / (numOfBin0Xa1 + numOfBin1Xa1)
    # Debug
    '''
    print('prob0Xa0:', prob0Xa0)
    print('prob1Xa0:', prob1Xa0)
    print('prob0Xa1:', prob0Xa1)
    print('prob1Xa1:', prob1Xa1)
    '''

    htXa0 = (-1)*(prob0Xa0*log2Func(prob0Xa0) + prob1Xa0*log2Func(prob1Xa0))
    htXa1 = (-1)*(prob0Xa1*log2Func(prob0Xa1) + prob1Xa1*log2Func(prob1Xa1))

    # Debug
    '''
    print('htXa0:', htXa0)
    print('htXa1:', htXa1)
    '''

    # Return the information gain for this attribute
    infoGain = ht - (prob0*htXa0 + prob1*htXa1)
    return(infoGain)

def id3(trainData, trainTarget, attrList, attrIndexLeft, infoGainList=None, debug=False):
    "Construct the tree structure using information gain"
    funcName = 'id3'

    # Minimal chi-square prob ~ p<0.01
    minChiSquareProb = 6.635

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
        maxIGcsProb = -1
        for i in attrIndexLeft:
            # Get the information gain for this attribute
            tmpIG = computeInfoGain(trainData, trainTarget, i)
            # Get the chi-square prob for this attribute
            csProb = computeChiSquareProb(trainData, trainTarget, i)
            if tmpIG > maxIG:
                # Update the max information gain and save the index
                # And chi-square prob
                maxIG = tmpIG
                maxIndex = i
                maxIGcsProb = csProb
            # Debug
            if debug == True:
                # Dump all the information gain at the root node
                # Dump all the chi-square prob at the root node
                infoGainList.append((attrList[i], tmpIG, csProb))

        if debug == True:
            for i in infoGainList:
                print(i)

        # Chi-Square test
        if maxIGcsProb < minChiSquareProb:
            # Stop branching! - Treat it like no attribute left
            # Mark this node as a leaf node with most common case
            root.name = '#'
            if mostCommonTarget == 0:
                root.target = '0'
            else:
                root.target = '1'
            return(root)

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

    # Construct the infoGainList
    infoGainList = []
    # Debug
    print(attrList, attrIndexLeft)

    # Get the root node at first
    dTree = id3(dataList,
                targetList,
                attrList,
                attrIndexLeft,
                infoGainList,
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
