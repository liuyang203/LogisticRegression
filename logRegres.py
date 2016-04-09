# -*- coding: utf-8 -*-
__author__ = 'Liu'

from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat    #dataMat m*3列表，矩阵  labelMat 1*m列表，行向量

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度下降
def gradDescent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)                 #转矩阵m*n
    labelMat = mat(classLabels).transpose()     #行转列向量m*1
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))                   #n*1列向量
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)     #h是m*1列向量
        error = (h - labelMat)              #error是m*1列向量
        weights = weights - alpha * dataMatrix.transpose() * error
    return weights

#随机梯度下降
def stocGradDescent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)                       #1*n数组
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))     #h是数值
        error = h - classLabels[i]                  #error是数值
        weights = weights - alpha * error * dataMatrix[i]
    return weights


def stocGradDescent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = h - classLabels[randIndex]
            weights = weights - alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights





def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
'''
dataMat, labelMat = loadDataSet()
weights = gradDescent(dataMat, labelMat)
weights0 = stocGradDescent0(array(dataMat), labelMat)
weights1 = stocGradDescent1(array(dataMat), labelMat)
#plotBestFit(weights.getA())
#plotBestFit(weights0)
#plotBestFit(weights1)
#print weights1
'''


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#打开测试集和训练集，对数据进行格式化处理的函数
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradDescent1(array(trainingSet),trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr, trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "The error rate of this test is :%f" % errorRate
    return errorRate
#调用colicTests()10次，并取平均
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is:%f" %(numTests, errorSum/float(numTests))




multiTest()




