# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator

# 用户字体路径/Library/Fonts
# 系统字体路径/System/Library/Fonts
fontPath = r"/System/Library/Fonts/Apple Braille.ttf"
font = FontProperties(fname=fontPath, size=14)


def createDataSet():
    # 贷款
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels


def createTree(dataset, labels, featLabels):
    # 最后一列为标签
    classList = [example[-1] for example in dataset]
    # 递归结束条件
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    # 选取最优列——增益最大、熵最小
    bestFeatIndex = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeatIndex]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}
    del labels[bestFeatIndex]
    # 最优列 样本值/"类型"
    featValue = [example2[bestFeatIndex] for example2 in dataset]
    uniqueVals = set(featValue)
    for value in uniqueVals:
        sublabels = labels[:]
        subDataSet = splitDataSet(dataset, bestFeatIndex, value)
        myTree[bestFeatLabel][value] = createTree(
            subDataSet, sublabels, featLabels)
    return myTree


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]


def chooseBestFeatureToSplit(dataset):
    # 特征数量
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0
    bestFeature = -1
    # 遍历特征样本 每个迭代一个列
    for i in range(numFeatures):
        # 每个特征列的值
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0
        # 每列值去重后的"类型"
        for val in uniqueVals:
            # 各类型的样本
            subDataSet = splitDataSet(dataset, i, val)
            # 各类型的概率
            prob = len(subDataSet)/float(len(dataset))
            # 每列的熵——各类型熵求和
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 每列的增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def splitDataSet(dataset, axis, val):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == val:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 香农熵
def calcShannonEnt(dataset):
    # 样本数量 15
    numexamples = len(dataset)
    # 标签数量 no:6 yes:9
    labelCounts = {}
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1

    shannonEnt = 0
    for key in labelCounts:
        prop = float(labelCounts[key])/numexamples
        shannonEnt -= prop*log(prop, 2)
    return shannonEnt


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args
                            # ,
                            # fontProperties=font
                            )


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center",
                        ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,
                     plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


if __name__ == '__main__':
    dataset, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataset, labels, featLabels)
    createPlot(myTree)
