# 划分数据集 信息增益 香农熵计算

from math import log
 
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  #
    labelCounts = {}
    # 以下五行为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]  #提取最后一项做为标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 书中有错
    # 0:{"yes":1} 1:{"yes":2}  2:{"no":1} 3:{"no":2} 4:{"no":3}
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算概率
        # 以2为底求对数
        shannonEnt -= prob * log(prob,2) # 递减求和得熵
    return shannonEnt
 
# 手动计算:  Ent = -0.4*log(2,0.4)-0.6*log(2,0.6)
# Ent_mannual = -(0.4 * log(0.4,2)) - (0.6 * log(0.6,2))
# print(Ent_mannual)
 
# 写一个数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels
 
def splitDataSet(dataSet,axis,value): # 三个输入参数：待划分的数据集、划分数据集的特征、需要返回的特征的值
    # 创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # dataSet[0]=0时执行以下操作
            # 以下三行抽取
            reducedFeatVec = featVec[:axis]   # featVec[:0]= [],即生成一个空列表
            reducedFeatVec.extend(featVec[axis + 1:]) # 添加index==1及后的元素 : 0/1/2 跳过,3:1,4:1
            retDataSet.append(reducedFeatVec) #整体作为元素添加 3:[[1,"no"]] , 4:[[1,"no"],[1,"no"]]
    return retDataSet
 
# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 去掉标签项
    baseEntropy = calcShannonEnt(dataSet) # 计算熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 以下两行创建唯一的分类标签列表
        featList = [example[i] for example in dataSet] # i=0:[1,1,1,0,0]  i=1:[1,1,0,1,1]
        uniqueVals = set(featList)  # i=0:{0,1}  i=1:{0,1}
        newEntropy = 0.0
        # 以下五行计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            print(subDataSet)
            # i=0:{(0,0),(0,1)} 返回:[[1, 'no'], [1, 'no']]      [[1,"yes"],[1,"yes"],[0,"no"]]
            # i=1:{(1,0),(1,1)} 返回:[[0,"no"]]       [[1,"yes"],[1,"yes"],[1,"no"],[1,"no"]]
            prob = len(subDataSet)/float(len(dataSet))
            # i=0:{(0,0),(0,1)} 返回:2/5 3/5
            # i=1:{(1,0),(1,1)} 返回:1/5 4/5
            newEntropy += prob * calcShannonEnt(subDataSet)  # 注意这里是subDataSet 不是 dataSet
        print("当i={}时得到的熵为".format(i),newEntropy)
        infoGain = baseEntropy - newEntropy # 信息增益
        if (infoGain > bestInfoGain):
            # 计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
 
if __name__ == "__main__":
    myDat,labels = createDataSet()
    a = splitDataSet(myDat,0,0)
    # print(a)
    b = chooseBestFeatureToSplit(myDat)
    print(b)
 
    # append()方法和extend()方法比较:
    # a = [1,2,3]
    # b = [4,5,6]
    # c = [7,8,9]
    # a.append(b)
    # print(a) # out:[1, 2, 3, [4, 5, 6]]
    # b.extend(c)
    # print(b)  # out:[4, 5, 6, 7, 8, 9]