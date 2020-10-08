import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#定义数据结构体，用于缓存，提高运行速度
class optStruct:
    def __init__(self, dataSet, labelSet, C, toler, kTup):
        self.dataMat = np.mat(dataSet) #原始数据，转换成m*n矩阵
        self.labelMat = np.mat(labelSet).T #标签数据 m*1矩阵
        self.C = C #惩罚参数
        self.toler = toler #容忍度
        self.m = np.shape(self.dataMat)[0] #原始数据行长度
        self.alphas = np.mat(np.zeros((self.m,1))) # alpha系数，m*1矩阵
        self.b = 0 #偏置
        self.eCache = np.mat(np.zeros((self.m,2))) # 保存原始数据每行的预测值
        self.K = np.mat(np.zeros((self.m,self.m))) # 核转换矩阵 m*m
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.dataMat, self.dataMat[i,:], kTup)

#SMO算法
def smoP(dataSet, labelSet, C, toler, maxInter, kTup = ('lin', 0)):
    #初始化结构体类，获取实例
    oS = optStruct(dataSet, labelSet, C, toler, kTup)
    iter = 0
    #全量遍历标志
    entireSet = True
    #alpha对是否优化标志
    alphaPairsChanged = 0
    #外循环 终止条件：1.达到最大次数 或者 2.alpha对没有优化
    while (iter < maxInter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #全量遍历 ，遍历每一行数据 alpha对有修改，alphaPairsChanged累加
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        else:
            #获取(0，C)范围内数据索引列表，也就是只遍历属于支持向量的数据
            nonBounds = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBounds:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        #全量遍历->支持向量遍历
        if entireSet:
            entireSet = False
        #支持向量遍历->全量遍历
        elif alphaPairsChanged == 0:
            entireSet = True
    return oS.b,oS.alphas

#计算原始数据第k项对应的预测误差  1*m m*1 =>1*1
def calEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#核转换函数rbf,用于低维空间映射到高维空间
def kernelTrans(dataMat, rowDataMat, kTup):
    m,n=np.shape(dataMat)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin':  # 线性核
        K = dataMat * rowDataMat.T
    elif kTup[0] == 'rbf':  # 非线性核
        for j in range(m):
            deltaRow = dataMat[j, :] - rowDataMat
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-2 * kTup[1] ** 2))
    return K

#第一次通过selectJrand()随机选取j,之后选取与i对应预测误差最大的j（步长最大）
def selectJ(i, oS, Ei):
    #初始化
    maxK = -1  #误差最大时对应索引
    maxDeltaE = 0 #最大误差
    Ej = 0 # j索引对应预测误差
    oS.eCache[i] = [1,Ei]
    #获取数据缓存结构中非0的索引列表(先将矩阵第0列转化为数组)
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    #遍历索引列表，寻找最大误差对应索引
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calEk(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j=i
        while (j == i):
            j = int(np.random.uniform(0, oS.m))
        Ej = calEk(oS, j)
    return j,Ej

#alpha范围剪辑
def clipAlpha(aj, L, H):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

#计算 w 权重系数
def calWs(alphas, dataSet, labelSet):
    dataMat = np.mat(dataSet)
    #1*100 => 100*1
    labelMat = np.mat(labelSet).T
    m, n = np.shape(dataMat)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], dataMat[i,:].T)
    return w

#计算原始数据每一行alpha,b，保存到数据结构中，有变化及时更新
def innerL(i, oS):
    #计算预测误差
    Ei = calEk(oS, i)
    #选择第一个alpha，违背KKT条件2
    if ((oS.labelMat[i] * Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.toler) and (oS.alphas[i] > 0)):
        #第一次随机选取不等于i的数据项，其后根据误差最大选取数据项
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        #    0 <= a1,a2 <= C 求出L,H
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H :
            return 0
        #内核分母
        eta = oS.K[i, i] + oS.K[j, j] - 2.0*oS.K[i, j]
        if eta <= 0:
            return 0
        #计算第一个alpha j
        oS.alphas[j] += oS.labelMat[j]*(Ei - Ej)/eta
        #修正alpha j的范围
        oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
        #alpha有改变，就需要更新缓存数据
        updateEk(oS, j)
        #如果优化后的alpha 与之前的alpha变化很小，则舍弃，并重新选择数据项的alpha
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            return 0
        #计算alpha对的另一个alpha i
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        #alpha有改变，就需要更新缓存数据
        updateEk(oS, i)
        #计算b1,b2
        bi = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[i,j]
        bj = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[j,j]
        #首选alpha i
        if (0 < oS.alphas[i]) and (oS.alphas[i] < oS.C):
            oS.b = bi
        elif (0 < oS.alphas[j]) and (oS.alphas[j] < oS.C):
            oS.b = bj
        else:
            oS.b = (bi + bj)/2.0
        return 1
    else:
        return 0

#alpha有改变都要更新
def updateEk(oS, k):
    Ek = calEk(oS, k)
    oS.eCache[k] = [1, Ek]

#训练结果
def getDataResult(dataSet, labelSet, b, alphas, k1=1.3):
    datMat = np.mat(dataSet)
    labelMat = np.mat(labelSet).T
    #alphas.A>0 获取大于0的索引列表，只有>0的alpha才对分类起作用
    svInd=[]
    for i in range(len(alphas.A)):
        if alphas.A[i]>0 and i<len(datMat):
            svInd.append(i)
    sVs=datMat[svInd]
    labelSV = labelMat[svInd]
    m,n = np.shape(datMat)
    rightCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict)==np.sign(labelSet[i]): rightCount += 1
    return str(float(rightCount)/m*100)+"%"

if __name__ == '__main__':
    # 获取Iris鸢尾花数据集
    iris = datasets.load_iris()
    features = iris.data  # 获取特征数据集
    labels = iris.target  # 获取标签即类
    featuresTrain,featuresTest,LabelsTrain,LabelsTest=train_test_split(features,labels,test_size=0.3)
    #尝试手写SVM代码测试，对SVM的理解还存在一些问题，代码存在一定问题。
    b,alphas=smoP(featuresTrain,LabelsTrain,1.0,0.1,1000,('rbf',0.1))
    print("手写SVM测试")
    print("训练集正确率为："+getDataResult(featuresTrain,LabelsTrain,b,alphas,0.1))
    print("测试集正确率为："+getDataResult(featuresTest,LabelsTest,b,alphas,0.1))
    #再调用库看看情况
    print("调用库中的SVM")
    model = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=0.1,
              kernel='linear', max_iter=-1, probability=False, random_state=None,
              shrinking=True, tol=0.001, verbose=False)
    #训练，计算训练集分类正确率
    model.fit(featuresTrain,LabelsTrain)
    rightRate=model.score(featuresTrain,LabelsTrain)
    print("训练集正确率为："+str(rightRate*100)+"%")
    #测试，计算测试集分类正确率
    rightRate=model.score(featuresTest,LabelsTest)
    print("分类正确率为"+str(rightRate*100)+"%")
