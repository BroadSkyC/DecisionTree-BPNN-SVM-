from sklearn import datasets

stack=[]#储存决策树所有的内部节点
allSubTrees=[]#储存所有剪枝过程中产生的子树

#决策树节点类
class dNode:
    def __init__(self,col=-1,value=None,results=None,leftNode=None,rightNode=None):
        self.col=col#节点划分的特征列
        self.value=value#节点划分的依据值
        self.results=results#叶节点所代表的类
        self.leftNode=leftNode#左子节点
        self.rightNode=rightNode#右子节点

#合并字典
def sumDic(a,b):
    c={}
    for key in a.keys():
        if key not in c.keys():
            c[key]=a[key]
        else:
            c[key]+=a[key]
    for key in b.keys():
        if key not in c.keys():
            c[key]=b[key]
        else:
            c[key]+=b[key]
    return c

#按7：3的比例将原始数据集划分为训练集和测试集
def chooseDatas(dataset,ratio):
    dic=combineSameValue(dataset)
    typeNum=len(dic)
    train=[]
    test=[]
    for e in dic.keys():
        l=[]
        for a in dataset:
            if a[len(a)-1]==e:
                l.append(a)
        for i in range(int(len(l)*ratio)):
            train.append(l[i])
        for i in range(int(len(l)*ratio),len(l)):
            test.append(l[i])
    return train,test

#统计样本中每个类的数量，用于计算数据集的基尼指数
def combineSameValue(items):
    res={}
    for item in items:
        e=item[len(item)-1]
        if e not in res:
            res[e]=0
        res[e]+=1
    return res

#分割数据集
def splitSet(dataset,value,column):
    leftSet=[]
    rightSet=[]
    #若列数据为数值型
    if isinstance(value,int) or isinstance(value,float):
        for item in dataset:
            # 如果某一行指定列值>=value，则将该行数据保存在leftList中，否则保存在rightList中
            if item[column]>=value:
                leftSet.append(item)
            else:
                rightSet.append(item)
    #若列数据为标称型
    else:
        for item in dataset:
            # 如果某一行指定列值=value，则将该行数据保存在leftList中，否则保存在rightList中
            if item[column]==value:
                leftSet.append(item)
            else:
                rightSet.append(item)
    return leftSet,rightSet

#基尼指数
def getGini(lists):
    length=len(lists)#数据所有行
    valueDic=combineSameValue(lists)
    gini=1.0
    for e in valueDic.keys():
        p=float(valueDic[e])/length
        gini-=p*p
    return gini

#构建决策树，采用CART算法
def buildTree(rows,scoref=getGini):
    # 若总基尼指数==0,则表示数据集中全是同一个类，作为叶节点返回
    if getGini(rows) == 0.0: return dNode(results=combineSameValue(rows))
    rowLength=len(rows)#计算数据集总行数
    columnCount = len(rows[0])#计算数据集总列数
    #记录最佳拆分的变量
    bestGini=float('inf')#初始化最小基尼指数
    bestCriteria=None#基尼指数最小时的列索引，以及划分数据集的样本值
    bestSets=None#基尼指数最小时，经过样本值划分后的数据子集

    #遍历除最后一列即标签列的所有数据,寻找最优即基尼指数最小的拆分
    for col in range(columnCount-1):
        colset=[list[col] for list in rows]#获取指定列的所有数据
        uniqueList=[]
        for e in colset:
            if e not in uniqueList:
                uniqueList.append(e)
        #若列数据为连续数据
        if isinstance(rows[0][col],int) or isinstance(rows[0][col],float):
            uniqueList.sort()
            for i in range(len(uniqueList) - 1):
                value = (uniqueList[i] + uniqueList[i + 1]) / 2  # 获取可能的离散分类点
                lSet, rSet = splitSet(rows, value, col)  # 分割数据集
                p = len(lSet) / rowLength  # 计算左数据集在总数据集中的概率
                giniSplit=p*getGini(lSet)+(1-p)*getGini(rSet)#获取特征条件下的基尼指数
                if giniSplit<bestGini:
                    bestGini=giniSplit
                    bestCriteria=(col,value)
                    bestSets=(lSet,rSet)
        #若列数据为离散数据
        else:
            for e in uniqueList:
                lSet,rSet=splitSet(rows,e,col)
                p = len(lSet) / rowLength  # 计算左数据集在总数据集中的概率
                giniSplit = p * getGini(lSet) + (1 - p) * getGini(rSet)#获取特征条件下的基尼指数
                if giniSplit<bestGini:
                    bestGini=giniSplit
                    bestCriteria=(col,e)
                    bestSets=(lSet,rSet)
    # 创建子分支
    if len(bestSets[0])>1:
        leftBranch=buildTree(bestSets[0],scoref)
    else:
        leftBranch=dNode(results=combineSameValue(bestSets[0]))
    if len(bestSets[1])>1:
        rightBranch=buildTree(bestSets[1],scoref)
    else:
        rightBranch=dNode(results=combineSameValue(bestSets[1]))
    return dNode(bestCriteria[0],bestCriteria[1],None,leftBranch,rightBranch)

#找到给定节点包含的类情况
def getEachTypeNum(node):
    if node.results!=None:return node.results
    return sumDic(getEachTypeNum(node.leftNode),getEachTypeNum(node.rightNode))

#获取树的所有节点
def getAllNodes(node):
    global stack
    if node.results==None:
        stack.append(node)
        getAllNodes(node.leftNode)
        getAllNodes(node.rightNode)

#获取所有叶结点的剪枝前误差
def getLeafMis(node,total):
    if node.results!=None:
        if len(node.results)==1:
            return [0.0,1]
        else:
            maxKey=None
            maxValue=0
            thisRes=0.0
            for key in node.results.keys():
                if node.results[key]>maxValue:
                    maxValue=node.results[key]
                    maxKey=key
            for key in node.results.keys():
                if key!=maxKey:
                    thisRes+=node.results[key]/total
            return thisRes
    else:
        return [getLeafMis(node.leftNode,total)[0]+getLeafMis(node.rightNode,total)[0],
                getLeafMis(node.leftNode,total)[1]+getLeafMis(node.rightNode,total)[1]]

#复制剪枝后的子树
def copyTree(root):
    root1 = dNode(col=root.col, value=root.value, results=root.results, leftNode=None,rightNode=None)
    if root.results==None:
        root1.leftNode=copyTree(root.leftNode)
        root1.rightNode=copyTree(root.rightNode)
    return root1

#采用CCP（代价复杂剪枝法）对CART决策树进行剪枝
def cut(tree):
    #若只剩树根则返回
    if tree.results!=None:
        allSubTrees.append(tree)
        return
    #计算数中样本总数
    total=0
    for value in getEachTypeNum(tree):
        total+=value
    #获取所有节点
    global stack
    stack=[]
    getAllNodes(tree)
    #设置参数
    miniA=float('inf')
    bestNode=None
    maxType = None
    maxValue=0
    #计算所有误差增加率，选取剪枝节点
    for node in stack:
        #找出该节点占比最大的类，计算剪枝后误差
        eachTypeNum=getEachTypeNum(node)
        for key in eachTypeNum.keys():
            if eachTypeNum[key]>maxValue:
                maxValue=eachTypeNum[key]
                maxType=key
        afterMis=0.0
        for key in eachTypeNum.keys():
            if key!=maxType:
                afterMis+=eachTypeNum[key]/total
        #计算剪枝前误差
        leafInfo=getLeafMis(node,total)
        beforeMis=leafInfo[0]
        leafNum=leafInfo[1]
        #计算整体损失
        loss=(afterMis-beforeMis)/(leafNum-1)
        #获取最小损失节点
        if miniA>loss:
            bestNode=node
            miniA=loss
    #对最小损失节点进行处理，合并左右子树，选取占比最大类作为新的分类，成为叶子节点
    bestNode.leftNode=None
    bestNode.rightNode=None
    bestNode.results={maxType:maxValue}
    #增添子树
    allSubTrees.append(copyTree(tree))
    #递归找出所有子树
    cut(tree)

#交叉验证，筛选出测试正确率最高的的剪枝后的子树
def chooseBestSubTree(testDataSet):
    global allSubTrees
    rightRate=0.0
    index=0
    for i in range(len(allSubTrees)):
        thisRightRate=test(testDataSet,allSubTrees[i])
        if rightRate<thisRightRate:
            rightRate=thisRightRate
            index=i
    return allSubTrees[index]

#测试集数据分类
def classify(data,tree):
    if tree.results!=None:return tree.results
    col=data[tree.col]
    branch=tree.rightNode
    if (isinstance(col,int) or isinstance(col,float)) and col>=tree.value:
        branch=tree.leftNode
    else:
        if col==tree.value: branch=tree.leftNode
    return classify(data,branch)

#对测试集进行分类，评估模型分类准确率
def test(dataset,tree):
    totalLen=len(dataset)
    rightAnswer=0
    for l in dataset:
        dic=classify(l,tree)
        if l[len(l)-1] in dic:
            rightAnswer+=1
    return rightAnswer/totalLen*100

#打印决策树
def printTree(tree,kong):
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.col)+":"+str(tree.value))
        print(kong+"Left->",end='')
        printTree(tree.leftNode," "+kong)
        print(kong+"Right->",end='')
        printTree(tree.rightNode," "+kong)

#主函数
def main(dataset,ratio):
    trainDataSet,testDataSet=chooseDatas(dataset,ratio)
    root=buildTree(trainDataSet)
    #printTree(root,"")
    t=test(testDataSet,root)
    print("剪枝前分类正确率为:"+str(t)+"%",end="\n\n")
    #对决策树进行剪枝
    global allSubTrees
    allSubTrees.append(copyTree(root))
    cut(root)
    #交叉验证找出最佳的子树
    bestTree=chooseBestSubTree(testDataSet)
    #printTree(bestTree,"")  #打印剪枝后最佳子树
    t = test(testDataSet, bestTree)
    print("剪枝后分类正确率为:" + str(t) + "%")

#获取Iris鸢尾花数据集
iris = datasets.load_iris()
features = iris.data  # 获取特征数据集
labels = iris.target  # 获取标签即类
dataset = []
# 合并特征数据集和标签数据集
for i in range(len(features)):
    l = []
    for e in features[i]:
        l.append(e)
    l.append(labels[i])
    dataset.append(l)
main(dataset,0.7)