import math
from sklearn.model_selection import train_test_split
from sklearn import datasets
import random

class NeuralNetwork:
    #初始化类
    def __init__(self,learningRate,iNum,hNum,oNum,ihw=None,ihb=None,how=None,hob=None):
        self.learningRate=learningRate
        self.iNum=iNum
        self.hNum=hNum
        self.oNum=oNum
        self.ihb=ihb
        self.hob=hob
        self.ihw=ihw
        self.how=how
        self.initIHWB()
        self.initHOWB()
        self.hiddenLayer = NeuronLayer(hNum, self.ihb,self.ihw)
        self.OutLayer = NeuronLayer(oNum, self.hob,self.how)

    #初始化输入层到隐含层的权重
    def initIHWB(self):
        if self.ihb==None:
            self.ihb=random.random()
        if self.ihw==None:
            self.ihw=[]
            for i in range(self.hNum):
                IthW=[]
                for j in range(self.iNum):
                    IthW.append(random.random())
                self.ihw.append(IthW)

    #初始化隐含层到输出层的权重
    def initHOWB(self):
        if self.hob==None:
            self.hob=random.random()
        if self.how==None:
            self.how=[]
            for i in range(self.oNum):
                IthW=[]
                for j in range(self.hNum):
                    IthW.append(random.random())
                self.how.append(IthW)

    #向前传播
    def forward(self,data):
        hiddenLayerOutput=self.hiddenLayer.getOutputs(data,"sigmoid")
        return self.OutLayer.getOutputs(hiddenLayerOutput,"ReLu")

    #训练
    def train(self,inputs,target):
        self.forward(inputs)

        #获取输出层输出
        outputLayerOut=self.OutLayer.output
        #获取输出层单个误差
        outputLayerMis=[]
        for i in range(self.oNum):
            mis=(outputLayerOut[i]-target[i])*1
            outputLayerMis.append(mis)
        #更新隐含层到输出层的权重
        for i in range(self.oNum):
            for j in range(self.hNum):
                self.how[i][j]-=self.learningRate*outputLayerMis[i]*self.hiddenLayer.output[j]
                self.OutLayer.neurons[i].weights[j]=self.how[i][j]
        #更新隐含层到输出层的权重
        temp=[]
        for i in range(self.hNum):
            this=0.0
            for j in range(self.oNum):
                this+=outputLayerMis[j]*self.hiddenLayer.neurons[i].weights[j]
            temp.append(this)
        for i in range(self.hNum):
            for j in range(self.iNum):
                self.ihw[i][j]-=self.learningRate*inputs[j]*self.hiddenLayer.output[i]*(1-self.hiddenLayer.output[i])*temp[i]
                self.hiddenLayer.neurons[i].weights[j]=self.ihw[i][j]

class NeuronLayer:
    def __init__(self,NeuronNum,b,weights):
        self.b=b#该层截距项b
        self.NeuronNum=NeuronNum#该层神经元个数
        self.neurons=[]#该层神经元列表
        self.output=[]
        for i in range(NeuronNum):
            self.neurons.append(Neuron(b,weights[i]))

    def getOutputs(self,data,f):
        self.output=[]
        for i in range(self.NeuronNum):
            self.neurons[i].output(data,f)
            self.output.append(self.neurons[i].out)
        return self.output

class Neuron:
    def __init__(self,b,weights):
        self.net=0.0
        self.out=0.0
        self.b=b
        self.weights=weights

    #获取神经元输出
    def output(self,data,f):
        self.net=self.b
        for i in range(len(self.weights)):
            self.net+=(self.weights[i])*float(data[i])
        if f=="sigmoid": self.out=self.sigmoid(self.net)
        else:self.out=self.ReLu(self.net)
        return self.out

    #激活函数sigmoid
    def sigmoid(self,num):
        return 1/(1+math.exp(-num))

    #激活函数ReLu
    def ReLu(self,num):
        return max(0,num)

#合并特征即和标签集
def combineFeaturesAndLabels(features,labels):
    dataset=[]
    for i in range(len(features)):
        l=[]
        for e in features[i]:
            l.append(e)
        l.append(labels[i])
        dataset.append(l)
    return dataset

def main():
    iris = datasets.load_iris()
    features = iris.data  # 获取特征数据集
    labels = iris.target  # 获取标签即类
    #设置神经网络权重和偏差值
    ihw,ihb,how,hob=[[0.1,0.9,0.2,0.1],[0.7,0.5,0.1,0.6]],0.6,[[0.5,0.8]],0.7
    #随机权重和偏差
    #ihw,ihb,how,hob=None,None,None,None
    #构建神经网络
    nw = NeuralNetwork(0.3, features.shape[1],int(math.log2(features.shape[1])),1,ihw,ihb,how,hob)
    #每次随机生成训练集、测试集
    featuresTrain,featuresTest,LabelsTrain,LabelsTest=train_test_split(features,labels,test_size=0.3)
    for i in range(1000):
        for j in range(len(featuresTrain)):
            nw.train(featuresTrain[j],[LabelsTrain[j]])
    # print("测试集测试结果与数据集标签比较")
    # for i in range(len(featuresTest)):
    #     print(nw.forward(featuresTest[i])[0],LabelsTest[i])

    #计算训练后BPNN分类正确率
    rightNum=0
    for i in range(len(featuresTest)):
        res=nw.forward(featuresTest[i])[0]
        #print(res,LabelsTest[i],round(res))
        res=round(res)
        if res==LabelsTest[i]: rightNum+=1
    rightRate=rightNum/len(featuresTest)
    print("测试集分类正确率为："+str(rightRate*100)+"%")

main()