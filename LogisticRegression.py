from LinearRegression import LinearRegression
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# 实现支持多分类的logistic回归,参数含义参考LinearRegression

class LogisticRegression:
    def __init__(self,learning_rate=1e-6,
                 max_epoch=8000,
                 fit_intercept_=True,
                 verbose=False):
        
        self.lr=learning_rate
        self.epochs=max_epoch
        self.fit_intercept_=fit_intercept_
        self.verbose=verbose
    
    def fit(self,X,y):
        size=X.shape[0] # 样本个数
        
        n_class=np.unique(y).shape[0] # 分类数目
        
        if self.fit_intercept_:
            X=np.concatenate((np.ones((size,1)),X),axis=1) #bias 加到第0维,全1
            
        feat_dim=X.shape[1]
        
        # 初始化权重矩阵，为一个[n_class,feat_dim]的矩阵
        self.weight=np.random.randn(n_class,feat_dim)
        # 独热编码，对标签进行编码
        enc=OneHotEncoder()
        y=enc.fit_transform(y.reshape(-1,1)).toarray()
        
        for iter in range(self.epochs):
            '''
            先计算一轮训练模型对每个样本在每个类别上的预测概率
            1. X*w(转置)--> [size,feat_dim] x [feat_dim,n_class]-->[size,n_class]
            2. 对得到的矩阵的每一行做softmax,得到概率数值,矩阵大小仍然是[size,n_class]
            '''
            probs=np.exp(X.dot(self.weight.T))/np.exp(X.dot(self.weight.T)).sum(axis=1).reshape(size,1) # 第1、2步
            
            '''
            3. 计算对每个wi更新的梯度,wi表示self.weight的每一行,即长为feat_dim的行向量,具体求法见文档
            '''
            
            grad_w=-(X.T).dot(1-probs) # --> [feat_dim,n_class]
            
            self.weight-=self.lr*grad_w.T
            
            # 每100轮输出一次loss数值，这里采用的是均方误差的均值
            if self.verbose==True and (iter+1)%100==0:
                
                # 计算交叉熵
                loss=(-1/size)*np.trace(np.log(probs.dot(y.T))) # 加上一个1e-6避免数值过小
                print(f'In epoch {iter+1}, Average Trainning loss is {loss}')
            
            # 得到的self.weight的第一维是bias 
        
    def predict(self,x):
        if len(x.shape)==1:
            x=x.reshape(1,-1) # 保证输入是两维[size,feat_dim]
        
        if self.fit_intercept_:
            # 有截距在第一维度加1
            X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        
        y_pred=np.exp(X.dot(self.weight.T))/np.exp(X.dot(self.weight.T)).sum(axis=1).reshape(x.shape[0],1)# 得到
        return y_pred  # 返回的是各个样本属于三类的概率
    

if __name__ =='__main__':
    data=load_iris()
    logistic=LogisticRegression(verbose=True)
    logistic.fit(data.data,data.target)
    logistic.predict(data.data)  