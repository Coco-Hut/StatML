import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.datasets import load_boston

warnings.filterwarnings('ignore') 

'''
__init__函数数值:
Model_name 模型名称: 可以实现基本的线性回归、岭回归和Lasso回归
lr: 学习率
max_epoch: 最大的迭代次数
threshold: 残差收敛条件,相邻两次损失函数数值的差值小于tol则模型优化结束
Lambda: 惩罚项系数
coef_: 训练的参数
fit_intercept_表示是否有bias
verbose: 是否有输出信息提示

fit
X: ndarray [epoch_size,dim] # 二维
y: ndarray [epoch_size,] # 一维

predict:
x: 一维向量或者二维向量,满足特征为维度为dim

'''

'''
注: 针对波士顿数据集党lr取1e-5以及更大的值后预测结果可能会爆炸,
可能要考虑数据normalize,这里用的是梯度下降法,原sklearn用的是坐标下降,会有差异
'''

class LinearRegression:
    
    def __init__(self,Model_name='Ridge',lr=1e-6,
                 max_epoch=8000,
                 threshold=1e-5,
                 Lambda=0.001,
                 fit_intercept_=True,
                 verbose=False):
        
        self.name=Model_name
        self.lr=lr
        self.epoch=max_epoch
        self.threshhold=threshold
        self.Lambda=Lambda 
        self.verbose=verbose
        self.fit_intercept_=fit_intercept_
        self.w=None
        
    def fit(self,X,y):
        size=X.shape[0] # 样本个数
        
        if self.fit_intercept_:
            X=np.concatenate((np.ones((size,1)),X),axis=1) #bias 加到第0维,全1
            
        feat_dim=X.shape[1]
        '''
        权重先用w表示,生成w的方式可以更改为正态分布和均匀分布的随机数
        '''
        # w=np.mat(np.ones((feat_dim,1))) 
        w=np.mat(np.random.randn(feat_dim,1))
    
        X=np.mat(X)
        y=np.mat(y.reshape(-1,1))
        
        pre_loss=0 # 前一轮的损失函数
        
        for iter in range(self.epoch):
            # 求梯度和权重更新
            if self.name=='Ridge':
            # Ridge的梯度计算 
                grad=X.T*(X*w-y)/size+self.Lambda*2*w
            elif self.name=='Lasso':
            # Lasso的梯度计算
                grad=X.T*(X*w-y)/size+self.Lambda*np.sign(w)
            else:
            # 线性回归的梯度计算
                grad=X.T*(X*w-y)/size
                
            w=w-self.lr*grad
            # 当下更新后的损失值
            diff=X*w-y
            loss=np.power(diff.getA(),2).mean()
            
            # 残差在收敛阈值内
            if abs(pre_loss-loss)<self.threshhold:
                if self.verbose:
                    print(f'Early Stopping in epoch {iter}, Trainning loss is {loss}')
                break
            else:
                pre_loss=loss
            
            # 每100轮输出一次loss数值，这里采用的是均方误差的均值
            if self.verbose==True and (iter+1)%500==0:
                print(f'In epoch {iter+1}, Trainning loss is {loss}')
        
        self.w=w.getA().reshape(1,-1)[0] # 权重，可能包含bias
        if self.fit_intercept_:
            # 有截距的情况
            self.coef_,self.intercept_=self.w[1:],self.w[0]
        else:
            # 无截距的情况
            self.coef_=self.w
        
    def predict(self,x):
        if len(x.shape)==1:
            x=x.reshape(1,-1) # 保证输入是两维[size,feat_dim]
        
        if self.fit_intercept_:
            # 有截距在第一维度加1
            X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        
        y_pred=X.dot(self.w)
        return y_pred    

if __name__ =='__main__':
    data=load_boston()
    LR=LinearRegression(verbose=True)
    LR.fit(data.data,data.target)
    LR.predict(data.data)