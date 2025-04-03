#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


# # 自定义卷积核(特征提取)

# In[2]:


'''
过去d天X值构成的时序数列和Y值构成的时序数列的相关系数
'''

class ts_corr(nn.Module):
    def __init__(self, d=10, stride=10):
        super(ts_corr, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X): #X:3D
        B, n, T = X.shape # 批量大小，特征数量，时间窗口长度
        
        w = (T - self.d) // self.stride + 1  # 窗口数量,例如T=10，d=3，stride=2时，w=4
        h = n * (n - 1) // 2  # 特征对的数量 C(n, 2)

        # 使用 unfold 提取滑动窗口 [形状: (B, n, w, d)]
        unfolded_X = X.unfold(2, self.d, self.stride)
        
        #生成C(n,2)组合数
        #例如当n=3时，rows = tensor([0,0,1]), cols = tensor([1,2,2])
        rows, cols = torch.triu_indices(n, n, offset=1)

        # 提取特征对数据得到x和y [形状: (B, h, w, d)]，分别对应batch维度，特征维度，窗口维度，时间维度
        #x为([[:,0,:,:],[:,0,:,:],[:,1,:,:])
        #y为([[:,1,:,:],[:,2,:,:],[:,2,:,:])
        
        x = unfolded_X[:, rows, :, :]
        y = unfolded_X[:, cols, :, :]
        
        x_mean = torch.mean(x, dim=3, keepdim=True) #keepdim维持原本维度不变,在维度3做mean
        y_mean = torch.mean(y, dim=3, keepdim=True)
        
        cov = ((x-x_mean)*(y-y_mean)).sum(dim=3) #(B, h, w)
        corr = cov / (torch.std(x, dim=3) * torch.std(y, dim=3)+ 1e-8) #分母添加极小值防止除零错误

        return corr


# In[3]:


'''
过去d天X值构成的时序数列和Y值构成的时序数列的协方差
'''

class ts_cov(nn.Module):
    def __init__(self, d=10, stride=10):
        super(ts_cov, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        B, n, T = X.shape 
        
        w = (T - self.d) // self.stride + 1  
        h = n * (n - 1) // 2  

        unfolded_X = X.unfold(2, self.d, self.stride)
        
        rows, cols = torch.triu_indices(n, n, offset=1)

        x = unfolded_X[:, rows, :, :]
        y = unfolded_X[:, cols, :, :]
        
        x_mean = torch.mean(x, dim=3, keepdim=True)
        y_mean = torch.mean(y, dim=3, keepdim=True)
        
        cov = ((x-x_mean)*(y-y_mean)).sum(dim=3) 

        return cov


# In[4]:


'''
过去d天X值构成的时序数列的标准差
'''

class ts_stddev(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_stddev,self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        #input:(B,n,T)，在T维度用unfold展开窗口，变为(B,n,w,d),w为窗口数量会自动计算
        unfolded_X = X.unfold(2, self.d, self.stride)
        #在每个窗口，即d维度上进行std计算
        std = torch.std(unfolded_X, dim=3) #输出形状为(B,n,w)
        
        return std


# In[5]:


'''
过去d天X值构成的时序数列的平均值除以标准差
'''

class ts_zscore(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_zscore,self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        
        unfolded_X = X.unfold(2, self.d, self.stride)
        
        mean = torch.mean(unfolded_X, dim=3)
        std = torch.std(unfolded_X, dim=3)
        zscore = mean / (std + 1e-8)
        
        return zscore


# In[6]:


'''
研报原话为：
(X - delay(X, d))/delay(X, d)-1, delay(X, d)为 X 在 d 天前的取值
这里可能有误，return为“收益率“，应该是误加了-1
为了保持代码一致性，这里计算的是(X - delay(X, d-1))/delay(X, d-1),  delay(X, d-1)为 X 在 d-1 天前的取值
在构造卷积核的逻辑上是相似的
'''

class ts_return(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_return,self).__init__()
        self.d = d
        self.stride = stride
        
    def forward(self, X):
        
        unfolded_X = X.unfold(2, self.d, self.stride)
        return1 = unfolded_X[:,:,:,-1] /(unfolded_X[:,:,:,0] + 1e-8) - 1
        
        return return1


# In[7]:


'''
过去d天X值构成的时序数列的加权平均值，权数为d, d – 1, …, 1(权数之和应为1，需进行归一化处理）
其中离现在越近的日子权数越大。 
'''

class ts_decaylinear(nn.Module):
    
    def __init__(self, d = 10, stride = 10):
        super(ts_decaylinear,self).__init__()
        self.d = d
        self.stride = stride
        #如下设计的权重系数满足离现在越近的日子权重越大
        weights = torch.arange(d, 0, -1, dtype = torch.float32) 
        weights = weights / weights.sum()
        #注册权重，不用在前向传播函数中重复计算
        #注册了一个形状为(d,)的一维张量，存放权重系数，以便在forward函数中使用
        self.register_buffer('weights', weights) 
        
    def forward(self, X):
        
        unfolded_X = X.unfold(2, self.d, self.stride)
        #view将一维张量广播为4D张量，并在时间维度上，将weights与unfoled_X相乘
        decaylinear = torch.sum(unfolded_X * self.weights.view(1,1,1,-1), dim=-1)
        
        return decaylinear


# # 神经网络结构设计 

# 原始路径(RawPath)：特征提取层→BN
# 
# 池化路径(PoolPath)：特征提取层→池化层→BN
# 
# 展平→全连接层→预测目标

# In[8]:


'''
原始路径：特征提取层+BN
'''
class RawPath(nn.Module):
    def __init__(self, extractor, bn_dim): #传入参数：卷积核，特征维度
        super().__init__()
        self.extractor = extractor
        self.bn = nn.BatchNorm1d(bn_dim)
        
    def forward(self, X):
        x = self.extractor(X) #extract
        x = self.bn(x) #BN

        return x


# In[9]:


'''
池化路径：特征提取层+池化层+BN
'''
class PoolPath(nn.Module): #传入参数：卷积核，特征维度
    def __init__(self, extractor, bn_dim, d_pool=3, s_pool=3):
        super().__init__()
        self.extractor = extractor
        
        self.avg_pool = nn.AvgPool1d(d_pool, s_pool)
        self.max_pool = nn.MaxPool1d(d_pool, s_pool)
        
        #每个池化操作bihv使用独立的 BatchNorm 层
        #否则会导致三种不同统计量（均值、最大值、最小值）的分布被强制归一化到同一参数
        self.bn_avg = nn.BatchNorm1d(bn_dim) 
        self.bn_max = nn.BatchNorm1d(bn_dim)
        self.bn_min = nn.BatchNorm1d(bn_dim)

    def forward(self, X):
        x = self.extractor(X)
        
        x_avg = self.bn_avg(self.avg_pool(x))
        x_max = self.bn_max(self.max_pool(x))
        x_min = self.bn_min(-self.max_pool(-x))#手动取反实现min_pool
        
        return torch.cat([x_avg, x_max, x_min], dim = 1) #在特征维度进行拼接


# In[10]:


class AlphaNet(nn.Module):

    def __init__(self, d=10, stride=10, d_pool=3, s_pool=3, n=9, T=30): #池化层窗口d=3，步长stride=3
        super(AlphaNet, self).__init__()
        
        self.d = d 
        self.stride = stride 
        h = n * (n - 1) // 2 #手动计算cov和corr特征提取后的特征维度大小
        w = (T - d) // stride + 1 #手动计算特征提取层窗口数
        w_pool = (w - d_pool) // s_pool + 1 #手动计算池化层窗口数
        
        #特征提取层列表，共7个
        self.extractors = nn.ModuleList([
            ts_corr(d,stride),
            ts_cov(d,stride),
            ts_stddev(d,stride),
            ts_zscore(d,stride),
            ts_return(d,stride),
            ts_decaylinear(d,stride),
            nn.AvgPool1d(d,stride) #原研报中的ts_mean
        ])
        

        # 初始化双路径
        self.raw_paths = nn.ModuleList()
        self.pool_paths = nn.ModuleList()
        
        # 前两个特征提取器使用h维BN
        for i in range(2):
            self.raw_paths.append(RawPath(self.extractors[i], h))
            self.pool_paths.append(PoolPath(self.extractors[i], h))
        
        # 后五个特征提取器使用n维BN
        for i in range(2, 7):
            self.raw_paths.append(RawPath(self.extractors[i], n))
            self.pool_paths.append(PoolPath(self.extractors[i], n))

        raw_dim = (2*h + 5*n)*w #计算初始路径展平后的维度
        pooled_dim = (h*2*3 + n*5*3)*w_pool #计算池化路径展平后的维度

        self.head = nn.Sequential(
            nn.Linear(raw_dim + pooled_dim, 30),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(30, 1)
        )

        # 初始化
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                
                #截断正态初始化
                fan_in = m.weight.size(1)
                std = math.sqrt(1.0 / fan_in)  # Xavier方差标准
                nn.init.trunc_normal_(m.weight, std=std, a=-2*std, b=2*std)
                nn.init.normal_(m.bias, std=1e-6)
                
                #Kaiming初始化
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                #Xavier初始化
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.normal_(m.bias, std=1e-6)
            
    def forward(self, X):
        
        raw_features = [path(X).flatten(1) for path in self.raw_paths] #原始路径得到的张量进行展平
        pool_features = [path(X).flatten(1) for path in self.pool_paths] #池化路径得到的张量进行展平
        all_features = torch.cat(raw_features + pool_features, dim=1) 
        
        return self.head(all_features)



