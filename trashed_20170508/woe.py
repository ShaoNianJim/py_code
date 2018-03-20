
#date:2016-05-01
#target:calcuate the WOE of features

import numpy as np
import math
import pandas as pd
#from pandas import Series
from scipy import stats
from sklearn.utils.multiclass import type_of_target
        
def discrete(x):
    #Discrete the input 1-D numpy array using 5 equal percentiles
    #:param x: 1-D numpy array
    #:return: discreted 1-D numpy array
   res = np.array([0] * (x.shape[-1]), dtype=int)
   #处理nan值
   if -99999 in x:
       res[x==-99999] = 0
       x_notnan = x[x!=-99999]
       for i in range(4):
           point1 = stats.scoreatpercentile(x_notnan,i*25)
           #print ('point1:',point1)
           point2 = stats.scoreatpercentile(x_notnan,(i+1)*25)
           #print ('point2:',point2)
           x1 = x[np.where((x>=point1) & (x<=point2))]
           mask = np.in1d(x,x1)
           res[mask] = i+1    
   else:
       for i in range(5):
           point1 = stats.scoreatpercentile(x,i*20)
           point2 = stats.scoreatpercentile(x,(i+1)*20)
           x1 = x[np.where((x>=point1) & (x<=point2))]
           mask = np.in1d(x,x1)
           res[mask] = i+1
   return res
   
def feature_discretion(X):
    #Discrete the continuous features of input data X, and keep other features unchanged.
    #:param X : numpy array
    #:return: the numpy array in which all continuous features are discreted
    temp=[]
    for i in range(X.shape[-1]):
        x = X[:,i]
        x_type = type_of_target(x)
        if x_type == 'continuous' or np.unique(x[-np.isnan(x)]).size>10:
            x = discrete(x)
            temp.append(x)
        else:
            temp.append(x)
    return np.array(temp).T

def check_target_binary(y):
    #check if the target variable is binary, raise error if not.
    #:param y:
    #:return:
    y_type = type_of_target(y)
    if y_type not in  ['binary']:
        raise ValueError('Label type must be binary')
        
def count_binary(a, event=1):
    event_count = a[a==event].sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count
    
    
def woe_single_x(x, y, event=1,adj=0.5):
#        calculate woe and information for a single feature
#        :param x: 1-D numpy starnds for single feature
#        :param y: 1-D numpy array target variable
#        :param event: value of binary stands for the event to predict
#        :return: dictionary contains woe values for categories of this feature
#                 information value of this feature
    check_target_binary(y)
    
    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    woe_dict = {}
    iv = 0
    for x1 in x_labels:
        y1 = y[np.where(x==x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = event_count*1.0 / event_total
        rate_non_event = non_event_count*1.0 / non_event_total
        if rate_non_event == 0:
            woe1 = math.log(((event_count*1.0+adj) / event_total)/((non_event_count*1.0+adj) / non_event_total))
        elif rate_event == 0:
             woe1 = math.log(((event_count*1.0+adj) / event_total)/((non_event_count*1.0+adj) / non_event_total))
        else:
            woe1 = math.log(rate_event / rate_non_event)
        woe_dict[x1] = woe1
        iv += (rate_event - rate_non_event)*woe1
    return woe_dict, iv
    
def woe(X, y, event=1):
#        Calculate woe of each feature category and information value
#        :param X: 2-D numpy array explanatory features which should be discreted already
#        :param y: 1-D numpy array target variable which should be binary
#        :param event: value of binary stands for the event to predict
#        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
#                 numpy array of information value of each feature
    X1 = feature_discretion(X)
    check_target_binary(y)
    
    res_woe = []
    res_iv = []
    for i in range(X1.shape[-1]):
        x = X1[:, i]
        woe_dict, iv1 = woe_single_x(x, y, event=event)
        res_woe.append(woe_dict)
        res_iv.append(iv1)
    return np.array(res_woe), np.array(res_iv)
    
def woe_replace(X, woe_arr):
#        replace the explanatory feature categories with its woe value
#        :param X: 2-D numpy array explanatory features which should be discreted already
#        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
#        :return: the new numpy array in which woe values filled
    X = feature_discretion(X)
    if X.shape[-1] != woe_arr.shape[-1]:
        raise ValueError('WOE dict array length must be equal with features length')
    res = np.copy(X).astype(float)
    idx = 0
    for woe_dict in woe_arr:
        for k in woe_dict.keys():
            woe = woe_dict[k]
            res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
        idx += 1
    return res 
    
def combine(list):
    res = ''
    for item in list:
        res+=str(item)
    return res
    
      
def combined_iv(X, y, masks, event=1):
#        calcute the information vlaue of combination features
#        :param X: 2-D numpy array explanatory features which should be discreted already
#        :param y: 1-D numpy array target variable
#        :param masks: 1-D numpy array of masks stands for which features are included in combination,
#                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
#        :param event: value of binary stands for the event to predict
#        :return: woe dictionary and information value of combined features
    if masks.shape[-1] != X.shape[-1]:
        raise ValueError('Masks array length must be equal with features length')

    x = X[:, np.where(masks == 1)[0]]
    tmp = []
    for i in range(x.shape[0]):
        tmp.append(combine(x[i, :]))

    dumy = np.array(tmp)
    # dumy_labels = np.unique(dumy)
    woe, iv = woe_single_x(dumy, y, event)
    return woe, iv
    

from pandas import DataFrame

data = pd.read_csv(r'E:\evan_mime\部门文件\项目文件\geTui\个推_LR模型测试_v2\lrData_v2.csv',encoding='gbk').fillna(value=-99999)
y = np.array(data.Y)
#x = np.array(data.衍生变量_暂去地个数)
res_woe,res_iv = woe(np.array(data),y)
res = woe_replace(np.array(data), res_woe)
res_frame = DataFrame(res,columns = data.columns)
res_frame.to_csv(r'C:\Users\memedai\Desktop\res_frame.csv')