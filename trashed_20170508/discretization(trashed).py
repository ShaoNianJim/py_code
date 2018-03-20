# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:13:53 2016
@author: evan
"""

import numpy as np
from math import log
from pandas import Series
import copy

def calcEntropy(x,y,event=1):
    
    """
     : calculate the entropy of the feature
    x: array, feature
    y: array, dependent variable    
    
    """    
    numExamples = len(x)
    xLabels = set(Series(x).dropna())
    entropyValue = 0
    for i in xLabels:
        #print (i)
        yTemp = y[np.where(x==i)]
        numTemp = len(yTemp)
        numPositive = yTemp[np.where(yTemp==event)].size
        numNegative = numTemp - numPositive
        probPositive = float(numPositive)/numTemp
        probNegative = float(numNegative)/numTemp
        weight = float(numTemp)/numExamples
        if probPositive==0 or probNegative==0:
            entropyValue -= 0
        else:
            entropyValue -= (probPositive*log(probPositive,2)+ probNegative*log(probNegative,2))*weight
        #print (entropyValue)
    return entropyValue
    
def binarySplit(x,splitPoint):
    """
    : at some point split x into binary
    x: array
    """
    f = x.copy()
    f[np.where(f<=splitPoint)] = 0
    f[np.where(f> splitPoint)] = 1
    return f
    
def searchBestPoint(x,y,event=1):
    """
    : get the best point to split x
    x: array
    y: array
    """
    intX  = set(Series(x).dropna())
    splitEntropy = {}
    for i in intX:
        binaryX = binarySplit(x,i)
        iEntropy = calcEntropy(binaryX,y,event)
        splitEntropy[i] = iEntropy
    sortEntropy = sorted(list(splitEntropy.items()),key=lambda x:x[1])
    return sortEntropy[0][0], sortEntropy[0][1]
    
def binaryBestPoints(x,y,event=1):
    points = []
    firstPoint, firstEntropy = searchBestPoint(x,y,event=1)
    points.append([firstPoint, firstEntropy])
    sortPoints = copy.deepcopy(points)
    sortPointsBins=len(sortPoints)
    if sortPointsBins==1:
        x1=x[np.where(x<=sortPoints[0][0])]
        y1=y[np.where(x<=sortPoints[0][0])]
        x2=x[np.where(x>sortPoints[0][0])]
        y2=y[np.where(x>sortPoints[0][0])]
        if len(x1)>len(x2):
            secPoint, secEntropy = searchBestPoint(x1,y1,event=1)  
        else:
            secPoint, secEntropy = searchBestPoint(x2,y2,event=1)
        points.append([secPoint, secEntropy])
    return points

if __name__=='__main__':
    x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    y=  np.array([1,0,0,0,0,0,1,1,1,1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])  
    print (binaryBestPoints(x,y))     
    