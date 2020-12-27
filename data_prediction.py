# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:11:04 2017

@author: jiajun
"""
import multiprocessing
import scipy.io as sio
from sklearn.externals import joblib
from sklearn import svm
from numpy import *
from testtestmn import mmn
from testtestmn import mmn1
from testtestmn import testmmn
from collections import Counter
from sklearn import preprocessing
def caipiao(slen,yanchi,alen,llen,shengxnum):
    k=zeros((yanchi+1,slen))
    ratio=zeros((yanchi+1,slen))
    pre=zeros((yanchi+1,slen))
    mn=zeros((yanchi+1,slen))
    print ('max','min','xulie','zhongjiangbi','zhongjianglv')
    for i in arange(yanchi+1):
        a,p,aa=mmn1(i,alen,llen,shengxnum)
        for j in arange(slen):
            k[i,j],ratio[i,j],pre[i,j],mn[i,j]=testmmn(j,llen,aa,a,p,shengxnum,i)
    return (k,ratio,pre,mn)

def user(a):
    yuce=3      #用来指定预测的期数，0表示最新一期
    zjnum=[]
    for shengxnum in arange(1,13,1):
        ratio=loadtxt("ratio"+str(shengxnum)+".txt")
        mn=loadtxt("mn"+str(shengxnum)+".txt")
        pre=loadtxt("pre"+str(shengxnum)+".txt")
        x=ratio[:,1:]
        y=pre[:,:-1]
        mn0=mn[:,1:]
        model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=False)    
        name='abcdefghijkl'
        xmaxloc=argsort(x,axis=1)[:,-1]
        xminloc=argsort(x,axis=1)[:,0]
        xt=amax(x, axis=1)
        xtm=amin(x,axis=1)
        xt=hstack((xt,xtm))    
        yt=y[arange(len(xmaxloc)),xmaxloc]
        ytm=y[arange(len(xminloc)),xminloc]
        yt=hstack((yt,ytm))
        xt00=mn0
        xt0=xt00[arange(len(xmaxloc)),xmaxloc]
        xt0m=xt00[arange(len(xminloc)),xminloc]
        xt0=hstack((xt0,xt0m))
        xtr=array([xt,xt0])
        xtr=xtr[:,where(xt!=inf)[0]]
        xtmx=xtr[0,:].max()
        xtmn=xtr[0,:].min()    
        xt0mx=xtr[1,:].max()
        xt0mn=xtr[1,:].min()    
        yt=yt[where(xt!=inf)[0]]
        xtr=preprocessing.minmax_scale(xtr)
        model.fit(xtr.T, yt)
        try:
            xtest=array([ratio[:,yuce],mn[:,yuce]])
        except:
            xtest=array([ratio,mn])
        xtest[where(xtest==inf)[0],where(xtest==inf)[1]]=100
        num0=intersect1d(where(xtest[0,:]<=xtmx)[0],where(xtest[0,:]>=xtmn)[0])
        num1=intersect1d(where(xtest[1,:]<=xt0mx)[0],where(xtest[1,:]>=xt0mn)[0])
        num=intersect1d(num0,num1)
        if len(num)!=0:
            predictedpro=model.predict_proba(preprocessing.minmax_scale(xtest.T))
            dx=0
            ycg=where(predictedpro[:,dx]==predictedpro[:,dx].max())[0]
            ycg0=where(predictedpro[:,dx]>=0.6)[0]
            ycg1=where(predictedpro[:,dx]<=0.7)[0]
            ycg0=intersect1d(ycg0,ycg1)
            ycg=intersect1d(ycg,ycg0)    
            ycg=intersect1d(ycg,num)
            for yanchi in ycg:
                a,p,_=mmn1(yanchi,4000,10,shengxnum)
                _,_,a3=mmn(yuce,yuce,shengxnum,1,a,p,1,yanchi)
                zjnum.append(name[a3]) 
    return (zjnum)
                
if __name__=="__main__":
    yuce=2     #用来指定预测的期数，0表示最新一期
    newstart=1  #用来判定是否需要重新获取数据,0需要,非0,不需要，第一次运行需要设定为0
    if newstart==0:
        for shengxnum in arange(1,13,1):
            _,ratio,pre,mn=caipiao(2,50,4000,10,shengxnum)
            savetxt("ratio"+str(shengxnum)+".txt",ratio)
            savetxt("mn"+str(shengxnum)+".txt",mn)
            savetxt("pre"+str(shengxnum)+".txt",pre)
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    zjnumall=[]
    bzjnumall=[]
    a=loadtxt('1.txt',dtype='int64')
    a=a[::-1,:]
    times=5
    xs=[a]*times
    zjnum=pool.map(user,xs)
    pool.close()
    pool.join()
    l=[]
    for zjnumc in zjnum:
            l+=zjnumc
    name='abcdefghijkl'
    if yuce is not 0:
        print('实际——',[name[s-1] for s in a[yuce-1]])
    print ('预测:',Counter(l),'统计总数',len(l),'计算次数',times)
