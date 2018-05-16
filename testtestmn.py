# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 09:35:22 2017

@author: jiajun
"""
def mmn(yuce,yuce0,shengxnum,zhongjnum,a,p,dayin,yanchi):                   
    import numpy as np
    name='鼠牛虎兔龙蛇马羊猴鸡狗猪'
    pre=np.zeros((1,yuce0-yuce+1),dtype='int64')
    js=-1
    for yuce1 in np.arange(yuce,yuce0+1,1):
        js+=1
        a0=np.sum(p[a[yuce1+yanchi]-1,:],axis=0)
        a2=np.argsort(a0)
        a3=a2[12-shengxnum]
        if dayin==0:
            print (yuce1,'期预测为：',name[a3])       
        try:
            if len(np.intersect1d(a3,a[yuce1-1]-1))>=zhongjnum:
                pre[:,js]=1    
        except:
            pass
    re=np.mean(pre)               
    return (re,pre,a3)

def mmn1(yanchi,k,llen,shengxnum):       
    import numpy as np
    a= np.loadtxt('1.txt',dtype='int64')
    a=a[::-1,:]
    m,n=a.shape
    p=np.zeros((12,12))
    for i in np.arange(m-yanchi-1):
        for j in np.arange(n):
            for h in np.arange(n):
                p[a[i,j]-1,a[i+yanchi+1,h]-1]=1+p[a[i,j]-1,a[i+yanchi+1,h]-1]
    aa=np.zeros((k,1))
    for i in np.arange(0,k,1):
        re,_,_=mmn(1+i,llen+i,shengxnum,1,a,p,1,yanchi)
        aa[i]=np.round(re,2)                           
    return (a,p,aa)    
    
def testmmn(qs,llen,aa,a,p,shengxnum,yanchi):# 滑移数据长度，滑移窗口设定为1
    import numpy as np
    lstart=llen-1    
    a1=np.zeros((llen+1,2))
    a1[:,0]=np.arange(0,llen+1,1)
    n=-1
    for j in np.arange(0,1+1/llen,1/llen):
        n+=1
        a1[n,1]=np.sum(aa==np.round(j,2))
    _,pre0,_=mmn(qs+1,qs+lstart,shengxnum,1,a,p,1,yanchi)    
    wmx=sum(pre0[0])+1
    wmn=sum(pre0[0])    
    ratio=a1[np.where(a1[:,0]==wmx)[0],1]/a1[np.where(a1[:,0]==wmn)[0],1]
    mn=a1[np.where(a1[:,0]==wmn)[0],1]
    print (wmx/llen,wmn/llen,pre0,ratio,1/ratio,np.mean(aa))
    return (np.mean(aa),ratio,pre0[0][0],mn)