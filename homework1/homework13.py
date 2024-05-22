import numpy as np
import random

def getdata():
        text =[]
        path = 'hw1_train.dat.txt'
        with open(path) as f:
           for line in f:
               text.append([float(i) for i in line.split()])
        mm=np.asarray(text)
        rows=len(mm)
        X=np.c_[np.ones(rows),mm[:,:-1]]
        Y=mm[:,-1]
       
        return X,Y

def sign(x):
    if x>0:
        return 1
    else:
        return -1

def pla(X,Y,k=False):
    n=len(X)
    cols = len(X[0])
    w=np.zeros(cols)
    idx=range(n)
    if k:
        idx=random.sample(idx,n)
    k=0
    update=False
    while True:
        i=idx[(random.sample(idx,1))[0]]
        if sign(np.dot(X[i],w))!=Y[i]:
            w=w+Y[i]*X[i]
            update=True
            k=0
        else:
            k+=1
        if k==500:
            if update==False:
                break
            k=0
            update=False
    return w

def random_cycle(n):
    X, Y = getdata()
    cnt=0
    sum = 0
    for i in range(n):
        cnt=pla(X,Y,k=True)
        for i  in range(11):
            sum += np.sum((cnt[i])**2)       
    return( sum)
if __name__=="__main__":
    su = 0
        
    su  = su +((random_cycle(1000))/1000)
    print(su) 

       
