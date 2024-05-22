#encoding=utf8
import sys
import numpy as np
import math
from random import *

def read_input_data(path):
    x = []
    y = []
    for line in open(path).readlines():
        items = line.strip().split(' ')
        tmp_x = []
        for i in range(0,len(items)-1): tmp_x.append(float(items[i]))
        x.append(tmp_x)
        y.append(float(items[-1]))
    return np.array(x),np.array(y)

##
# x : input x
# y : input y
# u : weighted vector from last AdaBoost iterator
def calculate_weighted_Ein(x,y,u):
    # calculate median of interval & negative infinite & positive infinite
    thetas = np.array( [float("-inf")]+[ (x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1) ]+[float("inf")] )
    Ein = sum(u) # initial Ein as all wrong
    sign = 1
    target_theta = 0.0
    # positive and negative rays
    for theta in thetas:
        y_positive = np.where(x>theta,1,-1)
        y_negative = np.where(x<theta,1,-1)
        # difference between conditional stump and AdaBoost-stump
        weighted_error_positive = sum((y_positive!=y)*u)
        weighted_error_negative = sum((y_negative!=y)*u)
        if weighted_error_positive>weighted_error_negative:
            if Ein>weighted_error_negative:
                Ein = weighted_error_negative
                sign = -1
                target_theta = theta
        else:
            if Ein>weighted_error_positive:
                Ein = weighted_error_positive
                sign = 1
                target_theta = theta
    # two corner cases
    if target_theta==float("inf"):
        target_theta = 1.0
    if target_theta==float("-inf"):
        target_theta = -1.0
    # calculate scaling factor
    scalingFactor = 0.5
    errorRate = 0
    if sign==1:
        errorRate = 1.0*sum((np.where(x>target_theta,1,-1)!=y)*u)/sum(u)
        scalingFactor = math.sqrt( (1-errorRate)/errorRate )
        # update weight 
        u = scalingFactor*(np.where(x>target_theta,1,-1)!=y)*u + (np.where(x>target_theta,1,-1)==y)*u/scalingFactor
    else:
        errorRate = 1.0*sum((np.where(x<target_theta,1,-1)!=y)*u)/sum(u)
        scalingFactor = math.sqrt( (1-errorRate)/errorRate )
        # update weight 
        u = scalingFactor*(np.where(x<target_theta,1,-1)!=y)*u + (np.where(x<target_theta,1,-1)==y)*u/scalingFactor
    alpha = math.log(scalingFactor,math.e)
    # print errorRate
    return errorRate, u, alpha, target_theta, sign

def main():
    x,y = read_input_data("hw6_train.dat.txt")
    sorted_index = []
    for i in range(0, x.shape[1]): sorted_index.append(np.argsort(x[:,i]))
    # each feature dimension has its own sample weigted vector
    u = np.ones(x.shape[0])/x.shape[0]
    u_next = u
    T = 5
    alpha = np.ones(T)
    theta = np.ones(T)
    sign = np.ones(T)
    index = np.zeros(T)
    # Q16 mini error rate
    mini_error = 1
    for t in range(0, T):
        # best parameter in iteration t
        alpha_t = 1
        theta_t = 1
        sign_t = 1
        index_t = 1
        Eu = float("inf")
        # pick best feature dimension and corresponding parameters
        for i in range(0,x.shape[1]):
            xi = x[sorted_index[i],i]
            yi = y[sorted_index[i]]
            E, ui, a, th, s = calculate_weighted_Ein(xi, yi, u[sorted_index[i]])
            # print "E:"+str(E)
            if Eu>E:
                if mini_error>E: mini_error = E
                Eu = E
                alpha_t = a
                theta_t = th
                sign_t = s
                index_t = i
                u_next = ui
        alpha[t] = alpha_t
        theta[t] = theta_t
        sign[t] = sign_t
        index[t] = index_t
        # update u corresponding to the best i
        u[sorted_index[index_t]] = u_next
        # Q12 Ein(g1)  Q14 Ut(2)
        if t==0:
            Ein = 0
            if sign[t]==1:
                Ein = 1.0*sum(np.where(x[:,index_t]>theta_t,1,-1)!=y)/x.shape[0]
            else:
                Ein = 1.0*sum(np.where(x[:,index_t]<theta_t,1,-1)!=y)/x.shape[0]
            print ("Ein1:"+str(Ein))
            print ("Ut2:"+str(sum(u_next)))
        # Q15
        if t==T-1:
            print ("UT:"+str(sum(u_next)))
    # Q13 Ein(G)
    predict_y = np.zeros(x.shape[0])
    for t in range(0,T):
        print(t)
        if sign[t]==1:
            predict_y = predict_y + alpha[t]*np.where(x[:,index[t]]>theta[t],1,-1)
        else:
            predict_y = predict_y + alpha[t]*np.where(x[:,index[t]]<theta[t],1,-1)
    EinG = 1.0*sum(np.where(predict_y>0,1,-1)!=y)/x.shape[0]
    print ("EinG:"+str(EinG))

    # Q15
    print ("mini error rate:"+str(mini_error))

    # Q17 Eoutg1 Q18 EoutG
    test_x,test_y = read_input_data("hw6_test.dat.txt")
    predict_y = np.zeros(test_x.shape[0])
    # Q17
    if sign[0]==1:
        predict_y = np.where(test_x[:,index[0]]>theta[0],1,-1)
    else:
        predict_y = np.where(test_x[:,index[0]]<theta[0],1,-1)
    Eoutg1 = sum(predict_y!=test_y)
    print ("Eout1:"+str(1.0*Eoutg1/test_x.shape[0]))
    # Q18
    for t in range(0,T):
        if sign[t]==1:
            predict_y = predict_y + np.where(test_x[:,index[t]]>theta[t],1,-1)*alpha[t]
        else:
            predict_y = predict_y + np.where(test_x[:,index[t]]<theta[t],1,-1)*alpha[t]
    Eout = sum(np.where(predict_y>0,1,-1)!=test_y)
    print ("Eout:"+str(Eout*1.0/test_x.shape[0]))


if __name__ == '__main__':
    main()