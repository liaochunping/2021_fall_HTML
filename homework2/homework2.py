import numpy as np
import random
import math
import matplotlib.pyplot as plt

def flipcointogetdata(n):
    # y = 0 -----> y = -1
    data_x =[]
    data_y =[]
    for i in range (n):
        #set random seed 
        random.seed()
        y = random.randint(0,1)
        if y == 1 :
            x1 = random.gauss(2, np.sqrt(0.6))
            x2 = random.gauss(3, np.sqrt(0.6))
            data_x.append([1,x1,x2])
            data_y.append(1)
        elif y ==0:
            x1 = random.gauss(0, np.sqrt(0.4))
            x2 = random.gauss(4, np.sqrt(0.4))
            data_x.append([1,x1,x2])
            data_y.append(-1)
    return data_x ,data_y
def flipcointogetdataadd(n,data):
    data_x =data[0]
    data_y =data[1]
    for i in range (n):
        #set random seed 
        random.seed()
       
        x1 = random.gauss(6, np.sqrt(0.3))
        x2 = random.gauss(0, np.sqrt(0.1))
        data_x.append([1,x1,x2])
        data_y.append(1)
        
    return data_x ,data_y

def linear_regression(data):
    x = np.array(list (data[0]))
    x_t =np.transpose(x)
    y = np.array(list (data[1]))
    t = x_t.dot(x)
    t_inv = np.linalg.inv(t)
    w_lin = (t_inv.dot(x_t)).dot(y)
   # y_ = w_lin[0]+w_lin[1]*data[0]+w_lin*data[1]
    
    return w_lin

def e_in(w_lin , data , n):
    x = np.array(list (data[0]))
    x_t =np.transpose(x)
    x_tx = x_t.dot(x)
    y = np.array(list (data[1]))
    y_t =np.transpose(y)
    y_ty = y_t.dot(y)
    x_ty = x_t.dot(y)
    x_txw = x_tx.dot(w_lin)
    w_t =np.transpose(w_lin)
    # e_in = 1/n(wt*xt*xw-2*w^t*x^t*y+y^t*y)
    e_in = (w_t.dot(x_txw)-2*(w_t.dot(x_ty))+y_ty)/n
    #print (e_in)
    return e_in

def sigmoid(s):
    return 1/(1 + math.exp(-s))    


def logistic_regression(data , eta,itr ):
        x = np.array(list (data[0]))
        y = np.array(list (data[1]))
        n = y.size
        w_t =np.zeros(x.shape[1])
        for i in range(itr):
            for i in range(n):
                xn =x[i]
                yn =y[i]
                e_grad = -sigmoid(-yn*np.ndarray.dot(w_t, xn))*yn*xn
                w_t += eta*(-e_grad)
        return (w_t)

def test_log(w,testdata,n):
    x = np.array(list (testdata[0]))
    y = np.array(list (testdata[1]))
    E_out_bin = 0
    error = 0
    
    for i in range(n):
        
        if (((sigmoid(-(x.dot(w)[i]))-0.5)*y[i]))> 0 :
            E_out_bin += 1
        elif (sigmoid(-(x.dot(w)[i]))-0.5)*y[i]< 0 :
            error += 1
    
    return (E_out_bin/n)
        


def linear01error(wlin,data,n):
    x = np.array(list (data[0]))
    y = np.array(list (data[1]))
    right = 0
    error = 0
    for i in range(n):
        if x.dot(wlin)[i]*y[i] >0:
            right+=1
        elif x.dot(wlin)[i]*y[i]<0:
            error+=1
            
    return error/n
    


# 13-14
sum = 0
sum2 = 0
sum3 = 0
sum4 = 0
eout10 = 0
ein10 = 0
ans = 0
n =100
test = 5000
train =200
ans1 = 0
itr = 500
#test
#15
for i in range(n):
    random.seed(n)
    traindata =flipcointogetdata(200)
    testdata = flipcointogetdata(5000)
    traindata = flipcointogetdataadd(20, traindata)
    w_lin = linear_regression(traindata)
    eout10 = linear01error(w_lin, testdata, 5000)
    
    w_log = logistic_regression(traindata, 0.1 ,itr )
    eout10log = test_log(w_log, testdata, test)
    sum3 += eout10
    sum4 += eout10log
sum3 = sum3/n
sum4 = sum4/n
print("Q15")
print("eout10linear(D):",sum3)
print('err log', sum4)

for i in range(n):
    random.seed(n)
    traindata =flipcointogetdata(train)
    testdata = flipcointogetdata(test)
    w_lin = linear_regression(traindata)
    eout10 = linear01error(w_lin, testdata, test)
    ein10 = linear01error(w_lin, traindata, train)
    a = e_in(w_lin , traindata , train)
    w_log = logistic_regression(traindata, 0.1 ,itr )
    eout10log = test_log(w_log, testdata, test)
    sum2 += a
    sum3 += eout10
    sum4 += eout10log
    
    ans += abs(eout10-ein10)
    
ans = ans/n
sum2 = sum2/n
sum3 = sum3/n
sum4 = sum4/n
#14
print("Q14")
print("error rate" , ans )
#13
print("Q13")
print("sqr" , sum2)
#16
print("Q16")
print("eout10linear(D):",sum3)
print("logerr", sum4)

'''
for i in range(200):
    y = w_lin[0]+w_lin[1]*traindata[0][i][1]+w_lin[2]*traindata[0][i][2]
   
    if traindata[1][i] == 1:
     plt.plot(traindata[0][i][1],traindata[0][i][2] , '.', c ='b')
    else:
     plt.plot(traindata[0][i][1],traindata[0][i][2] , '.', c ='r')
plt.show()
'''































