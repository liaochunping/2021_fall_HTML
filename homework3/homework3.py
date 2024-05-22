import numpy as np
import random

def getdata(a):
        text =[]
        if a ==1:
            path = 'hw3_train.dat.txt'
        elif a ==2:
            path = 'hw3_test.dat.txt'
        with open(path) as f:
           for line in f:
               text.append([float(i) for i in line.split()])
        mm=np.asarray(text)
        X=mm[:,:-1]
        Y=mm[:,-1]
       
        return X,Y

       
        return X,Y
def functionQ(X , q):
    function = []
    func_temp =[1]
    for i in X:
        for j in range(1,q+1):
            for x in i[0:]:
                func_temp.append(x**j)
        function.append(func_temp)
        func_temp = [1]
    function_array = np.array(function)
    return function_array
                    
def functionFullQ(X , q):
    function = []
    func_temp =[1]
    row = X.shape[1]
    a = 1
    b = 0
    for i in X:
        for j in range(1,q+1):
            for x in i[0:]:
                func_temp.append(x**j)
                for _ in range(a , row):
                    func_temp.append(x*X[b][_])
                a+=1
        function.append(func_temp)
        func_temp = [1]
        a = 1
        b+=1
    function_array = np.array(function)
    return function_array
def functionLower(X, q):
    function = []
    func_temp =[1]
    for i in X:
        for x in i[0:q]:
            func_temp.append(x)
        function.append(func_temp)
        func_temp = [1]
    function_array = np.array(function)
    return function_array
def functionRandom(X, n):
    random.seed(n)
    function = []
    func_temp =[1]
    list1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    list2 = [5,6,7,8,9]
    random.shuffle(list1)
    list3 = np.delete(list1,list2)
    for i in range(X.shape[0]):
        newX = np.delete(X[i],list3)
        for _ in range(newX.shape[0]):
            func_temp.append(newX[_])
        function.append(func_temp)
        func_temp = [1]
                
    function_array = np.array(function)
    return function_array


def linear_regression(x , y):   
    x_t =np.transpose(x) 
    t = x_t.dot(x)
    t_inv = np.linalg.inv(t)
    w_lin = (t_inv.dot(x_t)).dot(y)
    return w_lin

def E_in_minus_E_out_bin(w, x, y, x_test, y_test,n):
    
    # E_in_bin
    E_in_bin = 0
    N = y.size
    for xn, yn in zip(x, y):
        E_in_bin += (np.sign(xn.dot(w)) != yn)/N
    if n :
        print("E_in_bin:", E_in_bin)
    # E_out_bin
    E_out_bin = 0
    N_test = y_test.size
    for xn, yn in zip(x_test, y_test):
        E_out_bin += (np.sign(xn.dot(w)) != yn)/N_test
    if n :
        print("E_out_bin:", E_out_bin)
        print ("|E_in_bin - E_out_bin|:", abs(E_in_bin - E_out_bin))
    if n == 0:
        return abs(E_in_bin - E_out_bin)



x_train,y_train= getdata(1)
x_test,y_test = getdata(2)

print('q12')
x_train_trans_2 = functionQ(x_train , 2)
x_test_trans_2 =functionQ(x_test, 2)
wlin_train_2 = linear_regression(x_train_trans_2, y_train)
E_in_minus_E_out_bin(wlin_train_2, x_train_trans_2, y_train, x_test_trans_2, y_test,1)

print('q13')
x_train_trans_8 = functionQ(x_train , 8)
x_test_trans_8 =functionQ(x_test, 8)

print(x_test_trans_8.shape)
wlin_train_8 = linear_regression(x_train_trans_8, y_train)
E_in_minus_E_out_bin(wlin_train_8, x_train_trans_8, y_train, x_test_trans_8, y_test,1)

print('q14')
x_train_fulltrans_2 = functionFullQ(x_train, 2)
x_test_fulltrans_2 = functionFullQ(x_test, 2)
print(x_test_fulltrans_2.shape)
wlin_fulltrain_2 = linear_regression(x_train_fulltrans_2, y_train)
E_in_minus_E_out_bin(wlin_fulltrain_2, x_train_fulltrans_2, y_train, x_test_fulltrans_2, y_test,1)

print('q15')
x_train_compose =[]
for i in range(1,11):
    x_train_c = functionLower(x_train, i)
    x_test_c = functionLower(x_test, i)
    w_c = linear_regression(x_train_c, y_train)
    temp = E_in_minus_E_out_bin(w_c, x_train_c, y_train, x_test_c, y_test,0)
    x_train_compose.append(temp)
print("The minimum of i is ",x_train_compose.index(min(x_train_compose))+1)
print("|E_in_bin - E_out_bin|:",min(x_train_compose))

print('q16')
temprandom=0
for _ in range(200):
    n = random.random()
    x_train_random =functionRandom(x_train, n)
    x_test_random = functionRandom(x_test, n)
    w_random = linear_regression(x_train_random, y_train)
    temprandom += E_in_minus_E_out_bin(w_random, x_train_random, y_train, x_test_random, y_test, 0)
print("the average |E_in_bin - E_out_bin|over 200 experiments :" , temprandom/200)
    
    











