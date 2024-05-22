
import random
import numpy as np
import math

### HELPER FUNCTIONS ###
def sigmoid(s):
    return 1/(1 + math.exp(-s))

def convert_to_array(data):
    """
    This fumction takes in the data (which is in list form)
    and convert them into matrix and vector
    """
    X_list = []
    y_list = []
    temp_x = [1]
    i = 1
    for item in data:
        item = float(item)
        # xn
        if i % 11 != 0:
            temp_x.append(item)
        # yn
        if i % 11 == 0:
            X_list.append(temp_x)
            y_list.append(item)
            temp_x = [1]
        i += 1
    # Convert into np.array
    X = np.array(X_list)
    y = np.array(y_list)

    return (X, y)
def non_linear_transform(X, Q):
    """ This function augment the matrix X according to trandfrom """
    X_new_list = []
    xn_temp = [1]
    for xn in X:
        for order in range(1, Q+1):
            for x_elt in xn[1:]:
                xn_temp.append(x_elt**order)            
        X_new_list.append(xn_temp)
        xn_temp = [1]
    X_new = np.array(X_new_list)

    return X_new

def E_out_minus_E_in_bin(w, X, y, X_test, y_test):
    
    # E_in_bin
    E_in_bin = 0
    N = y.size
    for xn, yn in zip(X, y):
        E_in_bin += (np.sign(xn.dot(w)) != yn)/N
    print("E_in_bin:", E_in_bin)
    # E_out_bin
    E_out_bin = 0
    N_test = y_test.size
    for xn, yn in zip(X_test, y_test):
        E_out_bin += (np.sign(xn.dot(w)) != yn)/N_test
    print("E_out_bin:", E_out_bin)
    print ("|E_in_bin - E_out_bin|:", abs(E_out_bin - E_in_bin))

### HELPER FUNCTIONS END ###

# Linear Regression
def lin_reg(X, y):
    """
    This function calculate the weight using linear regression

    Input
        X: np.array, input matrix with input vectors being its "rows".
        y: np.array, the output vector

    Output
    (tuple)
        E_in_sqr: double, the avg error when using linear regression    
        w_lin: np.ndarray, the weight vector
    """
    # Linear regression
    w_lin = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    # Calculate E_in
    E_in_sqr = 0
    N = y.size
    for xn, yn in zip(X, y):
        E_in_sqr += (1/N)*(xn.dot(w_lin) - yn)**2
    # Return
    return (E_in_sqr, w_lin) 

# Stochastic Gradient Descent Linear
def SGD_linear (eta, w0, X, y, exp_times, E_w_lin):
    """
    This function implement the SGD for linear regression

    Inout
        eta: double, learning rate
        w0: np.array, Initial value for weight vector
        X: np.array, input matrix with input vectors being its "rows".
        y: np.array, the output vector
        exp_times: int, experiment times
        E_w_lin: double, error obtained by linear regression
    
    Output
        iteration: double, # of avg iteration until the criteria meet
    """
    # Loop for exp_times
    iteration = 0 
    i = 0
    for _ in range(exp_times):
        # Initialization 
        random.seed()
        iter_temp = 0
        w_t = w0
        # While the error is bigger than E_w_lin
        E_in_SGD = math.inf 
        while E_in_SGD > 1.01*E_w_lin:
            # Pick a sample ramdomly
            N = y.size
            pick = random.randint(0, N-1)
            xn = X[pick]
            yn = y[pick]
            # Calculate the gradient
            E_grad = 2*(np.ndarray.dot(w_t, xn) - yn)*xn 
            # Take a step (toward negative gradient)
            w_t = w_t + eta*(-E_grad)
            iter_temp += 1
            # Calculate E_in
            E_in_SGD = 0
            for xn, yn in zip(X, y):
                E_in_SGD += (1/N)*(xn.dot(w_t) - yn)**2
        # Record the iteration
        print("iteration", i, " success")
        i += 1
        iteration += iter_temp/exp_times
    # Return the average iteration
    return iteration

# Stochastic Gradient Descent Logistic
def SGD_logistic (eta, w0, X, y, exp_times):
    """
    This function use logistic SGD, and output the average error after 500
    iteration
    """
    # Loop for exp_times
    E_ce = 0
    for _ in range(exp_times):
        # Initialization 
        random.seed()
        w_t = w0
        N = y.size
        # Loop for 500 itertions
        for _ in range(200):
            xn = X[_]
            yn = y[_]
            # Calculate the gradient
            E_grad = -sigmoid(-yn*np.ndarray.dot(w_t, xn))*yn*xn
            # Take a step
            w_t += eta * (-E_grad)
        return w_t
        # Record the error
        E_ce_temp = 0
        for xn, yn in zip(X, y):
            E_ce_temp += (1/N)*math.log(1 + math.exp(-yn*w_t.dot(xn)))

        E_ce += E_ce_temp/exp_times
        
    # Return the average error
    return E_ce

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
    # y = 0 -----> y = -1
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
def e_out_bin(w,testdata,n):
    x = np.array(list (testdata[0]))
    y = np.array(list (testdata[1]))
    E_out_bin = 0
    for xn ,yn in zip(x, y):
        if (w.dot(xn))*yn >0:
            E_out_bin += 1
    print(E_out_bin)
    return (E_out_bin/n)
    
def test_log(w,testdata,n):
    x = np.array(list (testdata[0]))
    y = np.array(list (testdata[1]))
    w_t =np.transpose(w_lin)
    E_out_bin = 0
    error = 0
    sum = 0
    
    for i in range(n):
        print((sigmoid(-(x.dot(w)[i]))*y[i]))
        if (sigmoid(-(x.dot(w)[i]))*y[i])> 0 :
            E_out_bin += 1
        elif (sigmoid(-(x.dot(w)[i]))*y[i])< 0 :
            error +=1
    
    print(E_out_bin)
    print(error)
    print(sum)
    return (error/n)
    
    
    
    
    
    
    
    
    
    
# Get data from the website
data =flipcointogetdata(200)
test_data = flipcointogetdata(5000)
X  =np.array(list(data[0]))
y=np.array(list(data[1]))
X_test  =np.array(list(test_data[0]))
y_test =np.array(list(test_data[1]))
        
# Store the inputs and label into lists
print("X", X.shape)

# TEST: Problem 14
(E_in_sqr, w_lin)= lin_reg(X, y)
e_in = e_in(w_lin ,test_data , 5000)
print("Average E_in_sqr:", E_in_sqr)
print(e_in)
# TEST: Problem 15
w0 = np.ndarray((3, ))
w0.fill(0)

'''
iteration = SGD_linear(0.1, w0, X, y, 100, E_in_sqr)  
print("Average iteration:", iteration)
'''
w0.fill(0)
# TEST: Problem 16
E_in_logi = SGD_logistic(0.1, w0, X, y, 500)
print("Average cross-entropy error:", E_in_logi)
print("eout_bin",e_out_bin(w_lin,test_data,5000))
'''
# TEST : Problem 17
E_in_logi = SGD_logistic(0.1, w_lin, X, y, 100)
print("Average cross-entropy error:", E_in_logi)
'''
'''
# TEST: Problem 18
E_out_minus_E_in_bin(w_lin, X, y, X_test, y_test)
'''
'''
# TEST: Problem 19, 20
Q = 10
X_tran = non_linear_transform(X, Q)
X_test_tran = non_linear_transform(X_test, Q)
w_lin = lin_reg(X_tran, y)[1]
print (X_tran.shape)
print (w_lin.shape)
E_out_minus_E_in_bin(w_lin, X_tran, y, X_test_tran, y_test)
'''