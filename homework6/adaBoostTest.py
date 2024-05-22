import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
def getdata(a):
        text =[]
        if a ==1:
            path = 'hw6_train.dat.txt'
        elif a ==2:
            path = 'hw6_test.dat.txt'
        with open(path) as f:
           for line in f:
               text.append([float(i) for i in line.split()])
        mm=np.asarray(text)
        X=mm[:,:-1]
        Y=mm[:,-1]
       
        return X,Y
def plot_dataset(X, Y, filename, picname):

    plt.figure()
    plt.plot(X[Y > 0, 0], X[Y > 0, 1], 'bo')
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], 'ro')    

    
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(picname)
    plt.savefig(filename, dpi=300)
    plt.show()
# define dataset

trainX ,trainY  = getdata(1)[0],getdata(1)[1]
testX ,testY = getdata(2)[0],getdata(2)[1]
'''
plot_dataset(trainX, trainY, 'train_data.png', 'train_data')
plot_dataset(testX, testY, 'test_data.png', 'test_data')
'''
#弱分類器 , 限制最大深度1
def adaboost(trainX, trainY ,testX,testY,T ,weak_clftree= DecisionTreeClassifier(max_depth = 1)):
    n_train ,n_test = len(trainX),len(trainY)
    w = np.ones(n_train)
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(T):
        pass
    pass
def my_adaboost_clf(Y_train, X_train, Y_test, X_test, M=5000, weak_clf=DecisionTreeClassifier(max_depth = 4,max_features=(1))):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    for i in range(M):
        print('\rt = %d' % (i+1), end='', flush=True)
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        
        
        
        #ein
        #print("weak_clf_%02d train acc: %.4f"% (i + 1, 1 - sum(miss) / n_train),flush= True)
        
        
        # Error
        err_m = np.dot(w, miss)
        # Alpha
        alpha_m =float(0.5 * np.log((1 - err_m) / float(err_m)))
        # New weights
        miss2 = [x if x==1 else -1 for x in miss] # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)
        

        # Add to prediction
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)
    
    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1

    print("My AdaBoost clf train accuracy: %.4f" % (sum(pred_train == Y_train) / n_train))
    print("My AdaBoost clf test accuracy: %.4f" % (sum(pred_test == Y_test) / n_test))
my_adaboost_clf(trainY, trainX, testY, testX)