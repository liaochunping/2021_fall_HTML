import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
train_dataX = pd.read_csv('x_in.csv')
train_dataX = train_dataX.drop(labels='Customer ID',axis=1)
train_dataY = pd.read_csv('y_in.csv')
train_dataY = train_dataY.drop(labels='Customer ID',axis=1)
test_dataX = pd.read_csv('x_out.csv')
test_dataX = test_dataX.drop(labels='Customer ID',axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif



x_res , y_res = SMOTE().fit_resample(train_dataX, train_dataY)
under = RandomUnderSampler(sampling_strategy={0: 130, 1: 200, 2: 177, 3: 201, 4: 124, 5: 116}, random_state=10521517)
#x_res , y_res = under.fit_resample(train_dataX, train_dataY)

'''
model  = SelectKBest(mutual_info_classif, k=15)
X_new = model.fit_transform(x_res, y_res.values.ravel())
Xtest_new = model.transform(test_dataX)
x_train , x_test ,y_train,y_test = train_test_split(X_new,y_res,test_size=0.2,random_state = 32)
rfc = RandomForestClassifier(n_estimators = 6000,n_jobs=-1,max_depth=6)
rfc.fit(x_train,y_train.values.ravel())
y1 = rfc.predict(Xtest_new)
print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))
y = pd.DataFrame(y1,columns =['Churn Category'])
#y.to_csv('testrf.csv')
print(y.groupby('Churn Category').size())


#0.48683683014833623
#0.45430251202565475

#0 884
#1 203
#2 70
#3 54
#4 51
#5 147

#0.31770

'''
'''
model  = SelectKBest(mutual_info_classif, k=30)
X_new = model.fit_transform(x_res, y_res.values.ravel())
Xtest_new = model.transform(test_dataX)
x_train , x_test ,y_train,y_test = train_test_split(X_new,y_res,test_size=0.2,random_state = 32)
rfc = RandomForestClassifier(n_estimators = 2000,n_jobs=-1,max_depth=4, min_samples_split=4, min_samples_leaf=10)
rfc.fit(x_train,y_train.values.ravel())
y1 = rfc.predict(Xtest_new)
print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))
y = pd.DataFrame(y1,columns =['Churn Category'])
#y.to_csv('testrf.csv')
print(y.groupby('Churn Category').size())
'''





from sklearn.feature_selection import RFE

rfc = RandomForestClassifier(n_estimators = 1000,n_jobs=-1,oob_score=(True))
model =RFE(rfc, n_features_to_select=16)
X_newrf = model.fit_transform(x_res, y_res.values.ravel())
x_train , x_test ,y_train,y_test = train_test_split(X_newrf,y_res,test_size=0.2,random_state = 17)
rfc = RandomForestClassifier(n_estimators = 3000,n_jobs=-1, min_samples_leaf=25,oob_score=(True))
rfc.fit(x_train,y_train.values.ravel())
Xtest_new = model.transform(test_dataX)
y1 = rfc.predict(Xtest_new)
print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))
y = pd.DataFrame(y1,columns =['Churn Category'])
y.to_csv('testrfbyrf.csv')
print(y.groupby('Churn Category').size())



#904
#151
#106
#55
#75
#118
# 0.31979
