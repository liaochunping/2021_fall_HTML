import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
'''
train_dataX = pd.read_csv('x_in.csv')
train_dataX = train_dataX.drop(labels='Customer ID',axis=1)
train_dataY = pd.read_csv('y_in_binary.csv')
train_dataY = train_dataY.drop(labels='Customer ID',axis=1)
test_dataX = pd.read_csv('x_out.csv')
test_dataX = test_dataX.drop(labels='Customer ID',axis=1)
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
'''
rfc = RandomForestClassifier(n_estimators = 100,max_depth = None,min_samples_leaf = 25,n_jobs=-1,oob_score=(True))
under = RandomUnderSampler(sampling_strategy={0: 1108, 1: 1108}, random_state=10521517)
train_dataX , train_dataY= under.fit_resample(train_dataX, train_dataY)
model =RFE(rfc, n_features_to_select=17)
train_dataX = model.fit_transform(train_dataX, train_dataY)
testx = model.transform(test_dataX)
#
x_train , x_test ,y_train,y_test = train_test_split(train_dataX,train_dataY,test_size=0.2,random_state = 32)
rfc = RandomForestClassifier(n_estimators = 1000,max_depth = None,min_samples_leaf = 25,n_jobs=-1,oob_score=(True))
rfc.fit(x_train,y_train)
testy = rfc.predict(testx)

print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))
y = pd.DataFrame(testy,columns =['Churn Category'])
#y.to_csv('y_binary.csv')
print(y.groupby('Churn Category').size())
'''
train_dataXaf = pd.read_csv('x_in_afterbinary.csv')
train_dataXaf = train_dataXaf.drop(labels='Customer ID',axis=1)
train_dataYaf = pd.read_csv('y_in_afterbinary.csv')
train_dataYaf = train_dataYaf.drop(labels='Customer ID',axis=1)
test_dataXaf = pd.read_csv('x_out_afterbinary.csv')
test_dataXaf = test_dataXaf.drop(labels='Customer ID',axis=1)
rfc = RandomForestClassifier(n_estimators = 100,max_depth = None,min_samples_leaf = 25,n_jobs=-1,oob_score=(True))

model  = SelectKBest(mutual_info_classif, k=15)
train_dataXaf = model.fit_transform(train_dataXaf, train_dataYaf)
testx = model.transform(test_dataXaf)
#
x_train , x_test ,y_train,y_test = train_test_split(train_dataXaf,train_dataYaf,test_size=0.2,random_state = 32)
rfc = RandomForestClassifier(n_estimators =5,n_jobs=-1,oob_score=(True))
rfc.fit(x_train,y_train)
testy = rfc.predict(testx)

print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))
y = pd.DataFrame(testy,columns =['Churn Category'])
y.to_csv('y_binary_afterbinary.csv')
print(y.groupby('Churn Category').size())