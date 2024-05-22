import pandas as pd 
pd.set_option('mode.chained_assignment', None)
train = pd.read_csv('x_in.csv')
test = pd.read_csv('x_out.csv')
print(train)
#test.to_csv('testafter.csv')
x =train[['Referred a Friend','Tenure in Months','Phone Service','Avg Monthly Long Distance Charges',
          'Multiple Lines','Internet Service']]
xtest =test[['Referred a Friend','Tenure in Months','Phone Service','Avg Monthly Long Distance Charges',
          'Multiple Lines','Internet Service']]
y =train['Churn Category']
ytest =test['Churn Category']
y = y.map({'No Churn':0,'Competitor':1,'Dissatisfaction':2,'Attitude':3,'Price':4,'Other':5})
li =['Referred a Friend','Phone Service','Multiple Lines','Internet Service']
for i in li:
    x[i] = x[i].fillna('No')
    x[i] = x[i].map({'Yes':1,'No':0})
    xtest[i] = xtest[i].fillna('No')
    xtest[i] = xtest[i].map({'Yes':1,'No':0})
x['Tenure in Months'] =x['Tenure in Months'].fillna(x['Tenure in Months'].mean())
x['Avg Monthly Long Distance Charges'] =x['Avg Monthly Long Distance Charges'].fillna(x['Avg Monthly Long Distance Charges'].mean())
xtest['Tenure in Months'] =xtest['Tenure in Months'].fillna(xtest['Tenure in Months'].mean())
xtest['Avg Monthly Long Distance Charges'] =xtest['Avg Monthly Long Distance Charges'].fillna(xtest['Avg Monthly Long Distance Charges'].mean())
'''
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split(x,y,test_size=0.5)
'''
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x,y)
y_test = rfc.predict(xtest)
print(train)
pd.DataFrame(y_test).to_csv('pret.csv')
from xgboost import XGBClassifier
xgbc =XGBClassifier()
xgbc.fit(x,y)
y_testx = xgbc.predict(xtest)

