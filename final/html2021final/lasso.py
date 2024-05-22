import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from sklearn.metrics import f1_score

train_dataX = pd.read_csv('x_in.csv')
train_dataX = train_dataX.drop(labels='Customer ID',axis=1)
train_dataY = pd.read_csv('y_in.csv')
train_dataY = train_dataY.drop(labels='Customer ID',axis=1)
test_dataX = pd.read_csv('x_out.csv')
test_dataX = test_dataX.drop(labels='Customer ID',axis=1)

std = StandardScaler()
std.fit(train_dataX.values)
X_scaled = std.transform(train_dataX.values)

x_res , y_res = SMOTE().fit_resample(train_dataX, train_dataY)
alphavec = 10**np.linspace(-3,3,200) 
lasso_model = LassoCV(alphas = alphavec, cv=5)
rfc = RandomForestClassifier(n_estimators=2000,min_samples_leaf = 21,n_jobs=-1,oob_score=(True),random_state=(16))
model = rfc.fit(x_res,y_res)
(pd.Series(model.feature_importances_, index=train_dataX.columns)
   .nlargest(47)  
   .plot(kind='barh', figsize=[20,15])
    .invert_yaxis())
plt.yticks(size=15)
plt.title('Top Features derived by Random Forest', size=20)

model =RFE(lasso_model, n_features_to_select=17)
X_new = model.fit_transform(x_res,y_res)
testx = model.transform(test_dataX)

x_train , x_test ,y_train,y_test = train_test_split(X_new,y_res,test_size=0.3,random_state=23)
rfc = RandomForestClassifier(n_estimators = 8888,min_samples_split = 14,min_samples_leaf = 51,n_jobs=-1,oob_score=(True))
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)
testy = rfc.predict(testx)

print(rfc.score(x_train,y_train))
print(rfc.score(x_test,y_test))

print('f1 score ',f1_score(y_test, y_pred,average= 'macro'))

y = pd.DataFrame(testy,columns =['Churn Category'])
#y.to_csv('sss8888.csv')
print(y.groupby('Churn Category').size())
#0.34406 in public
#0.33458 in private