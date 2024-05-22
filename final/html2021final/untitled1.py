import pandas as pd
y = pd.read_csv('sss.csv')
print(y.groupby('Churn Category').size())
