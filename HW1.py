import numpy as np
import pandas as pd

#Q1
print("np version is", np.__version__)

#Q2
print("pd version is", pd.__version__)

pd.set_option('display.max_columns', None)
df = pd.read_csv('data.csv')
print(df.head())
print(df)

#Q3
df_BMW = df[
    df['Make'] == 'BMW'
]

print('Average price of the BMW:', df_BMW['MSRP'].mean())

#Q4
df_2015 = df[
    df['Year'] >= 2015
]

print(df_2015['Engine HP'].isnull().sum())

#Q5
mean_hp_before = df['Engine HP'].mean()
print('Engine HP with NA values:', round(mean_hp_before, 2))

df['Engine HP fillna'] = df['Engine HP'].fillna(df['Engine HP'].mean())
mean_hp_after = df['Engine HP fillna'].mean()
print('Engine HP without NA values:', round(mean_hp_after, 2))

#Q6
X = np.array(df[df.Make == "Rolls-Royce"][["Engine HP", "Engine Cylinders", "highway MPG"]].drop_duplicates())
print(X)
print(X.shape)

XTX = X.T.dot(X)
print(XTX)

XTX_invert = np.linalg.inv(XTX)
print(XTX_invert.sum())

#Q7
y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
w = (XTX_invert.dot(X.T)).dot(y)
print(w[0])