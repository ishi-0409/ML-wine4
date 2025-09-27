import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('winequality-red.csv',sep=';')

x = df.drop('quality',axis=1)
y = df['quality']

des = x.describe()
# 説明変数の統計量

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# 訓練データとテストデータを分割する
# 比率は訓練:テスト = 4:1 で分けている

ss = StandardScaler()
ss.fit(x_train)
x_train_std = ss.transform(x_train)
x_test_std = ss.transform(x_test)
# 訓練データだけ標準偏差を計算し、標準化を訓練、テストデータに対して行う
#　モデルはテストデータの標準偏差を知らない


model = LinearRegression()
model.fit(x_train_std,y_train)
print('回帰係数:',model.coef_)



