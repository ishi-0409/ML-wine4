import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('winequality-red.csv',sep=';')

x = df.drop('quality',axis=1)
y = df['quality']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# 訓練データとテストデータをx,yそれぞれ分割する

ss = StandardScaler()
ss.fit(x_train)
x_train_std = ss.transform(x_train)
x_test_std = ss.transform(x_test)
# fitにおいては訓練データのみ
# 標準化はどちらも行う

model = LinearRegression()
model.fit(x_train_std,y_train)
print('回帰係数:',model.coef_)