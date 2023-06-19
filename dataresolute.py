import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from imblearn.over_sampling import SMOTE

mydata=pd.read_csv('beijin2.csv')
ts=0.3
rs=13
df = mydata
xn =42
arr = ['x{}'.format(i+1) for i in range(xn)] #此为列表推导式，方便

feature_name = arr

x, y = df[feature_name], df.y1

smo = SMOTE(sampling_strategy=0.5, random_state=42)
x, y = smo.fit_resample(x, y)

print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts, random_state=rs)




#rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, n_jobs=-1)
#rfecv = RFECV(estimator=rf, cv=5)
#selector = rfecv.fit(x_train, y_train)


#print('RFECV 选择出的特征个数 ：', rfecv.n_features_)  # RFECV选择的特征个数
#print('特征优先级 ： ', rfecv.ranking_)        # 1代表选择的特征`


