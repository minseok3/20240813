import matplotlib.pyplot as plt
import  pandas as pd
import matplotlib.pyplot as plt
from  sklearn.preprocessing import MinMaxScaler
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from  sklearn.model_selection import KFold
from  sklearn.model_selection import cross_val_score
header = ['preg','plas', 'pres','skin','test', 'mass','pedi','age','class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv',names=header)

array = data.values
x = array[:, 0:8]
y = array[:, 8]
scalar  = MinMaxScaler(feature_range=(0,1))
rescaled_x = scalar.fit_transform(x)

# 데이터 분할
x_train,x_test,y_train,y_test = train_test_split(rescaled_x,y ,test_size=0.2)

# 모델 선택 및 학습
model = LogisticRegression()


fold = KFold(n_splits=10,shuffle=True)
acc = cross_val_score(model,rescaled_x,y, cv= fold, scoring= "accuracy")

sum_acc = sum(acc)/len(acc)
print(sum_acc)
