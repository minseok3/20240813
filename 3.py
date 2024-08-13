import matplotlib.pyplot as plt
import  pandas as pd
import matplotlib.pyplot as plt
from  sklearn.preprocessing import MinMaxScaler
from  sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from  sklearn.metrics import accuracy_score

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
model = DecisionTreeClassifier(max_depth=100, min_samples_split=50, )
model.fit(x_train,y_train)
# 예측갑생성
y_pred = model.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# 예측 정확도 확인
acc = accuracy_score(y_pred_binary,y_test)
print(acc)
# test값 저장 밒 ㅛ 예측값 저장
df_y_test = pd.DataFrame(y_test)
df_y_pred_binary = pd.DataFrame(y_pred_binary)
# df_y_test.to_csv("./results/y_test.csv")
# df_y_pred_binary.to_csv('./results/y_pred.csv')

# 결과값 예측
plt.figure(figsize= (10,6))
plt.scatter(range(len(y_test)), y_test, color= 'blue', label= 'Actual Values', marker='o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='Predicted', marker= 'x')

plt.title("comparaison of Actual and Predicted Values")
plt.xlabel("index")
plt.ylabel("class(0 or 1)")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)