import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

df = pd.read_csv(r"C:\Users\bhavya vegad\Desktop\vs project\ML\SUPERVISED\fake news\news.csv")

print(df)

print(df.head(5))

print(df.tail(5))

print(df.info())

print(df.columns)

print(df.corr(numeric_only=True))

print(df.cov(numeric_only=True))

print(df.dtypes)

print(df.describe())

print(df.isnull().sum())

print(df.dropna(how='any', inplace = True))
print(df.isnull().sum())

encoder = LabelEncoder()

df["title"] = encoder.fit_transform(df["title"])
df["text"] = encoder.fit_transform(df["text"])
df["label"] = encoder.fit_transform(df["label"])

#print(df.drop(['Unnamed: 0','title'], axis = 1, inplace = True))


print(df.dtypes)

df[(df >= 0).all(axis=1)]

x = df.drop(['label'],axis=1)
y = df["label"]

print(x.head(10))

print(y.head(10))

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=42)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

model = LogisticRegression ()

model.fit(x_train,y_train)

tranning_data_prediction = model.predict(x_train)

y_train = y_train[tranning_data_prediction >= 0]
tranning_data_prediction = tranning_data_prediction[tranning_data_prediction >= 0]

error_score = metrics.r2_score(y_train,tranning_data_prediction)*100
print("R squred error :", error_score)

plt.scatter(y_train, tranning_data_prediction)
plt.grid()
plt.show()


print(model.predict([[8476,6155,1514]]))

print(model.predict([[10294,5747,2185]]))

print(model.predict([[3608,2946,5165]]))

print(model.predict([[10142,653,5991]]))

print(model.predict([[875,4788,2733]]))

print(model.predict([[6903,4742,117]]))

print(model.predict([[7341,2054,4168]]))

