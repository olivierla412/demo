### DATA preparation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics


# Import data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

df_test = test.drop(['RowNumber',"PassengerId","Name","Ticket","Age"],axis=1)
df_train = train.drop(['RowNumber',"PassengerId","Name","Ticket","Age"],axis=1)

x = df_train.drop(['Survived'],axis=1)
y = df_train['Survived']


#Checking null percent of Train
null_perc = x.isnull().sum()/len(x)*100
null_perc.sort_values(ascending = False).head(10)

null_perc = df_test.isnull().sum()/len(df_test)*100
null_perc.sort_values(ascending = False).head(10)

x1 = x.drop(['Cabin'],axis=1)
df_test = df_test.drop(['Cabin'],axis=1)


# FILL
x1['Embarked']=x1['Embarked'].fillna(x1['Embarked'].mode()[0])
df_test['Embarked']=df_test['Embarked'].fillna(df_test['Embarked'].mode()[0])


# VERIFY with HEATMAP
sns.heatmap(x1.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


x1 = x1.values

# Encode categorical data and scale continuous data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
x1[:, 1] = labelencoder_1.fit_transform(x1[:, 1])
labelencoder_2 = LabelEncoder()
x1[:, 5] = labelencoder_2.fit_transform(x1[:, 5])
onehotencoder = OneHotEncoder(categorical_features=[1])
x1 = onehotencoder.fit_transform(x1).toarray()
x1 = x1[:, 1:]


#Checking the correlation with respect to 'Survived'
x1.columns
train.corr()['Survived']
train.corr()['Pclass']
train.corr()['Sex']
train.corr()['SibSp']
train.corr()['Parch']
train.corr()['Fare']
train.corr()['Embarked']


# Split data in train/test
x_train = x1
y_train = y
x_test = df_test

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2)


# modelling

# Logistic model
from sklearn.linear_model import LinearRegression
logic = LinearRegression()
logic.fit(x_train, y_train)
# Predicting the Test set results
y_pred = logic.predict(x_test)
y_pred = (y_pred > 0.5)


# Support vector classifier
from sklearn.svm import SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
#prediction
y_pred = linear_svc.predict(x_test)


metrics.accuracy_score(y_test, y_pred)
metrics.precision_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred)
metrics.classification_report(y_test, y_pred)



# otpimization

krn = ['linear','poly','rbf','sigmoid']
c = np.arange(1,10,52)
degree = np.arange(3,8)
coef =  np.arange(0.001,0.5,10)
gam = ['auto','scale']

#function
best_score = 0
for i in krn:
    for j in c:
        for k in degree:
            for x in coef:
                for z in gam:
                    linear_svc = SVC(kernel=i, C=j, degree=k, coef0=x, gamma=z)
                    linear_svc.fit(x_train.astype(float), y_train.astype(float))
                    #prediction 
                    y_pred = linear_svc.predict(x_test)
                    Accuracy = accuracy_score(y_test, y_pred)
                    if best_score < Accuracy:
                        bi=i
                        bj=j
                        bk=k
                        bx=x
                        bz=z
print(best_score,bi,bj,bk,bx,bz)





