import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from abupy import AbuML
train_data = pd.read_csv("../data/titanic/train.csv")
scaler = preprocessing.StandardScaler()
#train_data.info()
#
# train_data.groupby('Survived').count()
# print(train_data.head(3))

# print(train_data.Age.dropna().mean())
train_data.loc[(train_data.Age.isnull()), 'Age'] = train_data.Age.dropna().mean()
# print(train_data.loc[(train_data.Age.isnull()), 'Age'])
# 年龄归一化
# print(train_data.Age.reshape(-1,1))
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1,1))
# 票价归一化
train_data['Fare_scaled'] = scaler.fit_transform(train_data['Fare'].values.reshape(-1,1))

# print(train_data['Age_scaled'])

#等级 dummies
dummies_pclass = pd.get_dummies(train_data['Pclass'], prefix='Pclass')
# print(train_data.groupby('Survived').count())

#登船舱口 dummies
dummies_embarked = pd.get_dummies(train_data['Embarked'], prefix='Embarked')

dummies_sex = pd.get_dummies(train_data['Sex'], prefix='Sex')

result_data = pd.concat([train_data, dummies_sex, dummies_embarked, dummies_pclass], axis=1)
result_data.drop(['Name', 'Age', 'Fare', 'Ticket', 'Sex', 'Cabin', 'Embarked', 'Pclass'], axis=1, inplace=True)
train_feature = result_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
matrix_result = train_feature.as_matrix()
print('-------------------------------------------------------')
print(train_feature.head(1))
y = matrix_result[:, 0]
x = matrix_result[:, 1:]
print(x[0])

titanic = AbuML(x, y, train_feature)
titanic.estimator.logistic_classifier()

titanic.cross_val_accuracy_score()

titanic.plot_learning_curve()
print(titanic.importances_coef_pd())
titanic.plot_confusion_matrices()


