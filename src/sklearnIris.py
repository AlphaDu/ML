from sklearn import datasets, svm
iris = datasets.load_iris()
digits = datasets.load_digits()
from sklearn.externals import joblib
import pickle
import matplotlib.pyplot as plt
import numpy as np

# print(digits.data[8])
# print(len(digits.target))

# print(digits.images[0])
from sklearn import svm

# 支持向量分类器
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data, digits.target)

print(clf.predict(digits.data[-1:]))
print(digits.data[-1:])
print(digits.target)

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(digits.data[0:1])
digits.target[0]
joblib.dump(clf, 'filename.pkl')
