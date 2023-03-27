# SVM - Support Vector Machine

import sklearn
from sklearn import datasets
from sklearn import svm, metrics

data = datasets.load_breast_cancer()

X = data.data # assigning the attributes to X
y = data.target # assigning the target to y

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

svc = svm.SVC(kernel='linear',) # loading (linear) Support Vector Classifier 
svc.fit(x_train,y_train)

y_predict = svc.predict(x_test) # predicting the targets with values of x_test

acc = metrics.accuracy_score(y_test, y_predict) # matching the test results with predicted results
print(acc)
