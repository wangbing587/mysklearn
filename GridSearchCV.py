#!/usr/bin/env python
# -*- demo: utf-8 -*-
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris


def knn_iris():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

    std = StandardScaler()
    x_train_std = std.fit_transform(x_train)
    x_test_std = std.transform(x_test)

    knn = KNeighborsClassifier()
    param = {"n_neighbors": [1, 3, 5, 7, 9]}

    gc = GridSearchCV(knn, param_grid=param, cv=3)
    gc.fit(x_train_std, y_train)

    print("每个超参数每次交叉验证的结果：\n", gc.cv_results_)
    print("每个测试集上的准确率：\n", gc.score(x_test_std, y_test))
    print("在交叉验证当中最好的结果：\n", gc.best_score_)
    print("选择最好的模型是：\n", gc.best_estimator_)



if __name__ == '__main__':
    knn_iris()

