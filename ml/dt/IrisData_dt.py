# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


import matplotlib.pyplot as plt
from pylab import matplotlib,mpl
from matplotlib.font_manager import FontManager

import numpy as np
#分隔训练和测试数据
from sklearn.model_selection import train_test_split

class IrisData_dt:

    def load_data(self):
        iris = datasets.load_iris()
        #print iris
        X,y = iris['data'], iris['target']
        #print(list(y))
        #print(X,y)
        return X,y


    def process_data(self, X, y):
        #数据归一化处理
        X = preprocessing.scale(X)
        #print(X)
        #test_size 是测试比例；random_state伪随机的启动至; 240/600
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        #print(X_train.size, X_test.size)
        return X_train, X_test, y_train, y_test



    def train_decision_tree_model(self, X_train, X_test, y_train, y_test):
        #开始训练
        clf = DecisionTreeClassifier(random_state=0)
        clf.criterion = 'gini'
        print("decision tree criterion:",clf.criterion)
        clf.fit(X_train,y_train)
        print("total features:",clf.n_features_)
        print("total train samples:",len(X_train))
        print("total classes:",clf.n_classes_)

        print("feature_importances:",clf.feature_importances_)
        #测试模型分类的准确率96%左右
        print("score of test data:",clf.score(X_test, y_test))
        return clf


    def predict_one_sample(self, model : DecisionTreeClassifier , inputX):
        y_ = model.predict(inputX)
        print(model.n_classes_)
        print(y_)

    def visual_data(self, X_train, X_test, y_train, y_test):
        from matplotlib.colors import ListedColormap
        myfont = matplotlib.font_manager.FontProperties(
            fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')


        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.subplot(121)
        plt.title("train data")
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        plt.subplot(122)
        plt.title("test data")
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright)
        plt.show()




if __name__ == '__main__':
    dt_sk = IrisData_dt()
    data = dt_sk.load_data()
    print("X shape:",data[0].shape,"y shape:", data[1].shape)

    X_train, X_test, y_train, y_test = dt_sk.process_data(data[0], data[1])
    dt_sk.visual_data(X_train, X_test, y_train, y_test)
    model = dt_sk.train_decision_tree_model(X_train, X_test, y_train, y_test)
    sample = []
    sample.append(X_test[0])
    print(model.predict(sample))


