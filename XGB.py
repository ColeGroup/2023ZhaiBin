import numpy as np
import pandas as pd
import xgboost as xgb
from graphviz import Digraph
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import copy
from pandas import MultiIndex, Int64Index
from imblearn.over_sampling import SMOTE
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt

import os
os.environ["PATH"] += os.pathsep + 'C:/Users/zb/PycharmProjects/zb/venv/Lib/site-packages/graphviz/bin'''


class Node(object):
    """结点
       leaf_value ： 记录叶子结点值
       split_feature ：特征i
       split_value ： 特征i的值
       left ： 左子树
       right ： 右子树
    """
 
    def __init__(self, leaf_value=None, split_feature=None, split_value=None, left=None, right=None):
        self.leaf_value = leaf_value
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
 
    def show(self):
        print(
            f'weight: {self.leaf_value}, split_feature: {self.split_feature}, split_value: {self.split_value}.')
 
    def visualize_tree(self):
        """
        递归查找绘制树
        """
 
        def add_nodes_edges(self, dot=None):
            if dot is None:
                dot = Digraph()
                dot.node(name=str(self),
                         label=f'{self.split_feature}<{self.split_value}')
            # Add nodes
            if self.left:
                if self.left.leaf_value:
                    dot.node(name=str(self.left),
                             label=f'leaf={self.left.leaf_value:.10f}')
                else:
                    dot.node(name=str(self.left),
                             label=f'{self.left.split_feature}<{self.left.split_value}')
                dot.edge(str(self), str(self.left))
                dot = add_nodes_edges(self.left, dot=dot)
            if self.right:
                if self.right.leaf_value:
                    dot.node(name=str(self.right),
                             label=f'leaf={self.right.leaf_value:.10f}')
                else:
                    dot.node(name=str(self.right),
                             label=f'{self.right.split_feature}<{self.right.split_value}')
                dot.edge(str(self), str(self.right))
                dot = add_nodes_edges(self.right, dot=dot)
            return dot
 
        dot = add_nodes_edges(self)

        return dot
 
 
def log_loss_obj(preds, labels):
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess
 
 
def mse_obj(preds, labels):
    grad = labels-y_pred
    hess = np.ones_like(labels)
    return grad, hess
 
 
class XGB:
    def __init__(self, n_estimators=2, learning_rate=0.1, max_depth=3, min_samples_split=0, reg_lambda=1, base_score=0.5, loss=log_loss_obj):
        # 学习控制参数
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_score = base_score
        # 树参数
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.reg_lambda = reg_lambda
 
        self.trees = []
        self.feature_names = None
 
    def sigmoid_array(self, x):
        return 1 / (1 + np.exp(-x))
 
    def _predict(self, x, tree):
        # 循环终止条件：叶节点
        if tree.leaf_value is not None:
            return tree.leaf_value
        if x[tree.split_feature] < tree.split_value:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)
 
    def _build_tree(self, df, depth=1):
        df = df.copy()
        df['g'], df['h'] = self.loss(df.y_pred, df.y)
        G, H = df[['g', 'h']].sum()

        Gain_max = 0
        if df.shape[0] > self.min_samples_split and depth <= self.max_depth and df.y.nunique() > 1:
            for feature in self.feature_names:
                thresholds = sorted(set(df[feature]))
                for thresh_value in thresholds[1:]:
                    left_instance = df[df[feature] < thresh_value]
                    right_instance = df[df[feature] >= thresh_value]
                    G_left, H_left = left_instance[['g', 'h']].sum()
                    G_right, H_right = right_instance[['g', 'h']].sum()
 
                    Gain = G_left**2/(H_left+self.reg_lambda)+G_right**2 / \
                        (H_right+self.reg_lambda)-G**2/(H+self.reg_lambda)
                    Gain = Gain / 2
                    if Gain > Gain_max:
                        Gain_max = Gain
                        split_feature = feature
                        split_value = thresh_value
                        left_data = left_instance
                        right_data = right_instance
                    #print(feature,'Gain:',Gain,'G-Left',G_left,'H-left',H_left,'G-Right',G_right,'H-right',H_right,'----',thresh_value)
            print(Gain_max, split_feature, split_value)
 
            left = self._build_tree(left_data,  depth+1)
            right = self._build_tree(right_data,  depth+1)
            return Node(split_feature=split_feature, split_value=split_value, left=left, right=right)
        return Node(leaf_value=-G/(H+self.reg_lambda)*self.learning_rate)
 
    def fit(self, X, y):
        y_pred = -np.log((1/self.base_score)-1)
        df = pd.DataFrame(X)
        df['y'] = y
        self.feature_names = df.columns.tolist()[:-1]
 
        for i in range(self.n_estimators):
            df['y_pred'] = y_pred
            tree = self._build_tree(df)
            data_weight = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17',
        'x18', 'x19', 'x20']].apply(
                self._predict, tree=tree, axis=1)
            y_pred += data_weight
            self.trees.append(tree)
 
    def predict(self, X):
        df = pd.DataFrame(X)
        y_pred = -np.log((1/self.base_score)-1)
        for tree in self.trees:
            df['y_pred'] = y_pred
            data_weight = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17',
        'x18', 'x19', 'x20']].apply(
                self._predict, tree=tree, axis=1)
            y_pred += data_weight
        return self.sigmoid_array(y_pred)
 
    def __repr__(self):
        return 'XGBClassifier('+', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_'))+')'

if __name__  == "__main__":
    None

df = pd.read_csv('4.csv')
x = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17',
        'x18', 'x19', 'x20']]
y = df.y
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, random_state=13,test_size=0.3)
xgb1 = xgb(n_estimators=15, max_depth=3, reg_lambda=1, min_child_weight=1,objective = 'reg:linear', learning_rate=0.1)
xgb1.fit(x_train, y_train)
yp = xgb1.predict_proba(x_test)
plot_tree(xgb1, num_trees=0)
plt.show()

model = XGB()
model.fit(x_train, y_train)

output = model.predict(x_test)

import matplotlib.pyplot
def plot_roc_curve(y_true, y_score):
    """
    y_true:真实值
    y_score：预测概率
    """
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    fpr,tpr,threshold = roc_curve(y_true, y_score, pos_label=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title('roc curve')
    plt.plot(fpr,tpr,color='b',linewidth=0.8)
    plt.plot([0,1], [0,1], 'r--')


plot_roc_curve(y_test, y_pred)
matplotlib.pyplot.show()
model.trees[0].visualize_tree().render('2.gv', view=True)
#print(accuracy_score(output, y_test))





