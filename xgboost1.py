import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from operator import mod
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
mydata=pd.read_csv('4.csv')
ts=0.3
rs=13
df = mydata
xn =20
arr = ['x{}'.format(i+1) for i in range(xn)] #此为列表推导式，方便
myout1 = pd.DataFrame(columns=['xgb1', 'xgb2'])
myout2 = pd.DataFrame(columns=['lrb1', 'lrb2'])
myout3 = pd.DataFrame(columns=[ 'lr1', 'lr2'])
myout4 = pd.DataFrame(columns=[ 'ada1', 'ada2'])
myout5 = pd.DataFrame(columns=[ 'rf1', 'rf2'])



#'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24'
feature_name = arr

x, y = df[feature_name], df.y1
#smo = SMOTE(sampling_strategy=0.3, random_state=42)
#x, y = smo.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts, random_state=rs)

# 设置模型参数
xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(x_train, y_train)


y_pro = xgb_clf.predict_proba(x_test)[ : , 1]
y_pred = xgb_clf.predict(x_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
myout1.xgb1,myout1.xgb2,_ = precision_recall_curve(y_test, y_pro,pos_label=1)
#myout1.to_csv('xgb_prc_beijin.csv')
# 绘制特征重要性

print('xgboost:')
print("Accuracy:", accuracy)
print('roc_auc:', roc_auc_score(y_test, y_pro))
print('f1:', f1_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('RMSE:', metrics.mean_squared_error(y_test, y_pred)**0.5)
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))




import matplotlib.pyplot
def plot_roc_curve(y_true, y_score, style, name):
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
    plt.plot(fpr, tpr, color=style, linewidth=0.8, label=name)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(ncol=2)


def plot_prc_curve(y_true, y_score, style, name):
    """
    y_true:真实值
    y_score：预测概率
    """
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    precision, recall, _ = precision_recall_curve(y_true,y_score,pos_label=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('prc curve')
    plt.plot(recall, precision, color=style, linewidth=0.8, label=name)

    plt.legend(ncol=2)


plot_roc_curve(y_test, y_pro,  'g','xgb')




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
    grad = labels - y_pred
    hess = np.ones_like(labels)
    return grad, hess


class XGB:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=0, reg_lambda=1,
                 base_score=0.5, loss=log_loss_obj, leafset=None, LRset=None):
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
        self.leafset = []

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

    def recalltree(self, x, feature):
        if feature in x:
            return 1
        else:
            return 0

    def _build_tree(self, df, depth=1, temp=[]):
        df = df.copy()
        df['g'], df['h'] = self.loss(df.y_pred, df.y)
        G, H = df[['g', 'h']].sum()

        Gain_max = float('-inf')
        if df.shape[0] > self.min_samples_split and depth <= self.max_depth and df.y.nunique() > 1 :
            for feature in self.feature_names:
                thresholds = sorted(set(df[feature]))

                for thresh_value in thresholds[1:]:
                    left_instance = df[df[feature] < thresh_value]
                    right_instance = df[df[feature] >= thresh_value]
                    G_left, H_left = left_instance[['g', 'h']].sum()
                    G_right, H_right = right_instance[['g', 'h']].sum()

                    Gain = G_left ** 2 / (H_left + self.reg_lambda) + G_right ** 2 / \
                           (H_right + self.reg_lambda) - G ** 2 / (H + self.reg_lambda)
                    if Gain >= Gain_max and self.recalltree(temp, feature) == 0:
                        Gain_max = Gain
                        split_feature = feature
                        split_value = thresh_value
                        left_data = left_instance
                        right_data = right_instance
                    #print(feature,'Gain:',Gain,'G-Left',G_left,'H-left',H_left,'G-Right',G_right,'H-right',H_right,
                    #'----',thresh_value)
            # print(Gain_max, split_feature, split_value)
            if 'split_feature' in locals().keys():
                #elf.repeat.append((split_feature, copy.deepcopy(temp)))
                temp.append(split_feature)

            if 'left_data' in locals().keys():
                left = self._build_tree(left_data, depth + 1, copy.deepcopy(temp))
            if 'left_data' in locals().keys():
                right = self._build_tree(right_data, depth + 1, copy.deepcopy(temp))
            if 'split_feature' in locals().keys():
                return Node(split_feature=split_feature, split_value=split_value, left=left, right=right)

        return Node(leaf_value=-G / (H + self.reg_lambda) * self.learning_rate)

    def fit(self, x, y):
        y_pred = -np.log((1 / self.base_score) - 1)
        df = pd.DataFrame(x)
        df['y'] = y
        self.feature_names = df.columns.tolist()[:-1]

        for i in range(self.n_estimators):
            df['y_pred'] = y_pred
            tree = self._build_tree(df)
            #tree.visualize_tree().render('1.gv',view=True)
            data_weight = df[feature_name].apply(self._predict, tree=tree, axis=1)
            y_pred+= list(map(lambda x: x + self.learning_rate, data_weight))
            print("add", i, "tree")
            self.trees.append(tree)

    def predict(self, x, y):
        df = pd.DataFrame(x)
        y_predict = []
        for tree in self.trees:
            df['y_pred'] = y
            data_weight = df[feature_name].apply(
                self._predict, tree=tree, axis=1)
            df_tree = pd.DataFrame(x)
            df_tree['y'] = y
            df_tree['data_weight'] = data_weight

            t = df_tree.groupby('data_weight')
            tmp = dict(list(t))

            tname = [i for i, k in t]
            for i in range(len(tname)):
                lrout = LogisticRegression(max_iter=10000)
                tk = pd.DataFrame(tmp[tname[i]])
                if tk.y.nunique() > 1:
                    lrout.fit(tk[feature_name], tk.y)
                    self.leafset.append(lrout)
                    y_predict.append(lrout.predict_proba(df[feature_name])[:,1])
                else:
                    z = tk.y.unique()[0]
                    '''if int(z) == 1:
                        z=1
                        zx = 0
                    else:
                        z=0
                        zx = 1'''
                    y_nice = []
                    for l in range(df_tree.shape[0]):
                        if df_tree.iloc[l, xn+2] == tname[i]:
                            y_nice.append(z)
                        else:
                            y_nice.append(-1)
                    #print(y_nice)
                    lrout.fit(df_tree[feature_name], y_nice)
                    self.leafset.append(lrout)
                    y_predict.append(y_nice)

        return y_predict
        # return self.sigmoid_array(y_pred)

    def __repr__(self):
        return 'XGBClassifier(' + ', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')) + ')'


df = mydata

x = df[feature_name]

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = pd.DataFrame(x)
x.columns = feature_name
y = df.y1

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rs, test_size=ts)

model = XGB()

model.fit(x_train, y_train)
LRset = model.predict(x_train, y_train)
LRset = pd.DataFrame(list(zip(*LRset)))

ABC = AdaBoostClassifier(base_estimator=LogisticRegression(),algorithm="SAMME", n_estimators=100, learning_rate=0.1)
ABC.fit(LRset, y_train)
y_LR = []
for i in range(len(model.leafset)):
    y1 = model.leafset[i].predict(x_test)
    y_LR.append(y1)

testSet = pd.DataFrame(list(zip(*y_LR)))
y_pred = ABC.predict(testSet)
# model.trees[0].visualize_tree().render('2.gv', view=True)
y_pro=ABC.predict_proba(testSet)[:, 1]


myout2.lrb1,myout2.lrb2,_ = precision_recall_curve(y_test, y_pro,pos_label=1)
#myout2.to_csv('lrb_prc_my.csv')

print('accuracy:',accuracy_score(y_test, y_pred))
print('roc_auc:', roc_auc_score(y_test, y_pro))
print('f1:', f1_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('RMSE:', metrics.mean_squared_error(y_test, y_pred)**0.5)
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))



import matplotlib.pyplot



plot_roc_curve(y_test, y_pro,  'r', 'LRB')


df =mydata
x = df[feature_name]
y = df.y1
lr = LogisticRegression(max_iter=10000)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rs, test_size=ts)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pro= lr.predict_proba(x_test)[:, 1]


myout3.lr1, myout3.lr2,_ = precision_recall_curve(y_test, y_pro,pos_label=1)
#myout3.to_csv('lr_prc_my.csv')

print('LR:')
print('accuracy:',accuracy_score(y_test, y_pred))
print('roc_auc:', roc_auc_score(y_test, y_pro))
print('f1:', f1_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('RMSE:', metrics.mean_squared_error(y_test, y_pred)**0.5)
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))


plot_roc_curve(y_test, y_pro, 'y', 'LR')






from sklearn.ensemble import RandomForestClassifier

df = mydata
x = df[feature_name]
y = df.y1
#f rom sklearn.svm import LinearSVC
# svm_model = LinearSVC()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rs, test_size=ts)
ABC1 = AdaBoostClassifier(algorithm="SAMME", n_estimators=1000, learning_rate=0.01)
ABC1.fit(x_train, y_train)

y_pred = ABC1.predict(x_test)
y_pro = ABC1.predict_proba(x_test)[:, 1]

myout4.ada1,myout4.ada2,_ = precision_recall_curve(y_test, y_pro,pos_label=1)
#myout4.to_csv('ada_prc_my.csv')

print('adaboost:')

print('accuracy:',accuracy_score(y_test, y_pred))
print('roc_auc:', roc_auc_score(y_test, y_pro))
print('f1:', f1_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('RMSE:', metrics.mean_squared_error(y_test, y_pred)**0.5)
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))

plot_roc_curve(y_test, y_pro,'k', 'ada')



from sklearn.ensemble import RandomForestClassifier

df = mydata
x = df[feature_name]
y = df.y1
#f rom sklearn.svm import LinearSVC
# svm_model = LinearSVC()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rs, test_size=ts)
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pro= rfc.predict_proba(x_test)[:, 1]
y_pred = rfc.predict(x_test)

myout5.rf1,myout5.rf2,_ = precision_recall_curve(y_test, y_pro,pos_label=1)
#myout5.to_csv('rf_prc_my.csv')
print('RF:')

print('accuracy:',accuracy_score(y_test, y_pred))
print('roc_auc:', roc_auc_score(y_test, y_pro))
print('f1:', f1_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('RMSE:', metrics.mean_squared_error(y_test, y_pred)**0.5)
print('MAE:',metrics.mean_absolute_error(y_test, y_pred))

plot_roc_curve(y_test,y_pro, 'b', 'RF')
matplotlib.pyplot.show()

