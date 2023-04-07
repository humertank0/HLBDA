from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import numpy as np
# 适应度函数fitness_funtion,返回值为计算好的适应值cost
def fitness_function(feat, label, X):
    # 设置alpha、beta
    alpha = 0.99
    beta = 0.01
    # 求出原始特征数量maxFeat
    maxFeat = feat.shape[1]
    # 求出feat经过选择特征后得到的的result
    xIndex = []
    for i in range(len(X)):
        if X[i] == 1:
            xIndex.append(i)

    result = feat[:, xIndex]
    # 求错误率，使用KNN算法进行十倍交叉验证
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(result, label)
    acc = cross_val_score(knn, result, label, cv=10)
    #    print('acc:',acc)
    err = 1 - np.mean(acc)
    #    print('err:',err)
    # 选择的特征数量为Nsf个
    Nsf = sum(X == 1)
    # print('Features,err:',Nsf,err)
    # 代价函数
    cost = alpha * err + beta * (Nsf / maxFeat)
    return cost