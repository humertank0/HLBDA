'''
Hyper Learning Binary Dragonfly Algorithm source code demo version
DOI: https://doi.org/10.1016/j.knosys.2020.106553
'''

'''
导入相应的包
'''
from numpy.matlib import repmat
import pandas as pd
import numpy as np
import random
import math
from sklearn.preprocessing import MinMaxScaler
from fitness_function import *
from initial_population import *
from filter import reliefFScore,top_select
import time
from KneePointDivideData import findKneePoint
'''
读入数据
'''

# 读入待分类数据,赋值给feat
featTemp = np.loadtxt("..//datasets//BreastCancer1(96,24482).csv",delimiter=',',encoding='utf-8-sig')

# print(featTemp.info())
# 读入分类标签,赋值给label
# label = np.loadtxt("D:\OneDrive\post graduate\Datasets\dataset\ionosphere_351_34.2\iono_label.csv",delimiter=',',encoding='utf-8-sig')
feat=featTemp[:,:-1]
label=featTemp[:,-1]



'''
数据归一化
'''
stand = MinMaxScaler()
feat = stand.fit_transform(featTemp)


#如果是高维数据集的话，用reliefF提前降一下维度
if(feat.shape[1]>=7000):
    Score = reliefFScore(feat, label)
    # ratio=findKneePoint(Score)
    ratio=0.03
    top_index = top_select(Score,ratio)
    feat = feat[:, top_index]

'''
设置参数
'''
maxIteration = 100  # 最大迭代次数
N = 10  # 蜻蜓的个数
D = feat.shape[1]  # 特征数及维数
pl = 0.4  # 个体学习率
gl = 0.7  # 种群学习率



'''
HLBDA
'''
# 算法开始
start = time.perf_counter()#开始计时
print('开始进行HLBDA优化：')

# 种群初始化，得到初始的蜻蜓位置(蜻蜓数量,初始维度)
X = initial_population(N, D)
# print('初始位置的矩阵形状为:', X.shape)

# 位移变量的初始化(蜻蜓数量,初始维度)
DX = np.zeros((N, D))

# 适应度值的初始化(存储每个维度的适应度值),D个变量的一维数组
fit = np.zeros((1, D))

# 食物源初始化
fitF = np.inf

# 捕食者源初始化
fitE = -np.inf

# 存储每次迭代的准确率
curve = np.zeros(maxIteration)

# 初始迭代次数
t = 1

#
Xnew = np.zeros((N, D))
Dmax = 6

# 个体和群体
fitPB = np.ones((1, N))
fitPW = np.zeros((1, N))

# 初始化个体最优位置和最差位置(天敌)
Xpb = np.zeros((N, D))
Xpw = np.zeros((N, D))

# 开始迭代
if(D<N):
    print('蜻蜓数量太多了')
    exit()
while t <= maxIteration:
    for i in range(N):  # 每只蜻蜓进行一次计算,接近食物源，远离天敌，更新Xpb、Xpw、Xf、Xe、fitPB、fitPW
        fit[0, i] = fitness_function(feat, label, X[i, :])
        if fit[0, i] < fitF:
            fitF = fit[0, i]
            Xf = X[i, :]
        if fit[0, i] > fitE:
            fitE = fit[0, i]
            Xe = X[i, :]
        if fit[0, i] > fitPW[0, i]:
            fitPW[0, i] = fit[0, i]
            Xpw[i, :] = X[i, :]
        if fit[0, i] < fitPB[0, i]:
            fitPB[0, i] = fit[0, i]
            Xpb[i, :] = X[i, :]
    w = 0.9 - t * ((0.9 - 0.4) / maxIteration)
    rate = 0.1 - t * ((0.1 - 0) / (maxIteration / 2))
    s = 2 * random.random() * rate
    a = 2 * random.random() * rate
    c = 2 * random.random() * rate
    f = 2 * random.random()
    e = rate
    for i in range(N):  # 根据公式计算每只蜻蜓的位置
        index = 0
        nNeighbor = 1
        Xn = np.zeros((1, D))
        DXn = np.zeros((1, D))
        for j in range(N):
            if i != j:
                DXn = np.r_[DXn, [DX[j, :]]]
                #                DXn.r_[DXn,[DX[j,:]]]
                Xn = np.r_[Xn, [X[j, :]]]
                #                Xn.r_[Xn,[X[j,:]]]
                index = index + 1
                nNeighbor = nNeighbor + 1
        S = repmat(X[i, :], nNeighbor, 1) - Xn
        S = -sum(S, 1)
        A = sum(DXn, 1) / nNeighbor
        C = sum(Xn, 1) / nNeighbor
        C = C - X[i, :]
        F = ((Xpb[i, :] - X[i, :]) + (Xf - X[i, :])) / 2
        E = ((Xpw[i, :] + X[i, :]) + (Xe + X[i, :])) / 2
        for d in range(D):
            dX = (s * S[d] + a * A[d] + c * C[d] + f * F[d] + e * E[d]) + w * DX[i, d]
            #            dX(dX > Dmax) = Dmax;
            if dX > Dmax:
                dX = Dmax
            if dX < -Dmax:
                dX = -Dmax
            #            dX(dX < -Dmax) = -Dmax;
            DX[i, d] = dX
            TF = abs(DX[i, d] / math.sqrt((((DX[i, d]) ** 2) + 1)))
            r1 = random.random()
            if r1 >= 0 and r1 < pl:
                if random.random() < TF:
                    Xnew[i, d] = 1 - X[i, d]
                else:
                    Xnew[i, d] = X[i, d]
            elif r1 >= pl and r1 < gl:
                Xnew[i, d] = Xpb[i, d]
            else:
                Xnew[i, d] = Xf[d]
    X = Xnew
    curve[t - 1] = fitF
    print('\nIteration', t, ': min(HLBDA)= ', (1-curve[t - 1])*100,'%')
    t = t + 1

# Pos = np.arange(1, 35, 1)
PosIndex = []
for i in range(N):
    if Xf[i] == 1:
        PosIndex.append(i)
sFeat = feat[:, PosIndex]
Nf = len(PosIndex)
print('选择的特征数量为：', Nf)

acc = 1 - curve[maxIteration - 1]
print('数据的特征矩阵为：', feat.shape)
print('初始特征数量为：', D)
print('准确率为：', acc * 100, '%')
end = time.perf_counter()
print('程序运行总时间为:','{:.16f}'.format(end-start),'秒')