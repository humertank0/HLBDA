import numpy as np
import random


# 初始化蜻蜓的位置X的函数initial_population(二进制形式),阈值为0.5，返回值为0、1的numpy
def initial_population(Number, Dimension):
    # temp = np.zeros((Number, Dimension))
    # for r in range(Number):
    #     for c in range(Dimension):
    #         if random.random() >= 0.5:
    #             temp[r, c] = 1
    temp = np.random.rand(Number, Dimension)
    for r in range(Number):
        for c in range(Dimension):
            if temp[r, c] >= 0.8:
                temp[r, c] = 1
            else:
                temp[r, c] = 0
    return temp
