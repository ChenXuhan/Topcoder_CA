#coding=utf8
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats


def cumsum(arr):
    x = arr
    y = [float(i)/len(x) for i in range(1, len(x)+1)]
    return x, y


if __name__ == '__main__':
    tasks = pd.read_csv("../data/challenge_count.csv")
    print tasks.head()
    # fun = lambda x: if x.register_count<x.submit_count: x.register_count+=x.submit_count
    # users['register_count'].map(fun)
    print '检查出错项'
    for i in range(len(tasks.index)):
        if tasks.iloc[i, 1] < tasks.iloc[i, 2]:
            tasks.iloc[i, 1] += tasks.iloc[i, 2]
    tasks['submitRate'] = tasks['num_submit']/tasks['num_register']
    print tasks['submitRate'].describe()
    tasks = tasks.sort_values(by='submitRate')
    # sel_users = users[(users.register_count > 15) & (users.submitRate < 1.0)]
    tasks.to_csv("../data/challenge_count_withsubRate.csv")
    # tasks = tasks.sort_values(by='num_register')
    X = tasks.iloc[:, 1]
    Y = tasks.iloc[:, 3]
    plt.scatter(X, Y)
    plt.xlim(10, 100)
    plt.show()

    # X, Y = cumsum(tasks['submitRate'])
    # print X[X.values >= 1.0]
    # print len(X.values >= 1.0)
    # plt.plot(X, Y)
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    # plt.savefig("../output/summary/submitRate0(task).png")
    # plt.show()
    # # 绘制正态分布概率图，即检验数据是否符合正态分布。红线是标准正态分布。
    # # 本图属于长尾分布，比标准正态分布有更多的偏离数据
    # result = stats.probplot(X, fit=True, plot=plt)
    # print result
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    # plt.savefig("../output/summary/submitRate1(task).png")
    # plt.show()