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
    users = pd.read_csv("../data/user_count.csv")
    print users.head()
    # fun = lambda x: if x.register_count<x.submit_count: x.register_count+=x.submit_count
    # users['register_count'].map(fun)
    print '检查出错项'
    for i in range(len(users.index)):
        if users.iloc[i, 1] < users.iloc[i, 2]:
            users.iloc[i, 1] += users.iloc[i, 2]
    users['submitRate'] = users['submit_count']/users['register_count']
    print users['submitRate'].describe()
    users = users.sort_values(by='submitRate')
    sel_users = users[(users.register_count > 15) & (users.submitRate < 1.0)]
    sel_users.to_csv("../data/user_count_withsubRate.csv")
    X, Y = cumsum(sel_users['submitRate'])
    print X[X.values >= 1.0]
    print len(X.values >= 1.0)
    plt.plot(X, Y, '-', linewidth=2)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    # sns.distplot(X, kde=True, rug=True)
    plt.savefig("../output/summary/submitRate0.png")
    plt.show()
    print X.describe()
    # 绘制正态分布概率图，即检验数据是否符合正态分布。红线是标准正态分布。
    # 本图属于长尾分布，比标准正态分布有更多的偏离数据
    result = stats.probplot(X, fit=True, plot=plt)
    print result
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.savefig("../output/summary/submitRate1.png")
    plt.show()