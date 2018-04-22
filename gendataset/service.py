#coding=utf8
import numpy as np
import pandas as pd
import time
import json
import pickle
from datetime import datetime
from scipy.spatial.distance import pdist


def current_time():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def str_to_list(x):
    if type(x) == str:
        if x:
            return x.split(',')
    return []


def count_day(x):
    if type(x) == str:
        timeTuple = datetime.strptime(x[:10], "%Y-%m-%d")
        time0 = datetime.strptime("2006-10-01", "%Y-%m-%d")
        return (timeTuple-time0).days
    return 0


def dist(a, b):
    if a > b:
        d = a-b
    else:
        d = b-a
    return 1/(1+d)


def match(vec0, vec1):
    if vec0.sum() == 0 and vec1.sum() == 0:
        return 1
    if vec0.sum() == 0 or vec1.sum() == 0:
        return 0
    d = 1 - pdist([vec0, vec1], 'jaccard')
    return d[0]


class Service:
    def __init__(self, path="../output/dataset1/", regfile="registrant.data", simfile="sim_tasks_365.data"):
        self.path = path
        self.reg = pd.read_pickle(path+regfile)
        self.sub = pd.read_pickle(path+"submission.data")
        self.tasks = pd.read_pickle(path+"challenges.data")
        with open(path+simfile, "rb")as f:
            self.sim_tasks = pickle.load(f)

    def get_user_perf(self, task, user, time0, days):
        '''
        距离时间time0，在30天之内的，用户user在任务task的相似任务中
        '''
        open_tasks = self.tasks[(self.tasks["RegStart"] <= time0) & (self.tasks["SubEnd"] >= time0)].index
        reg_task0 = self.reg.loc[user]
        reg_task1 = reg_task0[reg_task0["regDate"] <= time0]
        reg_task0 = reg_task1[reg_task1["regDate"] >= time0 - days]
        reg_task_c = set(reg_task0.index) & set(open_tasks)
        reg_c = len(reg_task_c)
        reg_task_p = set(reg_task1.index) - set(open_tasks)
        reg_p = len(reg_task_p)
        sub_c = 0
        sub_p = 0
        win_p = 0
        if user in self.sub.index.levels[0]:
            sub_task0 = self.sub.loc[user]
            sub_task_c = sub_task0.reindex(reg_task_c).dropna()
            sub_task_c = sub_task_c[sub_task_c["subDate"] <= time0]
            sub_c = len(sub_task_c)
            if reg_p > 0:
                sub_task_p = sub_task0.reindex(reg_task_p).dropna()
                win_task_p = sub_task_p[sub_task_p["placement"] == 1]
                sub_p = len(sub_task_p)
                win_p = len(win_task_p)
        sim_tasks = self.sim_tasks[task]
        reg_x = 0
        if len(sim_tasks) > 0:
            reg_task = reg_task0.reindex(sim_tasks)
            reg_task.dropna(inplace=True)
            if len(reg_task) > 0:
                reg_x = len(reg_task)
        sub_rate = 0
        sub_score = 0
        sub_x = 0
        if sub_c > 0:
            sub_task = sub_task0.reindex(index=reg_task.index)
            sub_task.dropna(inplace=True)
            sub_task = sub_task[sub_task["subDate"] < time0]
            if len(sub_task) > 0:
                sub_x = len(sub_task)
                sub_rate = float(sub_x) / reg_x
                sub_score = sub_task["finalScore"].mean()
        return sub_rate, sub_score, sub_x, reg_x, reg_x - sub_x, reg_c, reg_c - sub_c, reg_p, sub_p, win_p

    def get_user_perfWithHis(self, user, time0, time1):
        '''
        用户user在任务task的开放的时间段内的任务状态和历史表现
        '''
        open_tasks = self.tasks[(self.tasks["RegStart"] <= time1) & (self.tasks["SubEnd"] >= time1)].index
        reg_task0 = self.reg.loc[user]
        reg_task0 = reg_task0[reg_task0["regDate"] <= time1]
        reg_task_c = reg_task0[reg_task0["regDate"] >= time0]
        reg_c = len(reg_task_c)
        reg_p = len(reg_task0)-reg_c
        sub_c = 0
        sub_p = 0
        win_p = 0
        not_sub_task = set(reg_task0.index) & set(open_tasks)
        not_sub = len(not_sub_task)
        sub_rate = 0.0
        sub_score = 0.0
        if user in self.sub.index.levels[0]:
            sub_task0 = self.sub.loc[user]
            sub_task0 = sub_task0[sub_task0["subDate"] <= time1]
            sub_task_c = sub_task0[sub_task0["subDate"] >= time0]
            win_task_p = sub_task0[sub_task0["placement"] == 1]
            sub_c = len(sub_task_c)
            sub_p = len(sub_task0)-sub_c
            win_p = len(win_task_p)
            not_sub_task = set(reg_task0.index)-set(sub_task0.index)
            not_sub_task = not_sub_task & set(open_tasks)
            not_sub = len(not_sub_task)
            if sub_c+sub_p > 0:
                if reg_p+reg_c > 0:
                    sub_rate = float(sub_p+sub_c)/(reg_p+reg_c)
                sub_score = sub_task0["finalScore"].mean()
        return reg_c, reg_p, sub_c, sub_p, win_p, not_sub, sub_rate, sub_score

    def get_user_his(self, user, time0):
        '''
        user perform history with brief count
        '''
        open_tasks = self.tasks[(self.tasks["RegStart"] <= time0) & (self.tasks["SubEnd"] >= time0)].index
        reg_task0 = self.reg.loc[user]
        reg_task = reg_task0[reg_task0["regDate"] <= time0]
        reg_h = len(reg_task)
        sub_h = 0
        win_h = 0
        reg_task_n = set(reg_task0[reg_task0["regDate"] > time0].index) & set(open_tasks)
        reg_n = len(reg_task_n)
        sub_n = 0
        win_n = 0
        if user in self.sub.index.levels[0]:
            sub_task0 = self.sub.loc[user]
            sub_task = sub_task0[sub_task0["subDate"] <= time0]
            sub_h = len(sub_task)
            win_task = sub_task[sub_task["placement"] == 1]
            win_h = len(win_task)
            sub_task_n = sub_task0.reindex(open_tasks).dropna()
            sub_task_n = sub_task_n[sub_task_n["subDate"] > time0]
            sub_n = len(sub_task_n)
            if sub_n > 0:
                win_task_n = sub_task_n[sub_task_n["placement"] == 1]
                win_n = len(win_task_n)
        return reg_h, sub_h, win_h, reg_n, sub_n, win_n


class Accuracy:
    def __init__(self, expect={}):
        self.expect = expect

    def top_n_accuracy(self, predict={}):
        PT = 0
        P = 0
        T = 0
        for u in predict.keys():
            PT += len(self.expect[u] & predict[u])
            P += len(self.expect[u])
            T += len(predict[u])
        print "Totally %d Positive Sample, recommend %d correctly from %d Tasks" % (P, PT, T)
        print "precision for TopN: %.3f; recall for TopN: %.3f"% (float(PT)/T, float(PT)/P)


if __name__=="__main__":
    print "basic service!"