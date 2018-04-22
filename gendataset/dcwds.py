#coding=utf8
import time
import pandas as pd
from service import Service, count_day
from multiprocessing import Process, Queue, Condition, Pool


class RegPredict:
    def __init__(self, path="../output/dataset1/"):
        self.path = path
        self.trainSet = pd.DataFrame()
        self.testSet = pd.DataFrame()
        self.regData0 = pd.read_csv(path+"registrant.csv", index_col=["handle", "TaskId"])
        self.tasks = pd.read_csv(path+"challenges.csv", index_col="TaskId")

    def sampleDataSet(self, trainSet, col):
        train_p = trainSet[trainSet[col] > 0]
        num_p = len(train_p)
        train_n = trainSet[trainSet[col] < 0]
        num_n = len(train_n)
        if num_n < num_p:
            trainSet = train_p.sample(n=num_n).append(train_n)
        elif num_n > num_p:
            trainSet = train_p.append(train_n.sample(n=num_p))
        else:
            trainSet = train_p.append(train_n)
        return trainSet

    def getUserHis(self, user, records):
        user_his = records.apply(lambda x: Service(self.path).get_user_perfWithHis(user, x.RegStart, x.regDate), axis=1)
        records["CurrentReg"] = user_his.map(lambda x: x[0])
        records["PriorReg"] = user_his.map(lambda x: x[1])
        records["CurrentSub"] = user_his.map(lambda x: x[2])
        records["PriorSub"] = user_his.map(lambda x: x[3])
        records["PriorWin"] = user_his.map(lambda x: x[4])
        records["PriorNotSub"] = user_his.map(lambda x: x[5])
        records["PriorSubRate"] = user_his.map(lambda x: x[6])
        records["PriorSubScore"] = user_his.map(lambda x: x[7])
        return records

    def getRecord(self, u, records, time0):
        start = time.time()
        print u, "相关项目：", len(records.index)
        #return
        regData = self.regData0.loc[u]
        records["handle"] = u
        reg_task = regData.index
        records["regDate"] = records["SubEnd"]
        records.loc[reg_task, "regDate"] = regData["regDate"]
        records["RegOrNot"] = -1
        records.loc[reg_task, "RegOrNot"] = 1
        records.reset_index(inplace=True)
        trainSet = records[records["regDate"] <= time0]
        trainSet = self.sampleDataSet(trainSet, "RegOrNot")
        trainSet = self.getUserHis(u, trainSet)
        testSet = records[records["regDate"] > time0]
        testSet = self.getUserHis(u, testSet)
        print trainSet.iloc[:3, -12:]
        # self.trainSet = self.trainSet.append(trainSet)
        # self.testSet = self.testSet.append(testSet)
        end = time.time()
        print "current DataSize: %d/%d, spending %.3f s." % (len(trainSet), len(testSet), end-start)


class MultipRecord(Process):
    def __init__(self, rgp, u, m, time0, queue=None, cond=None):
        Process.__init__(self)
        self.queue, self.cond = queue, cond
        self.rgp = rgp
        self.u = u
        self.m = m
        self.time0 = time0

    def run(self):
        records = self.rgp.tasks[self.rgp.tasks["SubEnd"] >= self.m]
        self.rgp.getRecord(self.u, records, self.time0)
        self.cond.acquire()
        self.queue.put(self.u)
        print "No.%d %s:" % (self.queue.size(),self.u)
        self.cond.notify()
        self.cond.release()


def getDataSet(time0, users, rgp):
        start = time.time()
        plt_queue = Queue()
        plt_cond = Condition()
        pool = []
        max_process = 2
        i = 0
        n_user = len(users.index)
        while i < n_user:
            if len(pool) >= max_process:
                plt_cond.acquire()
                if plt_queue.empty():
                    plt_cond.wait()
                while not plt_queue.empty():
                    pos = -1
                    uq = plt_queue.get()
                    for pos in range(len(pool)):
                        if pool[pos].u == uq:
                            break
                    pool[pos].join()
                    del pool[pos]
                plt_cond.release()

            u = users.index[i]
            m = users.loc[u, "memberSince"]
            p = MultipRecord(rgp, u, m, time0, plt_queue, plt_cond)
            pool.append(p)
            p.start()
            i += 1
        print "subProcess start....."
        [p.join() for p in pool]
        end = time.time()
        print "time for make Reg trainSet: %.3f s" % (end-start)


class DCWDS:
    TOPY = 5
    XDAYS = 60
    SIM_VALUE = 0.8

    def __init__(self, path="../output/dataset1/"):
        self.path = path
        self.trainSet = pd.DataFrame()
        self.testSet = pd.DataFrame()
        self.testReg = pd.DataFrame()
        self.regData0 = pd.read_csv(path+"registrant1.csv", index_col=["TaskId", "handle"])
        self.subData0 = pd.read_csv(path+"submission.csv", index_col=["TaskId", "handle"])
        self.tasks = pd.read_csv(path+"challenges.csv", index_col="TaskId")

    def loadData(self, filename="test-3623.csv"):
        self.trainSet = pd.read_csv(filename, index_col=["TaskId", "handle"])

    def loadRegData(self, predictReg):
        self.testReg = predictReg

    def getRecord(self, task):
        start = time.time()
        competitions = {}
        sum_perf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        count = 0
        records = self.regData0.loc[task]
        # records = records[records["regDate"] <= time0]
        subData = pd.DataFrame()
        if task in self.subData0.index.levels[0]:
            subData = self.subData0.loc[task]
        records["TaskId"] = task
        for user in records.index:
            records.loc[user, "Action"] = 2
            regDate = records.loc[user, "regDate"]
            # print task, user, regDate
            user_perf = Service(self.path).get_user_perf(task, user, regDate, self.XDAYS)
            # print user_perf#, competitions
            records.loc[user, "SubRate"] = user_perf[0]
            records.loc[user, " SubScore"] = user_perf[1]
            records.loc[user, "SubXDays"] = user_perf[2]
            records.loc[user, "RegXDays"] = user_perf[3]
            records.loc[user, "NotSubXDays"] = user_perf[4]
            records.loc[user, "CurrentReg"] = user_perf[5]
            records.loc[user, "CurrentNotSub"] = user_perf[6]
            if subData.empty:
                pass
            elif user in subData.index:
                records.loc[user, "Action"] = 1
                if subData.loc[user, "placement"] > 0:
                    if subData.loc[user, "subStatus"] == "Active" and subData.loc[user, "placement"] == 1.0:
                        records.loc[user, "Action"] = 0
                        competitions[user] = user_perf
                        count += 1
                        for i in range(7):
                            sum_perf[i] += user_perf[i]
                    elif subData.loc[user, "placement"] <= 5.0:
                        competitions[user] = user_perf
                        count += 1
                        for i in range(7):
                            sum_perf[i] += user_perf[i]
        print "Task %d Top %d users perf:" % (task, count)
        print sum_perf
        if count <= 1:
            records["SubRateForTopY"] = 0
            records["SubScoreForTopY"] = 0
            records["SubXDaysForTopY"] = 0
            records["RegXDaysForTopY"] = 0
            records["NotSubXDaysForTopY"] = 0
            records["CurrentRegForTopY"] = 0
            records["CurrentNotSubForTopY"] = 0
        else:
            records["SubRateForTopY"] = sum_perf[0]/count
            records["SubScoreForTopY"] = sum_perf[1]/count
            records["SubXDaysForTopY"] = sum_perf[2]/count
            records["RegXDaysForTopY"] = sum_perf[3]/count
            records["NotSubXDaysForTopY"] = sum_perf[4]/count
            records["CurrentRegForTopY"] = sum_perf[5]/count
            records["CurrentNotSubForTopY"] = sum_perf[6]/count
            for key in competitions.keys():
                # print key, competitions[key]
                records.loc[key, "SubRateForTopY"] = (sum_perf[0]-competitions[key][0])/(count-1)
                records.loc[key, "SubScoreForTopY"] = (sum_perf[1]-competitions[key][1])/(count-1)
                records.loc[key, "SubXDaysForTopY"] = (sum_perf[2]-competitions[key][2])/(count-1)
                records.loc[key, "RegXDaysForTopY"] = (sum_perf[3]-competitions[key][3])/(count-1)
                records.loc[key, "NotSubXDaysForTopY"] = (sum_perf[4]-competitions[key][4])/(count-1)
                records.loc[key, "CurrentRegForTopY"] = (sum_perf[5]-competitions[key][5])/(count-1)
                records.loc[key, "CurrentNotSubForTopY"] = (sum_perf[6]-competitions[key][6])/(count-1)
        self.trainSet = self.trainSet.append(records)
        end = time.time()
        print "time for get TrainSet for task %d : %.3f s" % (task, end-start)

    def getTrainSet(self, time0, size):
        start = time.time()
        tasksList = self.tasks[(self.tasks["RegStart"] >= time0-size) & (self.tasks["SubEnd"] <= time0)].index
        p = Pool()
        for task in tasksList:
            p.apply_async(self.getRecord, args=(task, ))
        p.close()
        p.join()
        self.trainSet.reset_index(inplace=True)
        self.trainSet.set_index(["TaskId", "handle"], inplace=True)
        end = time.time()
        print "time for make trainSet %d: %.3f s" % (len(self.trainSet), end-start)

    def getTestRecord(self, task):
        start = time.time()
        competitions = {}
        sum_perf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        count = 0
        records = self.testReg.loc[task]
        subData = pd.DataFrame()
        if task in self.subData0.index.levels[0]:
            subData = self.subData0.loc[task]
        records["TaskId"] = task
        for user in records.index:
            if user in self.regData0.loc[task].index:
                records.loc[user, "Action"] = 2
            else:
                records.loc[user, "Action"] = 3
            regDate = records.loc[user, "regDate"]
            # print task, user, regDate
            user_perf = Service(self.path).get_user_perf(task, user, regDate, self.XDAYS)
            # print user_perf#, competitions
            records.loc[user, "SubRate"] = user_perf[0]
            records.loc[user, " SubScore"] = user_perf[1]
            records.loc[user, "SubXDays"] = user_perf[2]
            records.loc[user, "RegXDays"] = user_perf[3]
            records.loc[user, "NotSubXDays"] = user_perf[4]
            records.loc[user, "CurrentReg"] = user_perf[5]
            records.loc[user, "CurrentNotSub"] = user_perf[6]
            if subData.empty:
                pass
            elif user in subData.index:
                records.loc[user, "Action"] = 1
                if subData.loc[user, "placement"] > 0:
                    if subData.loc[user, "subStatus"] == "Active" and subData.loc[user, "placement"] == 1.0:
                        records.loc[user, "Action"] = 0
                        competitions[user] = user_perf
                        count += 1
                        for i in range(7):
                            sum_perf[i] += user_perf[i]
                    elif subData.loc[user, "placement"] <= 5.0:
                        competitions[user] = user_perf
                        count += 1
                        for i in range(7):
                            sum_perf[i] += user_perf[i]
        print "Task %d Top %d users perf:" % (task, count)
        print sum_perf
        if count <= 1:
            records["SubRateForTopY"] = 0
            records["SubScoreForTopY"] = 0
            records["SubXDaysForTopY"] = 0
            records["RegXDaysForTopY"] = 0
            records["NotSubXDaysForTopY"] = 0
            records["CurrentRegForTopY"] = 0
            records["CurrentNotSubForTopY"] = 0
        else:
            records["SubRateForTopY"] = sum_perf[0]/count
            records["SubScoreForTopY"] = sum_perf[1]/count
            records["SubXDaysForTopY"] = sum_perf[2]/count
            records["RegXDaysForTopY"] = sum_perf[3]/count
            records["NotSubXDaysForTopY"] = sum_perf[4]/count
            records["CurrentRegForTopY"] = sum_perf[5]/count
            records["CurrentNotSubForTopY"] = sum_perf[6]/count
            for key in competitions.keys():
                # print key, competitions[key]
                records.loc[key, "SubRateForTopY"] = (sum_perf[0]-competitions[key][0])/(count-1)
                records.loc[key, "SubScoreForTopY"] = (sum_perf[1]-competitions[key][1])/(count-1)
                records.loc[key, "SubXDaysForTopY"] = (sum_perf[2]-competitions[key][2])/(count-1)
                records.loc[key, "RegXDaysForTopY"] = (sum_perf[3]-competitions[key][3])/(count-1)
                records.loc[key, "NotSubXDaysForTopY"] = (sum_perf[4]-competitions[key][4])/(count-1)
                records.loc[key, "CurrentRegForTopY"] = (sum_perf[5]-competitions[key][5])/(count-1)
                records.loc[key, "CurrentNotSubForTopY"] = (sum_perf[6]-competitions[key][6])/(count-1)
        self.testSet = self.testSet.append(records)
        end = time.time()
        print "time for get TrainSet for task %d : %.3f s" % (task, end-start)

    def getTestSet(self, time0):
        start = time.time()
        tasksList = self.tasks[(self.tasks["RegStart"] <= time0) & (self.tasks["SubEnd"] > time0)].index
        hasReg = self.regData0.loc[tasksList]
        hasReg = hasReg[hasReg["regDate"] <= time0]
        self.testReg.to_csv(self.path+"testReg0.csv")
        self.testReg = pd.concat([self.testReg, hasReg], join="outer")
        self.testReg.to_csv(self.path+"testReg.csv")
        tasksList = set(tasksList) & set(self.testReg.index.levels[0])
        p = Pool()
        for task in tasksList:
            p.apply_async(self.getTestRecord, args=(task, ))
        p.close()
        p.join()
        self.testSet.reset_index(inplace=True)
        self.testSet.set_index(["TaskId", "handle"], inplace=True)
        end = time.time()
        print "time for make trainSet %d: %.3f s" % (len(self.testSet), end-start)

    def add_task_to_project(self, records):
        start = time.time()
        records.reset_index(inplace=True)
        # print self.records.head()
        records.set_index("TaskId", inplace=True)
        srecords = pd.concat([records, self.tasks], axis=1, join='inner')
        records["RegStart"] = records["regDate"] - records["RegStart"]
        records["SubEnd"] = records["SubEnd"] - records["regDate"]
        # records.drop(["regDate", "rating"], inplace=True, axis=1)
        records.drop("regDate", inplace=True, axis=1)
        records.index.names = ["TaskId"]
        end = time.time()
        print "time for join task details with records: %.3f s" % (end-start)


class UserClass:
    def __init__(self, path="../output/dataset1/", file="users.data"):
        self.path = path
        self.active = None
        self.inActive = None
        self.users = pd.read_pickle(path+file)

    def getUsersCount(self, time0):
        start = time.time()
        temp = self.users.reset_index()
        user_his = temp.apply(lambda x: Service(self.path).get_user_his(x.handle, time0), axis=1)
        temp["RegNum"] = user_his.map(lambda x: x[0])
        temp["SubNum"] = user_his.map(lambda x: x[1])
        temp["WinNum"] = user_his.map(lambda x: x[2])
        temp["RegNumNext"] = user_his.map(lambda x: x[3])
        temp["SubNumNext"] = user_his.map(lambda x: x[4])
        temp["WinNumNext"] = user_his.map(lambda x: x[5])
        self.users = temp.set_index("handle")
        end = time.time()
        print "time for make getUsersCount: %.3f s" % (end-start)

    def getActiveU(self, type=0):
        if type == 0:
            self.active = self.users[self.users["RegNum"] > 10]
            self.inActive = self.users[self.users["RegNum"] > 2]
        if type == 1:
            self.active = self.users[self.users["SubNum"] > 0]
            self.active = self.active[self.active["RegNum"] > 10]
            self.inActive = self.users.drop(self.active.index)
            self.inActive = self.inActive[self.inActive["RegNum"] > 2]
        if type == 2:
            self.active = self.users[self.users["SubNum"] > 0]
            self.active = self.active[self.active["RegNum"] > 19]
            self.inActive = self.users.drop(self.active.index)
            self.inActive = self.inActive[self.inActive["RegNum"] > 2]
        return self.active


if __name__ == "__main__":
    time0 = count_day("2016-09-01")
    print time0
    path0 = "../output/dataset2/"
    # myService = Service(path0)
    # userClass.getUsersCount(time0)
    # print userClass.users.describe()
    # userClass.users.to_csv(path0+str(time0)+"/userCount.csv")
    # activeU = userClass.getActiveU(2)
    # userClass.active.to_csv(path0+str(time0)+"/ActiveUsers.csv")
    # userClass.inActive.to_csv(path0+str(time0)+"/inActiveUsers.csv")
    # print activeU.describe()
    # print activeU.sum()
    # activeU = pd.read_pickle(path0+str(time0)+"/ActiveUsers.data")
    # regPredict = RegPredict(path0)
    # #records = regPredict.tasks[regPredict.tasks["SubEnd"] >= activeU.loc["Minesweeper", "memberSince"]]
    # #regPredict.getRecord("Minesweeper", records, time0)
    # getDataSet(time0, activeU.iloc[:6, :], regPredict)
    # regPredict.trainSet.to_csv(path0+str(time0)+"/train_register.data")
    # regPredict.testSet.to_csv(path0+str(time0)+"/test_register.data")

    # trainL = 120
    # file_name = "tasks-%ddays-%d.csv" % (trainL, time0)
    # dcwds = DCWDS()
    # dcwds.loadData("../output/dataset1/DCWDS-1/"+file_name)#"train-60days-3623.csv")
    # predReg = pd.read_csv("../output/dataset1/result/predReg.csv", index_col=["TaskId", "handle"])
    # predReg = predReg.drop("RegOrNot", axis=1)
    # dcwds.loadRegData(predReg)
    # # print dcwds.testReg.loc[30054547]
    # # dcwds.sampleRecords()
    # # dcwds.getTrainSet(time0, trainL)
    # dcwds.getTestSet(time0)
    # # dcwds.add_task_to_project(dcwds.trainSet)
    # dcwds.add_task_to_project(dcwds.testSet)
    # # dcwds.trainSet.to_csv(dcwds.path+"train/"+file_name)
    # dcwds.testSet.to_csv(dcwds.path+"test/"+file_name)
    # print file_name
