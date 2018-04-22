#coding=utf8

from gendataset.service import *
from rating.elorating import Elorating
from multiprocessing import Process, Queue, Condition
import pickle


def preprocess(dataStr="dataset1/"):
    """
    预处理提交数据，删除没有注册信息的提交
    :return:
    """
    fromPath = "../data/"+dataStr
    toPath = "../output/"+dataStr
    chas0 = pd.read_csv(fromPath+"challenges0.csv", index_col="TaskId")
    regData0 = pd.read_csv(fromPath+"registrant0.csv", index_col=["TaskId", "handle"], \
                           dtype={'regDate': 'int32', 'rating': 'float32'})
    regData0.fillna(0, inplace=True)
    regData = regData0.loc[list(chas0.index)]
    regData["regDate"] = regData["regDate"].map(lambda x: count_day(x))
    regData.reset_index(inplace=True)
    regData.dropna(inplace=True)
    regData.sort_values(["TaskId", "regDate"], inplace=True)
    regData.set_index(["TaskId", "handle"], inplace=True)
    regData.to_pickle(toPath+"registrant.data")
    print toPath+"registrant.data"
    chas = chas0.loc[list(regData.index.levels[0])]
    chas.to_pickle(toPath+"challenges0.data")
    print toPath+"challenges0.data"
    subData0 = pd.read_csv(fromPath+"submission0.csv", index_col=["TaskId", "handle"], \
                           dtype={'placement': 'int32', 'subDate': 'int32', 'finalScore': 'float32'})
    subData0.fillna(600, inplace=True)
    subData = subData0.loc[regData.index].dropna()
    subData["subDate"] = subData["subDate"].map(lambda x: count_day(x))
    subData.reset_index(inplace=True)
    subData.sort_values(["TaskId", "placement"], inplace=True)
    subData.set_index(["TaskId", "handle"], inplace=True)
    subData.to_pickle(toPath+"submission.data")
    print toPath+"submission.data"
    users0 = pd.read_csv(fromPath+"users0.csv", index_col="handle")
    users = users0.reindex(regData.index.levels[1])
    users["memberSince"] = users["memberSince"].map(lambda x: count_day(x))
    users.sort_values("memberSince", inplace=True)
    users.to_pickle(toPath+"users.data")
    print toPath+"users.data"


def task_features(path="../output/dataset1/"):
    '''
    提取项目特征的过程
    :return: 返回描述项目特征的Dataframe
    '''
    # 获取technology
    def get_tech(x):
        if type(x) == str:
            if x:
                tech_set.update(x.split(','))

    def tech_to_num(col, type):
        tasks0[col].map(lambda x: get_tech(x))
        tech_list = sorted(list(tech_set))
        dictObj[type] = tech_list
        print type+"("+str(len(tech_list))+"): ", dictObj[type]
        tech_dict = {}
        i = 0
        for tech in tech_list:
            tech_dict[tech] = type + str(i) + ":" + tech
            i += 1
        for tech in tech_list:
            name = tech_dict[tech]
            tasks0[name] = tasks0[col].map(lambda x: 1 if tech in str_to_list(x) else 0).astype('int8')

    def count_prize(strlist):
        max = 0
        sum = 0
        for item in strlist:
            p = float(item)
            sum += p
            if max < p:
                max = p
        return max, sum

    def prize_to_num(col):
        tasks0['maxPrize'] = tasks0[col].map(lambda x: count_prize(str_to_list(x))[0]).astype('float32')
        tasks0['totalPrize'] = tasks0[col].map(lambda x: count_prize(str_to_list(x))[1]).astype('float32')

    def date_to_num(col, type):
        tasks0[type] = tasks0[col].map(lambda x: count_day(x)).astype('int32')

    tasks0 = pd.read_pickle(path+"challenges0.data")
    tech_set = set()
    tasks0['challengeType'].map(lambda x: get_tech(x))
    type_list = sorted(list(tech_set))
    dictObj = {}
    dictObj["Task Type"] = type_list
    print "Task Type("+str(len(type_list))+"): ", dictObj["Task Type"]
    tasks0['Task Type'] = tasks0["challengeType"].map(lambda x: type_list.index(x)).astype('uint8')
    tech_set = set()
    tech_to_num('technology', 'Tech')
    tech_set = set()
    tech_to_num('platforms', 'Plat')
    prize_to_num("prize")
    date_to_num("postingDate", "RegStart")
    date_to_num("submissionEndDate", "SubEnd")
    tasks0["Duration"] = tasks0["SubEnd"] - tasks0["RegStart"]+1
    tasks = tasks0.iloc[:, 6:]
    tasks.sort_values(["SubEnd", "RegStart"], inplace=True)
    tasks.to_pickle(path+"challenges.data")
    with open(path+"features.json", "w") as f:
        json.dump(dictObj, f)


class CompeteRate:
    def updateRate0(self, rankList):
        w = rankList.pop(0)
        for u in rankList:
            score = Elorating(self.users.loc[w, "Compete1"], self.users.loc[u, "Compete1"])
            score.setResult(1)
            self.users.loc[w, "Compete1"] = score.ratingA
            self.users.loc[u, "Compete1"] = score.ratingB

    def scoreing1(self, path, ELO_RATING_DEFAULT=1500):
        tasks = pd.read_pickle(path+"challenges.data")
        regData = pd.read_pickle(path+"registrant.data").drop("rating", axis=1)
        subData = pd.read_pickle(path+"submission.data")
        self.users = pd.read_pickle(path+"users.data")
        self.users["Compete1"] = ELO_RATING_DEFAULT
        self.users["Compete1"] = self.users["Compete1"].astype('float32')
        count = 0
        flag = 0
        max = len(tasks.index)
        for task in tasks.index:
            count += 1
            if count*5/max > flag:
                print count, count*100/max, "%"
                regData.to_pickle(path+"registrant1.data")
                flag += 1
            registers = regData.loc[task].index
            indices = [(task, reg) for reg in registers]
            numReg = len(registers)
            regData.loc[indices, "Compete1"] = pd.Series(self.users.loc[registers, "Compete1"].tolist(),\
                                                         index=[[task]*numReg, registers])
            if task in subData.index.levels[0]:
                submitter = subData.loc[task].index
                if isinstance(submitter, str):
                    pass
                else:
                    rankList = submitter.tolist()
                    self.updateRate0(rankList)
        regData = regData.astype({"Compete1": 'float32'})
        regData.to_pickle(path+"registrant_Compete1.data")
        self.users.to_pickle(path+"user_Com bpete.data")


class SimTasks(Process):
    def __init__(self, filepath, task0, tasks, thd, queue, cond):
        Process.__init__(self)
        self.filepath = filepath
        self.task0 = task0
        self.tasks = tasks
        self.thd = thd
        self.queue = queue
        self.cond = cond

    def run(self):
        self.get_sim_tasks()

        self.cond.acquire()
        self.queue.put(self.task0.name)
        self.cond.notify()
        self.cond.release()

    def get_sim_tasks(self):
        sim_tasks = []
        for id1 in self.tasks.index:
            # if id1 == id0:
            #     continue
            task1 = self.tasks.loc[id1]
            sim = 0
            sim += dist(self.task0["maxPrize"], task1["maxPrize"])
            sim += dist(self.task0["totalPrize"], task1["totalPrize"])
            # sim += dist(self.task0["RegStart"], task1["RegStart"])
            # sim += dist(self.task0["SubEnd"], task1["SubEnd"])
            sim += dist(self.task0["Duration"], task1["Duration"])
            sim += 1 if self.task0["Task Type"] == task1["Task Type"] else 0
            sim += match(self.task0[2:177], task1[2:177])
            sim += match(self.task0[177:208], task1[177:208])
            if sim > self.thd:
                sim_tasks.append(id1)
        print self.task0.name, "'s similar tasks:", len(sim_tasks)
        with open(self.filepath, "wb")as f:
            pickle.dump(sim_tasks, f)


def genSimTask(path, days=0, thd=0.8):
    start = time.time()
    pool = []
    selTasks = pd.read_pickle(path+"challenges.data")
    selTasks = selTasks.iloc[:100, :]
    taskIndex = selTasks.index
    maxProcess = 64
    sel_queue = Queue()
    sel_cond = Condition()
    for i in range(len(taskIndex)):
        if len(pool) >= maxProcess:
            sel_cond.acquire()
            if sel_queue.empty():
                sel_cond.wait()
            while not sel_queue.empty():
                pos = -1
                name0 = sel_queue.get()
                for pos in range(len(pool)):
                    if pool[pos].task0.name == name0:
                        break
                pool[pos].join()
                del pool[pos]
            sel_cond.release()

        task0 = selTasks.iloc[i]
        cand_tasks = selTasks[selTasks["RegStart"] <= task0["SubEnd"]]
        if days > 0:
            cand_tasks = cand_tasks[task0["RegStart"]-cand_tasks["RegStart"] <= days]
        filepath0 = path+"simTasks%d/sim_tasks_%d.data" % (days, taskIndex[i])
        proc = SimTasks(filepath0, task0, cand_tasks, thd, sel_queue, sel_cond)
        proc.start()
        pool.append(proc)

    [pro.join() for pro in pool]
    end = time.time()
    print "time for generate sim_task %.3f s." % (end-start)
    simTaskContainer = {}
    for taskI in taskIndex:
        filepath0 = path+"simTasks%d/sim_tasks_%d.data" % (days, taskIndex[i])
        with open(filepath0, "rb")as f:
            sim_tasks = pickle.load(f)
            simTaskContainer[taskI] = sim_tasks
    sum_path = path+"simTasks/sim_tasks_%d.data" % days
    with open( sum_path, "wb")as f:
        pickle.dump(simTaskContainer, f)


if __name__ == '__main__':
    dataStr = "dataset2/"
    path0 = "../output/"+dataStr
    # preprocess(dataStr)
    # task_features(path0)
    # CompeteRate().scoreing1(path0)
    genSimTask(path0, 20)
    # frame0 = pd.read_pickle(path0+"challenges.data")
    # series0 = frame0.loc[30061160]
    # print series0.name



