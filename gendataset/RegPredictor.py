from service import *
from multiprocessing import Pool


def sampleDataSet(trainSet, col):
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


def getUserHis(user, records):
    user_his = records.apply(lambda x: Service(path).get_user_perfWithHis(user, x.RegStart, x.regDate), axis=1)
    records["CurrentReg"] = user_his.map(lambda x: x[0])
    records["PriorReg"] = user_his.map(lambda x: x[1])
    records["CurrentSub"] = user_his.map(lambda x: x[2])
    records["PriorSub"] = user_his.map(lambda x: x[3])
    records["PriorWin"] = user_his.map(lambda x: x[4])
    records["PriorNotSub"] = user_his.map(lambda x: x[5])
    records["PriorSubRate"] = user_his.map(lambda x: x[6])
    records["PriorSubScore"] = user_his.map(lambda x: x[7])
    return records


def getRecord(m, u, records, time0):
    regData = regData0.loc[u]
    records["handle"] = u
    records.reset_index(inplace=True)
    records["regDate"] = records["TaskId"].map(
        lambda t: regData.loc[t, "regDate"] if t in regData.index else records.loc[t, "SubEnd"])
    records["RegOrNot"] = records["TaskId"].map(lambda t: 1 if t in regData.index else -1)
    trainSet = records[records["regDate"] <= time0]
    trainSet = sampleDataSet(trainSet, "RegOrNot")
    trainSet = getUserHis(u, trainSet)
    # trainSet.to_csv(path + "train/reg_" + str(i) + ".csv", index=False)
    testSet = records[records["regDate"] > time0]
    testSet = getUserHis(u, testSet)
    # testSet.to_csv(path + "test/reg_" + str(i) + ".csv", index=False)
    trainSet = trainSet.append(trainSet)
    testSet = testSet.append(testSet)
    print u, len(trainSet), len(testSet)


def getDataSet(time0, users):
    start = time.time()
    tasks = tasks0[tasks0["RegStart"] <= time0]
    i = 0
    p = Pool()
    for u in users.index:
        i += 1
        m = users.loc[u, "memberSince"]
        records = tasks[tasks["SubEnd"] > m]
        p.apply_async(getRecord, args=(m, u, records, time0,))
    print "subProcess start....."
    p.close()
    p.join()
    end = time.time()
    print "time for make Reg trainSet %d: %.3f s" % (len(trainSet), end - start)

time0 = count_day("2016-09-01")
path = "../output/dataset2/"
trainSet = pd.DataFrame()
testSet = pd.DataFrame()
regData0 = pd.read_csv(path + "registrant.csv", index_col=["handle", "TaskId"])
tasks0 = pd.read_csv(path + "challenges.csv", index_col="TaskId")
activeU = pd.read_csv(path+str(time0)+"/ActiveUsers.csv", index_col="handle")
getDataSet(time0, activeU)
