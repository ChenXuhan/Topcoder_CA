# coding=utf8
__author__ = "Chenxh"

# from gendataset.service import *
from sumarrycode.SummaryData import *
from sklearn import metrics
# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def import_ESEM(type, seq):
    '''
    import train set(type=1) and test set(type=0) from csv files
    '''
    if type == 0:
        path = "../data/ESEM2016/crowd-project2-task-test-dataset"
    elif type == 1:
        path = "../data/ESEM2016/crowd-project2-task-training-dataset"
    else:
        return None, None
    file_name = path + seq + ".csv"
    print file_name
    data = pd.read_csv(file_name, index_col=["ChallengeID", "WorkerID"])
    classlist = ['Winner', 'Submitter', 'Quitter', 'NotInterested']
    get_classid = lambda x: classlist.index(x)
    data['WinnerOrSubmitter'] = data['WinnerOrSubmitter'].map(get_classid)
    y = data.iloc[:, -1]
    # print y.head()
    typelist = ['Architecture', 'Assembly Competition', 'Bug Hunt', 'Code', 'Conceptualization',
                'Content Creation', 'Copilot Posting', 'Design', 'Development', 'First2Finish',
                'Specification', 'Test Scenarios', 'Test Suites', 'UI Prototype Competition']
    x = data.iloc[:, :-1]
    x['Task Type'] = x['Task Type'].map(lambda x: typelist.index(x))
    # print x.head()
    return x, y


def feature_selection(x, selected=[], unselected=[]):
    '''
    选择特征
    :param selected: 为空时表示全选，否则表示选择特定的集合
    :param unselected: 为空时表示不删除任何特征
    '''
    result = x.iloc[:, :]
    if len(selected) > 0:
        result = x.iloc[:, selected]
    if len(unselected) > 0:
        result = result.drop(unselected, axis=1)
    return result


def summary_rf_model(train_x, train_y, test_x, test_y, numFeatures, path):
    missClassError = []
    nTreeList = range(25, 175, 25)
    for iTrees in nTreeList:
        depth = None
        rf_model = RandomForestClassifier(n_estimators=iTrees, max_depth=depth, max_features=numFeatures, oob_score=False, random_state=531)

        rf_model.fit(train_x, train_y)
        predicted = rf_model.predict(test_x)

        correct = metrics.accuracy_score(test_y, predicted)
        missClassError.append(1.0-correct)
    # 错误率
    print "Missclassification Error"
    print missClassError
    # 生成混淆矩阵
    confusionMat = metrics.confusion_matrix(test_y, predicted)
    print "Confusion Matrix"
    print confusionMat
    # 分析错误率随模型包含的决策树数量之间的关系
    plt.plot(nTreeList, missClassError)
    plt.xlabel('Number of Trees in Ensemble')
    plt.ylabel('Missclassification Error rating')
    plt.ylim([0.0, 1.1*max(missClassError)])
    plt.savefig(path+"mer_numTrees("+str(numFeatures)+").png")
    plt.show()
    # feature importances
    featureImportance = rf_model.feature_importances_
    # normalize by max importance
    featureImportance = featureImportance / featureImportance.max()
    # plot variable importance
    features = train_x.columns
    idxSorted = np.argsort(featureImportance)
    feature_series = pd.Series(featureImportance[idxSorted], index=features[idxSorted])
    feature_series.to_csv(path+"importance.csv")


class Predictor:
    def __init__(self):
        self.regT = pd.DataFrame()
        self.regP = pd.DataFrame()

    def getAction(self, x):
        pass

    def loadData(self, path="../output/dataset1/result/"):
        self.regT = pd.read_csv(path+"regT.csv", index_col=["TaskId", "handle"])
        self.regP = pd.read_csv(path+"regP.csv", index_col=["TaskId", "handle"])

    def predictRegister(self, model, train_x, train_y, test_x, test_y):
        start = time.time()
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        predicted = pd.DataFrame(predicted, index=test_y.index)
        predicted.columns = ["RegOrNot"]
        test_y = test_y.to_frame("RegOrNot")
        regT = predicted[predicted["RegOrNot"] > 0]
        regP = test_y[test_y["RegOrNot"] > 0]
        self.regT = pd.concat([self.regT, regT])
        self.regP = pd.concat([self.regP, regP])
        end = time.time()
        print "time for predict Register：%.3f s"%(end-start)

    def predictSubmiter(self, model, train_x, train_y, test_x, test_y, top_n=1):
        start = time.time()
        model.fit(train_x, train_y)
        predicted = model.predict_prob(test_x)
        predicted = pd.DataFrame(predicted, index=test_y.index)
        predicted["Action"] = predicted[0].map(lambda x: 0 if x > 0.33 else None)
        test_y = test_y.to_frame("Action")
        end = time.time()
        print "time for predict Submiter：%.3f s"%(end-start)

    def analyzeTP(self):
        regTP = pd.concat([self.regT, self.regP], axis=1, join='inner')
        regTP = regTP.iloc[:, :1]
        regT = self.regT.reset_index()
        regP = self.regP.reset_index()
        regTP.reset_index(inplace=True)
        userSet = set(regT['handle']) | set(regP['handle'])
        self.regTP = pd.DataFrame(index=list(userSet))
        self.regTP["regT"] = regT.groupby('handle').size()
        self.regTP["regP"] = regP.groupby('handle').size()
        self.regTP["regTP"] = regTP.groupby('handle').size()
        self.regTP.fillna(0, inplace=True)
        self.regTP["reg_precision"] = self.regTP.apply(lambda s: s["regTP"]/s["regT"] if s["regT"] > 0 else 0, axis=1)
        self.regTP["reg_recall"] = self.regTP.apply(lambda s: s["regTP"]/s["regP"] if s["regP"] > 0 else 0, axis=1)

    def genPredictReg(self, time0, path="../output/dataset1/"):
        predReg = self.regP
        predReg["regDate"] = time0
        return predReg


if __name__ == '__main__':
    import time
    start = time.time()
    pred = Predictor()
    file_name = lambda type, num: "../output/dataset1/2/%s/reg_%d.csv" % (type, num)
    trainSet_al = pd.DataFrame()
    testSet_al = pd.DataFrame()
    for i in range(69):
        file_train = file_name("train", i+1)
        trainSet = pd.read_csv(file_train, index_col=["challengeId", "handle"])
        file_test = file_name("test", i+1)
        testSet = pd.read_csv(file_test, index_col=["challengeId", "handle"])
        trainSet_al = pd.concat([trainSet_al, trainSet])
        testSet_al = pd.concat([testSet_al, testSet])
        train_x = trainSet.drop("RegOrNot", axis=1)
        train_y = trainSet["RegOrNot"]
        test_x = testSet.drop("RegOrNot", axis=1)
        test_y = testSet["RegOrNot"]
        train_x["regDate"] = train_x["SubEnd"]-train_x["regDate"]
        test_x["regDate"] = test_x["SubEnd"]-test_x["regDate"]
        train_x = feature_selection(train_x, unselected=["RegStart", "SubEnd"])
        test_x = feature_selection(test_x, unselected=["RegStart", "SubEnd"])
        model = RandomForestClassifier()
        print "RandomForest, Sample", i+1
        pred.predictRegister(model, train_x, train_y, test_x, test_y)
    pred.regT.reset_index().to_csv("../output/dataset1/register/regT.csv", index=False, header=["TaskId", "handle", "RegOrNot"])
    pred.regP.reset_index().to_csv("../output/dataset1/register/regP.csv", index=False, header=["TaskId", "handle", "RegOrNot"])
    # pred.loadData()
    pred.analyzeTP()
    print pred.regTP.describe()
    print pred.regTP.sum()
    pred.regTP.to_csv("regTP.csv")
    # ignore difference between developers
    pred_al = Predictor()
    train_x = trainSet_al.drop("RegOrNot", axis=1)
    train_y = trainSet_al["RegOrNot"]
    test_x = testSet_al.drop("RegOrNot", axis=1)
    test_y = testSet_al["RegOrNot"]
    train_x["regDate"] = train_x["SubEnd"]-train_x["regDate"]
    test_x["regDate"] = test_x["SubEnd"]-test_x["regDate"]
    train_x = feature_selection(train_x, unselected=["RegStart", "SubEnd"])
    test_x = feature_selection(test_x, unselected=["RegStart", "SubEnd"])
    model = RandomForestClassifier()
    print "RandomForestClassifier, all: "
    pred_al.predictRegister(model, train_x, train_y, test_x, test_y)
    pred_al.regT.reset_index().to_csv("../output/dataset1/register/all_regT.csv", index=False, header=["TaskId", "handle", "RegOrNot"])
    pred_al.regP.reset_index().to_csv("../output/dataset1/register/all_regP.csv", index=False, header=["TaskId", "handle", "RegOrNot"])
    pred_al.analyzeTP()
    print pred_al.regTP.describe()
    print pred_al.regTP.sum()
    pred_al.regTP.to_csv("regTP_al.csv")
    end = time.time()
    print "模型训练与预测耗时：%.3f s"%(end-start)
    # dataSet = pd.read_csv("../output/dataset1/DCWDS-1/tasks-90days-3623.csv").drop(["TaskId", "handle"], axis=1)#, index_col=["TaskId", "handle"])
    # X = dataSet.drop("Action", axis=1)
    # Y = dataSet["Action"]
    # trainSize = int(0.8*len(dataSet))
    # train_x = X.iloc[:trainSize, :]
    # train_y = Y[:trainSize]
    # print "训练集大小：", trainSize
    # train_x, train_y = import_ESEM(1, "0")
    # summary_stats(train_x, train_y, "ESEM_all/training")
    # test = pd.read_csv("../gendataset/task-test-3623.csv", index_col=["TaskId", "handle"])
    # test_x = X.iloc[trainSize:, :]
    # test_y = Y[trainSize:]
    # test_x, test_y = import_ESEM(0, "0")
    # summary_stats(test_x, test_y, "ESEM_all/test")
    # train_x = feature_selection(train_x, selected=[], unselected=[])
    # train_x.to_csv("../output/ESEM2016/train_x.csv")
    # test_x = feature_selection(test_x, selected=[], unselected=[])
    # test_x.to_csv("../output/ESEM2016/test_x.csv")
    # train_x = feature_selection(train_x, unselected=["OverAllReliabilityScore", \
    # "AvgSuccessRateOnSimilarTasks","AvgSubmissionQualityOnSimilarTasks","AverageofAverageSubmissionQualityForTopYCompetitiors","AverageofAverageSucessRateForTopYCompetitiors"])
    # test_x = feature_selection(test_x, unselected=["OverAllReliabilityScore",\
    # "AvgSuccessRateOnSimilarTasks","AvgSubmissionQualityOnSimilarTasks","AverageofAverageSubmissionQualityForTopYCompetitiors","AverageofAverageSucessRateForTopYCompetitiors"])
    # end = time.time()
    # print "读取数据耗时：%.3f s"%(end-start)
    # model = RandomForestClassifier()
    # predict_report(model, train_x, train_y, test_x, test_y, "../output/dataset1/result/")
    # end = time.time()
    # print "模型训练与预测耗时：%.3f s"%(end-start)
