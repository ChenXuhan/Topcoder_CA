# coding=utf8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summary_stats(x, y, file_name):
    file_name = "../output/summary/" + file_name
    try:
        sns.distplot(x["OverAllReliabilityScore"])
        plt.savefig(file_name + "reliabilityPro.png")
        plt.show()
    finally:
        print file_name + " no attribute named OverAllReliabilityScore from x"
    with open(file_name+'.txt', 'w') as file_obj:
        x.info(buf=file_obj)
        file_obj.write("\n")
        file_obj.write("Workers      " + str(len(x.groupby("WorkerID"))) + "\n")
        file_obj.write("Challenges   " + str(len(x.groupby("ChallengeID"))) + "\n")
        file_obj.write("\n")
        file_obj.write(str(y.sum()))
        file_obj.write("\n\n")
        x.groupby("WorkerID").size().to_csv(file_name + "GroupByWorker.csv")
        x.groupby("ChallengeID").size().to_csv(file_name + "GroupByTask.csv")
        x.groupby(["WorkerID", "Type"]).size().to_csv(file_name + "GroupByWorkerTaskType.csv")
        # tasks = x.drop_duplicates(subset=["ChallengeID"], keep='first', inplace=False)  # 去重
        # tasks.groupby("Type").size().to_csv(file_name + "TaskNumGroupByTaskType.csv")
        x.groupby("Type").agg({"ChallengeID": pd.Series.nunique}).to_csv(file_name + "TaskNumGroupByTaskType.csv")
        # file_obj.writelines(text)


def summary_groupby(cols, filename="../data/ESEM2016/crowd-project2-task-training-dataset1.csv", index_cols=None):
    dataset = pd.read_csv(filename, index_col=index_cols)
    group_size = dataset.groupby(cols).size()
    print cols, group_size.max()
    for col in cols:
        group_size = dataset.groupby(col).size()
        print col, group_size.min()


if __name__=="__main__":
    summary_groupby(["WorkerID", "ChallengeID"])