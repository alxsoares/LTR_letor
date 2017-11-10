# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import os

from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from common import cal_group_avg_metric
from common import colors
from common import cal_group_avg_metric
import conf

def prepare():
    if os.path.exists(conf.classify_binary):
        os.popen("rm -rf %s"%conf.classify_binary)    
    if os.path.exists(conf.learn_binary):
        os.popen("rm -rf %s"%conf.learn_binary)    
    
    cmd = os.popen("uname -a").read()
    if "x86_32" in cmd:
        os.popen("cd RankSVM && tar -xzf svm_rank_linux32.tar.gz")
    else:
        os.popen("cd RankSVM && tar -xzf svm_rank_linux64.tar.gz")

def data_preprocess(ranksvm_train_file, ranksvm_test_file):
    """
        对于原始的训练数据特征进行归一化处理
    """
    x_train, y_train, train_query_id = load_svmlight_file(conf.raw_training_data, query_id=True)
    x_test, y_test, test_query_id = load_svmlight_file(conf.raw_test_data, query_id=True)
    
    x_train = x_train.todense()
    x_test = x_test.todense()
    mms = MinMaxScaler().fit(x_train)
    new_x_train = mms.transform(x_train)
    new_x_test = mms.transform(x_test)

    dump_svmlight_file(new_x_train, y_train, ranksvm_train_file, zero_based=False, query_id=train_query_id)
    dump_svmlight_file(new_x_test, y_test, ranksvm_test_file, zero_based=False, query_id=test_query_id)
    
    return y_train, y_test

if __name__ == "__main__":
    prepare()
    y_train, y_test = data_preprocess(conf.normalized_train_data, conf.normalized_test_data)
    
    train_cmd = "%s -c 10 %s %s"%(conf.learn_binary, conf.normalized_train_data, conf.ranksvm_model_path)
    res = os.popen(train_cmd)
    print res.read()

    train_predict_cmd = "%s %s %s %s"%(conf.classify_binary, conf.normalized_train_data, conf.ranksvm_model_path, conf.ranksvm_train_prediction)
    test_predict_cmd = "%s %s %s %s"%(conf.classify_binary, conf.normalized_test_data, conf.ranksvm_model_path, conf.ranksvm_test_prediction)
    
    res = os.popen(train_predict_cmd)
    print res.read()
    res = os.popen(test_predict_cmd)
    print res.read()

    train_pred_score_list = [float(line.strip()) for line in open(conf.ranksvm_train_prediction)] 
    test_pred_score_list = [float(line.strip()) for line in open(conf.ranksvm_test_prediction)]
    
    print "In the training data set: "
    cal_group_avg_metric(y_train, train_pred_score_list, conf.training_data_group_len)
    print "In the testing data set: "
    cal_group_avg_metric(y_test, test_pred_score_list, conf.test_data_group_len)
