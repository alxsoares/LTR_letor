# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import conf
from XgbRanker import XgbRanker 
from RankSVMRanker import RankSVMRanker

def load_fmap(fmap="fmap.txt"):
    feature_map = {}
    with open(fmap, "r") as fr:
        for line in fr:
            content = line.rstrip().split("\t")
            try:
                feature_map["f"+content[0]] = content[1] + "--" + content[2]
            except:
                feature_map["f"+content[0]] = content[1]
    return feature_map

def load_raw_data(data_file):
    x_data, y_data, query_id_list = load_svmlight_file(data_file, query_id=True)
    data_grp_len = []
    query_cnt = 0
    last_query_id = None
    for query_id in query_id_list:
        if query_id != last_query_id:
            if last_query_id is not None:
                data_grp_len.append(query_cnt)
            query_cnt = 1
        else:
            query_cnt += 1
        last_query_id = query_id
    if query_cnt != 0:
        data_grp_len.append(query_cnt)
    return x_data, y_data, data_grp_len

def xgb_pointwise_regression(feature_map):
    xgb_ranker = XgbRanker.XgbRanker(conf.pointwise_reg_params, feature_map)
    x_train, y_train, train_grp_len = load_raw_data(conf.training_data) 
    x_test, y_test, test_grp_len = load_raw_data(conf.test_data) 
    xgb_ranker.pointwise_train(x_train, y_train, x_test, y_test, train_grp_len, test_grp_len, useRegression=True)

def xgb_pointwise_classification(feature_map):
    xgb_ranker = XgbRanker.XgbRanker(conf.pointwise_class_params, feature_map)
    x_train, y_train, train_grp_len = load_raw_data(conf.training_data) 
    x_test, y_test, test_grp_len = load_raw_data(conf.test_data) 
    xgb_ranker.pointwise_train(x_train, y_train, x_test, y_test, train_grp_len, test_grp_len, useRegression=False)

def xgb_pairwise_train(feature_map):
    xgb_ranker = XgbRanker.XgbRanker(conf.pairwise_params, feature_map) 
    x_train, y_train, train_grp_len = load_raw_data(conf.training_data) 
    x_test, y_test, test_grp_len = load_raw_data(conf.test_data) 
    xgb_ranker.pairwise_train(x_train, y_train, x_test, y_test, train_grp_len, test_grp_len)

def ranksvm_train():
    svm_ranker = RankSVMRanker.RankSVMRanker() 

if __name__ == '__main__':
    ranksvm_train() 
    #feature_map = load_fmap()
    #xgb_pointwise_regression(feature_map)
    #xgb_pointwise_classification(feature_map)
    #xgb_pairwise_train(feature_map)
