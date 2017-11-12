# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import lightgbm as lgb
import numpy as np
from common import cal_group_avg_metric
from common import colors


class LambdaMartRanker(object):
    def __init__(self, params):
        self.params = params
    
    def train(self, x_train, y_train, x_test, y_test, train_grp_len_list, test_grp_len_list):
        lgb_train = lgb.Dataset(x_train, y_train, group=train_grp_len_list, free_raw_data=False)
        lgb_test = lgb.Dataset(x_test, y_test, group=test_grp_len_list, free_raw_data=False)
        gbm = lgb.train(self.params, lgb_train, num_boost_round=150)
        y_train_pred = gbm.predict(x_train, num_iteration=gbm.best_iteration)
        y_test_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
        print 'In the training data set: '
        cal_group_avg_metric(y_train, y_train_pred, train_grp_len_list)
        print 'In the test data set: '
        cal_group_avg_metric(y_test, y_test_pred, test_grp_len_list)


