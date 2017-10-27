# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date: 2017-10-27
"""
import sys
reload(sys).setdefaultencoding("utf-8")

from sklearn.datasets import load_svmlight_file
import xgboost as xgb
import numpy as np

import conf
from common import cal_feature_importance
from common import cal_group_avg_metric

def xgb_pointwise_regression_train(x_train, y_train, x_test, y_test, train_grp_len_f, test_grp_len_f, params):
    """
        利用xgboost回归树实现pointwise的学习方法
    
    """ 
     
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(x_train, y_train)
    fscores, scores = cal_feature_importance(xgb_model)
    y_train_pred = xgb_model.predict(x_train)
    y_test_pred = xgb_model.predict(x_test)
    
    print "In the training data set: "
    cal_group_avg_metric(y_train, y_train_pred, train_grp_len_f) 
    print "In the test data set: "
    cal_group_avg_metric(y_test, y_test_pred, test_grp_len_f) 
     
def xgb_pointwise_classification_train(x_train, y_train, x_test, y_test, train_grp_len_f, test_grp_len_f, params):
    """
        利用xgboost分类树实现pointwise的学习方法
    
    """ 
     
    xgb_model = xgb.XGBClassifier(**params)
    new_y_train = np.array([int(item >= 2) for item in y_train])
    
    xgb_model.fit(x_train, new_y_train)
    fscores, scores = cal_feature_importance(xgb_model)
    y_train_pred = np.array([prob[1] for prob in xgb_model.predict_proba(x_train)])
    y_test_pred = np.array([prob[1] for prob in xgb_model.predict_proba(x_test)])
   
    print "In the training data set: "
    cal_group_avg_metric(y_train, y_train_pred, train_grp_len_f) 
    print "In the test data set: "
    cal_group_avg_metric(y_test, y_test_pred, test_grp_len_f) 

if __name__ == "__main__":
    x_train, y_train = load_svmlight_file(conf.training_data)
    x_test, y_test = load_svmlight_file(conf.test_data)
    #xgb_pointwise_regression_train(x_train, y_train, x_test, y_test, conf.training_data_group_len, conf.test_data_group_len, conf.pw_reg_params)
    xgb_pointwise_classification_train(x_train, y_train, x_test, y_test, conf.training_data_group_len, conf.test_data_group_len, conf.pw_class_params)
