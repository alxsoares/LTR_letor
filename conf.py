# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

training_data="./train.txt"
training_data_group_len="./train_len.txt"
test_data="./test.txt"
test_data_group_len="./test_len.txt"

pw_reg_params = {'max_depth':5, 'learning_rate':0.05, 'n_estimators':100, 'silent':True,'objective':'reg:linear', 'gamma':0, 'min_child_weight':1, 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1, 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'base_score':0.5, 'seed':0, 'missing':None}

pw_class_params = {'max_depth':5, 'learning_rate':0.05, 'n_estimators':150, 'silent':True,'objective':'binary:logistic', 'gamma':0, 'min_child_weight':1, 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1, 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'base_score':0.5, 'seed':0, 'missing':None}
