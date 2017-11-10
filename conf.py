# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

#common config
training_data="./tmp/raw_train.txt"
test_data="./tmp/raw_test.txt"

#xgboost config
pointwise_reg_params = {'max_depth':5, 'learning_rate':0.05, 'n_estimators':100, 'silent':True,'objective':'reg:linear', 'gamma':0, 'min_child_weight':1, 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1, 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'base_score':0.5, 'seed':0, 'missing':None}

pointwise_class_params = {'max_depth':5, 'learning_rate':0.05, 'n_estimators':150, 'silent':True,'objective':'binary:logistic', 'gamma':0, 'min_child_weight':1, 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1, 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'base_score':0.5, 'seed':0, 'missing':None}

pairwise_params = {'max_depth':6, 'learning_rate':0.1, 'n_estimators':150, 'silent':True,'objective':'rank:pairwise', 'gamma':0, 'min_child_weight':1, 'max_delta_step':0, 'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1, 'reg_alpha':0, 'reg_lambda':1, 'scale_pos_weight':1, 'base_score':0.5, 'seed':0, 'missing':None}

#RankSVM
learn_binary="RankSVM/svm_rank_learn"
classify_binary="RankSVM/svm_rank_classify"
normalized_train_data="./normalized_train_data.txt"
normalized_test_data="./normalized_test_data.txt"
ranksvm_model_path="model.txt"
ranksvm_train_prediction="ranksvm_train_prediction.txt"
ranksvm_test_prediction="ranksvm_test_prediction.txt"
