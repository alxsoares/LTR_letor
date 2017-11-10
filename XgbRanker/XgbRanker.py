#!/usr/bin/env python
# encoding: utf-8
'''
    Description:
    Author: shelldream
    Date:
'''
import sys
reload(sys).setdefaultencoding('utf-8')
sys.path.append('../BaseRanker')
sys.path.append('../')
import xgboost as xgb
import numpy as np
from common import cal_group_avg_metric
from common import colors

class XgbRanker(object):
    def __init__(self, params, feature_map):
        self.params = params
        self.xgb_model = None
        self.feature_map = feature_map

    def cal_feature_importance(self, booster, importance_type = 'gain'):
        ''' 
            Args:
                booster:
                importance_type:
            Rets:
                sorted_fscores:
                sorted_scores:
        '''
        fscores = booster.get_fscore()
        sorted_fscores = sorted(fscores.items(), key = lambda x: x[1], reverse = True)
        scores = booster.get_score(importance_type = importance_type)
        sorted_scores = sorted(scores.items(), key = lambda x: x[1], reverse = True)
        print colors.BLUE + '-' * 30 + ' The feature importance of each feature ' + '-' * 30 + colors.ENDC
        pos = 0
        for (feature, value) in sorted_fscores:
            print colors.BLUE + '\t\t\t%d\t%s\t%f' % (pos, self.feature_map[feature], value) + colors.ENDC
            pos += 1
        
        print colors.BLUE + '\n\n' + '-' * 30 + ' The feature importance score (%s) ' % importance_type + '-' * 30 + colors.ENDC
        pos = 0
        for (feature, value) in sorted_scores:
            print colors.BLUE + '\t\t\t%d\t%s\t%f' % (pos, self.feature_map[feature], value) + colors.ENDC
            pos += 1
        
        return (sorted_fscores, sorted_scores)

    def pointwise_train(self, x_train, y_train, x_test, y_test, train_grp_len_list, test_grp_len_list, useRegression = True):
        if useRegression:
            self.xgb_model = xgb.XGBRegressor(**None)
            self.xgb_model.fit(x_train, y_train)
            y_train_pred = self.xgb_model.predict(x_train)
            y_test_pred = self.xgb_model.predict(x_test)
        else:
            self.xgb_model = xgb.XGBClassifier(**None)
            new_y_train = np.array([int(item >= 2) for item in y_train])
            self.xgb_model.fit(x_train, new_y_train)
            y_train_pred = np.array([prob[1] for prob in xgb_model.predict_proba(x_train)])
            y_test_pred = np.array([prob[1] for prob in xgb_model.predict_proba(x_test)])
        
        booster = self.xgb_model.booster()
        (fscores, scores) = self.cal_feature_importance(booster)
        print 'In the training data set: '
        cal_group_avg_metric(y_train, y_train_pred, train_grp_len_list)
        print 'In the test data set: '
        cal_group_avg_metric(y_test, y_test_pred, test_grp_len_list)

    def pairwise_train(self, x_train, y_train, x_test, y_test, train_grp_len_list, test_grp_len_list):
        dtrain = xgb.DMatrix(x_train, label = y_train)
        dtrain.set_group(train_grp_len_list)
        dtest = xgb.DMatrix(x_test, label = y_test)
        dtest.set_group(test_grp_len_list)
        booster = xgb.train(params = self.params, dtrain = dtrain, num_boost_round = 150)
        (fscores, scores) = self.cal_feature_importance(booster)
        y_train_pred = booster.predict(dtrain)
        y_test_pred = booster.predict(dtest)
        print 'In the training data set: '
        cal_group_avg_metric(y_train, y_train_pred, train_grp_len_list)
        print 'In the test data set: '
        cal_group_avg_metric(y_test, y_test_pred, test_grp_len_list)
