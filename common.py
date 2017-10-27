# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
from ranking_metrics import cal_ndcg 

feature_map = {}
with open("fmap.txt", "r") as fr:
    for line in fr:
        content = line.rstrip().split("\t")
        try:
            feature_map["f"+content[0]] = content[1] + "--" + content[2]
        except:
            feature_map["f"+content[0]] = content[1]


class colors:
    BLUE = '\033[01;34m'
    GREEN = '\033[01;32m'
    RED = '\033[01;31m'
    YELLOW = '\033[01;33m'
    ENDC = '\033[00m'


def cal_feature_importance(xgb_model, importance_type="gain"):
    """ 
        返回按照特征重要性的分析结果
        Args:
            importance_type: str, 特征重要性类型, 目前只支持 "weight"、"gain"、"cover"
        Rets:
            sorted_fscores:
            sorted_scores:
    """
    fscores = xgb_model.booster().get_fscore()
    sorted_fscores = sorted(fscores.items(), key=lambda x:x[1], reverse=True) 
    scores = xgb_model.booster().get_score(importance_type=importance_type)
    sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    print colors.BLUE + "-"*30 + " The feature importance of each feature " + "-"*30 + colors.ENDC
    pos = 0
    for (feature, value) in sorted_fscores:
        print colors.BLUE + "\t\t\t%d\t%s\t%f"%(pos, feature_map[feature], value) + colors.ENDC
        pos += 1
    
    print colors.BLUE + "\n\n" + "-"*30 + " The feature importance score (%s) "%importance_type + "-"*30 + colors.ENDC
    pos = 0
    for (feature, value) in sorted_scores:
        print colors.BLUE + "\t\t\t%d\t%s\t%f"%(pos, feature_map[feature], value) + colors.ENDC
        pos += 1

    return sorted_fscores, sorted_scores


def cal_group_avg_metric(score_list, pred_score_list, group_len_filename):
    group_len_list = [int(line.strip()) for line in open(group_len_filename)]
    group_cnt = len(group_len_list)

    instance_cnt = len(score_list)
    if instance_cnt != sum(group_len_list):
        raise ValueError("The instance count does not match the group length info!!")

    begin = 0
    ndcg_at5_sum = 0.0 
    ndcg_at10_sum = 0.0 
    for grp_len in group_len_list:
        sub_score_list = score_list[begin:begin+grp_len]
        sub_pred_score_list = pred_score_list[begin:begin+grp_len]
        begin += grp_len
        
        ndcg_at5_sum += cal_ndcg(sub_score_list, sub_pred_score_list, 5)
        ndcg_at10_sum += cal_ndcg(sub_score_list, sub_pred_score_list, 10)

    print colors.RED + "average ndcg@5: %f"%(ndcg_at5_sum/group_cnt) + colors.ENDC
    print colors.RED + "average ndcg@10: %f"%(ndcg_at10_sum/group_cnt) + colors.ENDC
