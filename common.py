# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
from ranking_metrics import cal_ndcg 
from ranking_metrics import cal_err 


class colors:
    BLUE = '\033[01;34m'
    GREEN = '\033[01;32m'
    RED = '\033[01;31m'
    YELLOW = '\033[01;33m'
    ENDC = '\033[00m'


def cal_group_avg_metric(score_list, pred_score_list, group_len_list):
    group_cnt = len(group_len_list)

    instance_cnt = len(score_list)
    if instance_cnt != sum(group_len_list):
        raise ValueError("The instance count does not match the group length info!!")

    begin = 0
    ndcg_at5_sum = 0.0 
    ndcg_at10_sum = 0.0 
    err_at5_sum = 0.0
    err_at10_sum = 0.0

    for grp_len in group_len_list:
        sub_score_list = score_list[begin:begin+grp_len]
        sub_pred_score_list = pred_score_list[begin:begin+grp_len]
        begin += grp_len
        
        ndcg_at5_sum += cal_ndcg(sub_score_list, sub_pred_score_list, 5)
        ndcg_at10_sum += cal_ndcg(sub_score_list, sub_pred_score_list, 10)
        err_at5_sum += cal_err(sub_score_list, sub_pred_score_list, 5)
        err_at10_sum += cal_err(sub_score_list, sub_pred_score_list, 10)

    print colors.RED + "average ndcg@5: %f"%(ndcg_at5_sum/group_cnt) + colors.ENDC
    print colors.RED + "average ndcg@10: %f"%(ndcg_at10_sum/group_cnt) + colors.ENDC
    print colors.RED + "average err@5: %f"%(err_at5_sum/group_cnt) + colors.ENDC
    print colors.RED + "average err@10: %f"%(err_at10_sum/group_cnt) + colors.ENDC
