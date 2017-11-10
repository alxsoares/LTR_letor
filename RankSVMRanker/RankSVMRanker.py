# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import os 

import conf

class RankSVMRanker(object):
    def __init__(self):
        if os.path.exists(conf.classify_binary):
            os.popen("rm -rf %s"%conf.classify_binary)    
        if os.path.exists(conf.learn_binary):
            os.popen("rm -rf %s"%conf.learn_binary)    
        
        cmd = os.popen("uname -a").read()
        if "x86_32" in cmd:
            os.popen("cd RankSVMRanker && tar -xzf svm_rank_linux32.tar.gz")
        else:
            os.popen("cd RankSVMRanker && tar -xzf svm_rank_linux64.tar.gz")
        
        self.classify_binary = conf.classify_binary
        self.learn_binary = conf.learn_binary


