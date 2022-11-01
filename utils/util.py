"""
utils
"""
import os
from statistics import mode

def read_sighan(fp):
    with open(fp, mode="r", encoding="utf-8") as f:
        data_list=[]
        for x in f.readlines():
            x=x.strip()
            if len(x)>=1:
                data_list.append(x)
        f.close()
        return data_list
