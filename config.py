"""
config
"""
import os
import sys
pwd_path = os.path.abspath(os.path.dirname(__file__))+"/"
import torch
from gpu_info import get_min_used_gpu_id

class CFG:
    data_path=pwd_path + "data/sighan/simplified/"
    train13_correct_path=data_path + "train13_correct.txt"
    train13_error_path=data_path + "train13_error.txt"
    test13_correct_path=data_path + "test13_correct.txt"
    test13_error_path=data_path + "test13_error.txt"

    """
    training setting
    """
    seed=2022
    device="cuda:0"
    # device=torch.device(f"cuda:{get_min_used_gpu_id('memory')}" if torch.cuda.is_available() else "cpu")
    print(f"We are using device: {device}")
    """
    model_name: Baseline_Bert Baseline_Chinese_Bert
    """
    model_name="Baseline_Bert"  
    """
    bert_name list: bert-base-chinese, 
    """
    bert_name="bert-base-chinese"
    lr=1e-5
    max_len=128
    max_steps=30000
    save_every=1000
    train_bs=4
    eval_bs=32
    
    """
    model save
    """
    model_save_dir=pwd_path+"saved_model/"
    
    """
    result dir 
    """
    result_dir=pwd_path+"result/"

if __name__=="__main__":
    print(f"We are using device: {CFG.device}")