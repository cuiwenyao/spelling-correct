import os
import sys
sys.path.insert(0, os.getcwd())
pwd_path = os.path.abspath(os.path.dirname(__file__))+"/"

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from config import CFG
from utils.util import read_sighan
from dataset.sighan13_dataset_chinese_bert import Sighan13_Chinese_Bert
from model.baseline_chinese_bert import Baseline_Chinese_Bert
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import json

def read_data():
    train_errors=read_sighan(CFG.train13_error_path)
    train_corrects=read_sighan(CFG.train13_correct_path)
    test_errors=read_sighan(CFG.test13_error_path)
    test_corrects=read_sighan(CFG.test13_correct_path)
    
    return train_errors, train_corrects, test_errors, test_corrects
    
def save_eval_result(strings_error, strings_predict, strings_correct, fp):
    
    metrics_result="there is metrics result!!!"
    
    with open(fp, mode="w", encoding="utf-8") as f:
        # eaxmples=[{
        #     "error": error,
        #     "predict": predict,
        #     "correct": correct,
        # } for error, predict, correct in zip(strings_error, strings_predict, strings_correct)]
        examples=[]
        for error, predict, correct in zip(strings_error, strings_predict, strings_correct):
            low=0
            high=error.find("[PAD]")
            error=error[low: high]
            correct=correct[low: high]
            high=predict.find("[PAD]")
            predict=predict[low: high]
            examples.append({
                "error": error,
                "predict": predict,
                "correct": correct,
                "is_righty": 1 if predict[5:-5]==correct[5:-5] else 0,
                })
    
        right_ratio=len([x for x in examples if x["is_righty"]==1])/len(examples)
        metrics_result={
            "right_ratio": right_ratio,
        }
        metrics_result_str=json.dumps(metrics_result, ensure_ascii=False, indent=4)    
        example_str=json.dumps(examples, ensure_ascii=False, indent=4)
        
        all_msg=metrics_result_str+"\n"+example_str
        f.write(all_msg)
        f.close()
        return
    
    
def train_fn(model, batch):
    """
    训练一个batch
    """
    model.train()
    
    outputs=model(batch)
    loss=outputs["loss"]
    loss.backward()
    return loss.detach().item()
    
def eval_fn(model, dev_loader):
    loss_sum=0
    total_num=len(dev_loader)
    strings_error=[]
    strings_predict=[]
    strings_correct=[]
    with torch.no_grad():
        for batch in dev_loader:
            outputs=model(batch)
            loss=outputs["loss"].detach().cpu().item()
            loss_sum+=loss

            strings_error.extend(outputs["strings_error"])
            strings_predict.extend(outputs["strings_predict"])
            strings_correct.extend(outputs["strings_correct"])
            
    save_eval_result(strings_error, strings_predict, strings_correct, CFG.result_dir+f"eval_result_{CFG.model_name}.json")
    loss_avg=loss_sum/total_num
    return loss_avg
            

if __name__== "__main__":
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # 读取数据
    train_errors, train_corrects, test_errors, test_corrects=read_data()
    # dataset
    train_dataset=Sighan13_Chinese_Bert(train_errors, train_corrects)
    test_dataset=Sighan13_Chinese_Bert(test_errors, test_corrects)
    
    
    # model and training setting
    CFG.bert_name="/data1/2022/cuiwenyao/spelling_correct/saved_model/ChineseBERT-base/"
    CFG.model_name="Baseline_Chinese_Bert"
    model=Baseline_Chinese_Bert(CFG).to(CFG.device)
    optimizer=AdamW(model.parameters(), lr=CFG.lr)
    num_warmup_steps=len(train_dataset)
    num_train_step=CFG.max_steps
    scheduler=get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps =num_warmup_steps,
                                                num_training_steps =num_train_step)
    

    # load pretrained model
    # model.load_state_dict(torch.load(f"{CFG.model_save_dir + CFG.model_name}_best.pth"))
    
    # training
    training_record={
        "best_loss": 100000,
        "global_step": 0,
    }
    pbar=tqdm(range(CFG.max_steps), total=CFG.max_steps)
    while 1:
        train_loader=DataLoader(train_dataset, batch_size=CFG.train_bs, shuffle=True)
        for batch in train_loader:
            training_record["global_step"]+=1
            pbar.update()
            
            optimizer.zero_grad()
            loss=train_fn(model, batch)
            optimizer.step()
            scheduler.step()
            
            # eval
            if training_record["global_step"]%CFG.save_every==0:
                # it is model.state_dict() not model!!!
                torch.save(model.state_dict(), f"{CFG.model_save_dir + CFG.model_name}_cur.pth")
                dev_loader=DataLoader(test_dataset, batch_size=CFG.eval_bs)
                eval_loss=eval_fn(model, dev_loader)
                if eval_loss<=training_record["best_loss"]:
                    msg=f"loss: {training_record['best_loss']} >> {eval_loss} saving best at step: {training_record['global_step']}===================="
                    print(msg)
                    training_record["best_loss"]=eval_loss
                    torch.save(model.state_dict(), f"{CFG.model_save_dir + CFG.model_name}_best.pth")
            if training_record["global_step"]>=CFG.max_steps:
                print(f"训练结束！！！！！！！！！")
                exit(0)
            
            pbar.set_postfix_str("loss: %.6f lr: %.6f" % (loss, optimizer.param_groups[0]["lr"]))        
            
            
    