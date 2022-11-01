"""
Describe:
bert-base-chinese 使用detection、correction结构。
"""

import torch
import torch.nn as nn
from transformers import AutoModel, BertForMaskedLM, AutoTokenizer


class Baseline_Bert_02(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print(cfg.bert_name)
        assert cfg.bert_name=="bert-base-chinese"
        self.bert=BertForMaskedLM.from_pretrained(cfg.bert_name)
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.bert_name)
        self.device=cfg.device
        
        """
        net
        """
        self.detector=nn.Linear(self.bert.hidden_size, 2)
        
        """
        loss
        """
        self.Loss=torch.nn.CrossEntropyLoss()
        
    def forward(self, batch):
        """
        输入错误的和正确的，直接使用MLM的loss
        准备 batch 
        """
        error_ids=batch["error_ids"].to(self.device)     #(bs, max_len)
        error_masks=batch["error_masks"].to(self.device)
        correct_ids=batch["correct_ids"].to(self.device)
        correct_masks=batch["correct_masks"].to(self.device)
        detection_labels=batch["detection_labels"].to(self.device)
        
        
        
        bert_outputs=self.bert(input_ids=error_ids, attention_mask=error_masks, return_dict=True, labels=correct_ids)

        logits =bert_outputs["logits"]
        
        loss =bert_outputs["loss"]
        # loss=self.loss_fn(logits, correct_ids)
        strings_error=self.ids_2_str(error_ids)
        strings_predict=self.ids_prob_2_str(logits)
        strings_correct=self.ids_2_str(correct_ids)
        
        outputs={
            "loss": loss,
            "logits": logits,
            "strings_error": strings_error,
            "strings_predict": strings_predict,
            "strings_correct": strings_correct,
        }
        return outputs
    
    def ids_prob_2_str(self, ids:torch.Tensor):
        ids=torch.argmax(ids, dim=-1).detach().cpu().tolist()   #(bs, max_len)
        strings=["".join(self.tokenizer.convert_ids_to_tokens(x)) for x in ids]
        return strings

    def ids_2_str(self, ids:torch.Tensor):
        strings=["".join(self.tokenizer.convert_ids_to_tokens(x)) for x in ids]
        return strings
        
    def loss_fn(self, pred_logits, correct_ids):
        loss=self.Loss(pred_logits.permute(0,2,1), correct_ids)
        return loss