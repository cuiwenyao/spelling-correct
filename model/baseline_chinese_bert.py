"""
Describe:
简单的MLM预测，没有使用detection、correction结构。
"""

import torch
import torch.nn as nn
import sys
sys.path.append("../") 
from ChineseBert.models.modeling_glycebert import GlyceBertForMaskedLM
from ChineseBert.datasets.bert_dataset import BertDataset


class Baseline_Chinese_Bert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print(cfg.bert_name)
        if not cfg.bert_name == "/data1/2022/cuiwenyao/spelling_correct/saved_model/ChineseBERT-base/":
            cfg.bert_name="/data1/2022/cuiwenyao/spelling_correct/saved_model/ChineseBERT-base/"
            print(f"你没有正确指定chinese bert的路径,已自动帮你更改为{cfg.bert_name}")

        self.bert=GlyceBertForMaskedLM.from_pretrained(cfg.bert_name)
        self.tokenizer = BertDataset(cfg.bert_name)
        self.device=cfg.device
        """
        loss
        """
        self.Loss=torch.nn.CrossEntropyLoss()
        
    def forward(self, batch):
        """
        输入错误的和正确的，直接使用MLM的loss
        准备 batch 
        """
        error_input_ids=batch["error_input_ids"].to(self.device)     #(bs, max_len)
        error_pinyin_ids=batch["error_pinyin_ids"].to(self.device)
        error_masks=batch["error_masks"].to(self.device)
        correct_input_ids=batch["correct_input_ids"].to(self.device)     #(bs, max_len)
        correct_pinyin_ids=batch["correct_pinyin_ids"].to(self.device)
        correct_masks=batch["correct_masks"].to(self.device)
        detection_labels=batch["detection_labels"].to(self.device)
        
        ################
        #   shape
        ################
        bs=error_input_ids.shape[0]
        length=error_input_ids.shape[1]
        error_pinyin_ids=error_pinyin_ids.view(bs, length, 8)  #(bs, max_len, 8)
        correct_pinyin_ids=correct_pinyin_ids.view(bs, length, 8)  #(bs, max_len, 8)
        
        
        """
                return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        """
        bert_outputs=self.bert.forward(input_ids=error_input_ids, 
                                       pinyin_ids=error_pinyin_ids,
                                       attention_mask=error_masks, 
                                       return_dict=True, labels=correct_input_ids)

        logits =bert_outputs["logits"]
        
        loss =bert_outputs["loss"]
        strings_error=self.ids_2_str(error_input_ids)
        strings_predict=self.ids_prob_2_str(logits)
        strings_correct=self.ids_2_str(correct_input_ids)
        
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