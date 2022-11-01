from re import T
from torch.utils.data import Dataset
import torch
from config import CFG
from transformers import BertTokenizer
from tqdm import tqdm


class Sighan13(Dataset):
    def __init__(self, errors, corrects):
        super().__init__()
        assert len(errors)==len(corrects)
        self.tokenizer=BertTokenizer.from_pretrained(CFG.bert_name)
        self.error_ids=[]
        self.error_masks=[]
        self.error_segment_ids=[]
        self.correct_ids=[]
        self.correct_masks=[]
        self.detection_labels=[]
        for e, c in tqdm(zip(errors, corrects), total=len(errors)):
            outputs=self.convert_one_example(e, c)
            self.error_ids.append(outputs["error_id"])
            self.error_masks.append(outputs["error_mask"])
            self.error_segment_ids.append(outputs["error_segment_id"])
            self.correct_ids.append(outputs["correct_id"])
            self.correct_masks.append(outputs["correct_mask"])
            self.detection_labels.append(outputs["detection_label"])
        

    def convert_one_example(self, error, correct):
        # error
        tokenize_out=self.tokenizer(error, padding="max_length", max_length=CFG.max_len, truncation=True)
        error_id=tokenize_out["input_ids"]
        error_mask=tokenize_out["attention_mask"]
        error_segment_id=tokenize_out["token_type_ids"]
        # correct
        tokenize_out=self.tokenizer(correct, padding="max_length", max_length=CFG.max_len, truncation=True)
        correct_id=tokenize_out["input_ids"]
        correct_mask=tokenize_out["attention_mask"]
        # detection label
        detection_label=[int(not x) for x in [a==b for a,b in zip(error_id, correct_id)]]
        
        outputs={
            "error_id": error_id,
            "error_mask": error_mask,
            "error_segment_id": error_segment_id,
            "correct_id": correct_id,
            "correct_mask": correct_mask,
            "detection_label": detection_label,
        }
        return outputs
    def __len__(self,):
        return len(self.error_ids)
    
    def __getitem__(self, index):
        error_id=self.error_ids[index]
        error_mask=self.error_masks[index]
        error_segment_id=self.error_segment_ids[index]
        correct_id=self.correct_ids[index]
        correct_mask=self.correct_masks[index]
        detection_label=self.detection_labels[index]
        
        error_id=torch.tensor(error_id)
        error_mask=torch.tensor(error_mask)
        error_segment_id=torch.tensor(error_segment_id)
        correct_id=torch.tensor(correct_id)
        correct_mask=torch.tensor(correct_mask)
        detection_label=torch.tensor(detection_label)
        
        
                
        outputs={
            "error_ids": error_id,
            "error_masks": error_mask,
            "error_segment_ids": error_segment_id,
            "correct_ids": correct_id,
            "correct_masks": correct_mask,
            "detection_labels": detection_label,
        }
        # print(outputs)
        return outputs