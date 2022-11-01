"""
for chinese bert
https://github.com/ShannonAI/ChineseBert
拼音和字形信息
"""

from torch.utils.data import Dataset
import torch
from config import CFG
from tqdm import tqdm
from ChineseBert.datasets.bert_dataset import BertDataset

class Sighan13_Chinese_Bert(Dataset):
    def __init__(self, errors, corrects):
        super().__init__()
        assert len(errors)==len(corrects)
        model_path="/data1/2022/cuiwenyao/spelling_correct/saved_model/ChineseBERT-base/"
        self.tokenizer = BertDataset(model_path)

        self.error_input_ids=[]
        self.error_pinyin_ids=[]
        self.error_masks=[]
        self.correct_input_ids=[]
        self.correct_pinyin_ids=[]
        self.correct_masks=[]
        self.detection_labels=[]
        for e, c in tqdm(zip(errors, corrects), total=len(errors)):
            outputs=self.convert_one_example(e, c)
            self.error_input_ids.append(outputs["error_input_id"])
            self.error_pinyin_ids.append(outputs["error_pinyin_id"])
            self.error_masks.append(outputs["error_mask"])
            self.correct_input_ids.append(outputs["correct_input_id"])
            self.correct_pinyin_ids.append(outputs["correct_pinyin_id"])
            self.correct_masks.append(outputs["correct_mask"])
            self.detection_labels.append(outputs["detection_label"])
        

    def convert_one_example(self, error, correct):
        """
        error_input_id: (max_len)
        error_pinyin_id:    (max_len*8)
        error_mask: (max_len)
        """
        # error
        error_input_id, error_pinyin_id, error_mask = self.tokenizer.encode(error, length=CFG.max_len)
        # correct
        correct_input_id, correct_pinyin_id, correct_mask = self.tokenizer.encode(correct, length=CFG.max_len)
        # detection label
        detection_label=[int(not x) for x in [a==b for a,b in zip(error_input_id, correct_input_id)]]
        
        outputs={
            "error_input_id": error_input_id,
            "error_pinyin_id": error_pinyin_id,
            "error_mask": error_mask,
            "correct_input_id": correct_input_id,
            "correct_pinyin_id": correct_pinyin_id,
            "correct_mask": correct_mask,
            "detection_label": detection_label,
        }
        return outputs
    
    def __len__(self,):
        return len(self.error_input_ids)
    
    def __getitem__(self, index):
        error_input_id=self.error_input_ids[index]
        error_pinyin_id=self.error_pinyin_ids[index]
        error_mask=self.error_masks[index]
        correct_input_id=self.correct_input_ids[index]
        correct_pinyin_id=self.correct_pinyin_ids[index]
        correct_mask=self.correct_masks[index]
        detection_label=self.detection_labels[index]
        
        # error_input_id=torch.tensor(error_input_id)
        # error_pinyin_id=torch.tensor(error_pinyin_id)
        # error_mask=torch.tensor(error_mask)
        # correct_input_id=torch.tensor(correct_input_id)
        # correct_pinyin_id=torch.tensor(correct_pinyin_id)
        # correct_mask=torch.tensor(correct_mask)
        
        detection_label=torch.tensor(detection_label)
        
        outputs={
            "error_input_ids": error_input_id,
            "error_pinyin_ids": error_pinyin_id,
            "error_masks": error_mask,
            "correct_input_ids": correct_input_id,
            "correct_pinyin_ids": correct_pinyin_id,
            "correct_masks": correct_mask,
            "detection_labels": detection_label,
        }
        # print(outputs)
        assert all([type(x)==torch.Tensor for x in outputs.values()])
        return outputs