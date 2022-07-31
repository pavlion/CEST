import os
import six
import copy
import random
import pickle
# import logging
from collections import defaultdict

from torch.utils.data import Dataset
from transformers import AutoTokenizer

# logger = logging.getLogger('Dataset')
mean = lambda lst: sum(lst)/len(lst)

class STTextDataset(Dataset): #dataset for self-training

    def __init__(self, data_path, transformers_model_name, max_seq_len=512, parse_data=True, mode='test', **kwargs):
        
        self.mode = mode
        self.data_path = data_path
        self.transformers_model_name = transformers_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformers_model_name)
        self.max_seq_len = max_seq_len

        if parse_data:
            self.data, self.num_classes = self._parse_data(data_path)
        else:
            ### Cloning the dataset using copy.deepcopy keep breaking when copying the tokenizer
            ### Use the `clone` member function instead
            assert 'data' in kwargs, "`data` should be provided when parse_data=False"
            assert 'num_classes' in kwargs, "`num_classes` should be provided when parse_data=False"
            self.data = kwargs['data']
            self.num_classes = kwargs['num_classes']


    def __getitem__(self, index):
        
        data = self.data.iloc[index]
        X = self.tokenizer(data['text'], padding='max_length', truncation=True, return_tensors='pt',
            add_special_tokens=True, max_length=self.max_seq_len, return_token_type_ids=True, return_attention_mask=True)

        return {
            "input_ids": X["input_ids"].squeeze(0), 
            "token_type_ids": X["token_type_ids"].squeeze(0), 
            "attention_mask": X["attention_mask"].squeeze(0),
            "label": data['label'],
            "pseudo_label": data['pseudo_label'],
            "variance": data['variance'],
            "index": data['index']
        }


    def __len__(self):

        return len(self.data)


    def subseting_dataset(self, indices):

        self.old_data = copy.deepcopy(self.data)
        new_data = []
        for idx in indices:
            new_data.append(self.data[idx])

        self.data = new_data
        
        return self


    def update_data(self, index, content):   
        
        #if isinstance(index, list)
        # original_pl = self.data.iloc[index]['pseudo_label']

        for key in content.keys():
            self.data[key][index] = content[key]
        
        # print(self.data['label'][index], self.data['pseudo_label'][index], self.data['text'][index])
        # print(self.data.iloc[index])
        # print("\n")

        # after_pl = self.data.iloc[index]['pseudo_label']
        # print(f"{original_pl}, {after_pl}", end=" / ") 


    def clone(self):
        #return copy.deepcopy(self)        
        
        return STTextDataset(self.data_path, self.transformers_model_name, 
            parse_data=False, data=self.data, num_classes=self.num_classes)
        

    def _parse_data(self, data_path):

        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
            dataset = dataset[['text', 'label']]  # re-order
            
            dataset['pseudo_label'] = 0
            dataset['variance'] = 0.0 # broadcasting for placeholder
            
            # print("[Before]", list(dataset.index)[0:20])
            dataset.reset_index(inplace=True)
            dataset['index'] = dataset.index
            # print("[After]", dataset['index'][0:20])

        num_classes = 0
        if self.mode == 'train':
            label_set = set()
            word_len = [0]*len(dataset)
            for i, row in enumerate(dataset.itertuples()):
                text, label = row.text, row.label
                label_set.add(label)

                # word_len[i] = 0
                X = self.tokenizer(text, padding=False, truncation=False, add_special_tokens=True, 
                        return_token_type_ids=False, return_attention_mask=False)

                word_len[i] = len(X['input_ids'])
            
            num_classes = len(label_set)
            print(f"Text length (max/min/avg): {max(word_len)}/{min(word_len)}/{mean(word_len)}")

        return dataset, num_classes



def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")





if __name__ == '__main__':

    import torch
    import numpy as np
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    from sampler import IndicesSampler
    transformers_model_name = 'bert-base-uncased'
    
    test_dset = STTextDataset(
        "ssl_data/split1/ag_news/train.pkl", 
        transformers_model_name,         
        max_seq_len=20, 
        mode='train'
    )
    # print(test_dset[0])
    
    
    ### Test single label assignment
    ids = [0, 15, 60, 99]
    pls = [20, 21, 22, 23]
    for idx, label in zip(ids, pls):
        test_dset.update_data(idx, {"pseudo_label": label})

    for idx in ids:
        print(idx, test_dset[idx]["index"], test_dset[idx]["label"], test_dset[idx]["pseudo_label"])

    unlabeled_data_loader = DataLoader(test_dset, 
        batch_size=1, sampler=IndicesSampler(ids, shuffle=False))

    for batch in unlabeled_data_loader:
        print(batch["index"], batch["label"], batch["pseudo_label"])

    # test_dset.assign_label([0, 1, 2, 3], [11, 1e-10, 1e10, -100])

    # test_loader = DataLoader(test_dset, batch_size=4, shuffle=False)
    # batch = next(iter(test_loader))
    # input_ids = batch['input_ids']
    # attn_mask = batch['attention_mask']
    # label = batch['label']
    # print(input_ids.shape, attn_mask.shape, label)

    # test_dset.assign_label(0, 20)
    # # test_loader = DataLoader(test_dset, batch_size=4, shuffle=False)
    # batch = next(iter(test_loader))
    # input_ids = batch['input_ids']
    # attn_mask = batch['attention_mask']
    # label = batch['label']
    # print(input_ids.shape, attn_mask.shape, label) 
    # print(test_dset.data.head())

    
    

    # test_loader = DataLoader(test_dset, batch_size=4, shuffle=False)
    # batch = next(iter(test_loader))
    # input_ids = batch['input_ids']
    # attn_mask = batch['attention_mask']
    # label = batch['label']
    # print(input_ids.shape, attn_mask.shape, label) 
    

    