import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer

NO_TOKEN_TYPE_IDS = ('distilbert-base-uncased')


class BaseModel(nn.Module):
    '''
    Sentence embedding using average pooling.
    '''
    def __init__(self, transformers_model_name, num_classes, emb_dim=16, 
            fc_dropout=0.5, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3,
            output_hidden_states=False, output_attentions=False):     
                    
        super().__init__()

        self.transformers_model_name = transformers_model_name

        self.fc_dropout = fc_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

        ### Init config and encoder
        config = AutoConfig.from_pretrained(transformers_model_name, num_labels=num_classes)
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.hidden_dropout_prob = hidden_dropout_prob
        config.return_dict = True
        config.output_hidden_states = False
        config.output_attentions = True
        encoder = AutoModel.from_pretrained(transformers_model_name, config=config)
        
        self.hidden_size = config.hidden_size
        self.emb_dim = emb_dim
        self.encoder = encoder
        # self.scorer = nn.Sequential(
        #     nn.Linear(self.hidden_size, 512),
        #     nn.Linear(512, emb_dim),
        # )
        self.fc_emb = nn.Sequential(
            nn.Linear(self.hidden_size, emb_dim)
        )
        self.fc_cls = nn.Sequential(
            nn.Dropout(fc_dropout), 
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'distilbert' not in self.transformers_model_name:
            kwargs['token_type_ids'] = token_type_ids

        encoded_results = self.encoder(**kwargs)
        
        cls_state = torch.mean(encoded_results['last_hidden_state'], dim=1)         
        embedding = self.fc_emb(cls_state)
        logits = self.fc_cls(embedding)
        output = {'logits': logits}

        if self.output_attentions:
            output['attention'] = encoded_results['attentions']
        
        if self.output_hidden_states:
            output['hidden_state'] = embedding
        
        return output


    
    def load(self, path, device='cuda'):
        self.load_state_dict(torch.load(path, map_location=device))



if __name__ == '__main__':

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    transformers_model_name = 'bert-base-uncased'
    num_classes = 10
    # config = AutoConfig.from_pretrained(transformers_model_name)
    # auto_model = AutoModel.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(transformers_model_name)
    model = BaseModel(transformers_model_name, num_classes)

    inputs = tokenizer('Hello World!', return_tensors='pt')
    print('Inputs:', inputs)

    outputs = model(**inputs)
    print('Output:', outputs)
