import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from stopper import EarlyStopping
from logger import TrainingLogger, MetricMeters

from loss import PHuberCrossEntropy
from dataset import STTextDataset
from model import BaseModel
from config import Config
from utils import Timer, ensure_path_exists

from transformers import logging

logging.set_verbosity_error()
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

mean = lambda x: sum(x)/len(x)

def train_teacher(config):

    ### Set random seed
    set_random_seed(config.seed)          

    ### Get data loader    
    train_dset = STTextDataset(f'{config.ssl_data_path}/{config.dataset_name}/train.pkl', config.transformers_model_name, config.max_seq_len, mode='train')
    val_dset = STTextDataset(f'{config.ssl_data_path}/{config.dataset_name}/val.pkl', config.transformers_model_name, config.max_seq_len)
    test_dset = STTextDataset(f'{config.ssl_data_path}/{config.dataset_name}/test.pkl', config.transformers_model_name, config.max_seq_len)
    num_classes = train_dset.num_classes

    train_loader = DataLoader(train_dset, batch_size=config.teacher_train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=config.eval_batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size=config.eval_batch_size, pin_memory=True, num_workers=4)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BaseModel(config.transformers_model_name, num_classes=num_classes, emb_dim=config.emb_dim, output_hidden_states=True)
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config.encoder_lr}, 
        {'params': model.fc_emb.parameters(), 'lr': config.fc_lr},
        {'params': model.fc_cls.parameters(), 'lr': config.fc_lr}
    ])
    
    if config.use_data_parallel:
        model = nn.DataParallel(model)
        

    logger = TrainingLogger(dest_path=os.path.join(config.dest_dir, f'teacher_std_log.txt'))

    logger.print('Number of classes: {}'.format(num_classes))
    logger.print('Length of train/val/test datasets: {}, {}, {}'.format(
        len(train_dset), len(val_dset), len(test_dset)))
    logger.print('Length of train/val/test dataloaders: {}, {}, {}'.format(
        len(train_loader), len(val_loader), len(test_loader)))
    logger.print(f'Using {torch.cuda.device_count()} GPUs: '
        + ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    logger.print('Config:\n' + str(config))
    logger.print('\n')


    #------------- Start Training Teacher -------------#
    model = model.to(device)
    metric_meter = MetricMeters()
    epoch_early_stopper = EarlyStopping(model, patience=10, print_fn=logger.print)
    train_steps = 0

    
    while True:
        
        ### Training 

        metric_meter.reset(mode='train')
        train(train_loader, model, optimizer, metric_meter, device, config)
        train_steps += 1
        scores = metric_meter.epoch_get_scores()
        logger.print('Steps {:2d} | Train Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Train Acc: {:5f}%'.format(
            train_steps, scores['total_loss'], scores['class_loss'], scores['metric_loss'], scores['accuracy']*100.0))
 

        ### Validating
        if train_steps >= 10: 

            metric_meter.reset(mode='val')
            val(val_loader, model, metric_meter, device, config)   
            scores = metric_meter.epoch_get_scores()
            logger.print('Steps {:2d} |  Val  Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Val Acc: {:5f}%'.format(
                train_steps, scores['total_loss'], scores['class_loss'], scores['metric_loss'], scores['accuracy']*100.0))

            save_ckpt = epoch_early_stopper(scores['class_loss'], scores['accuracy'])

            if save_ckpt:
                save_checkpoint(model, path=os.path.join(config.dest_dir, '_best_teacher.pth'))
                logger.print(f'Best loss = {scores["class_loss"]:.5f} achieved.Checkpoint saved.')

            if epoch_early_stopper.early_stop == True:
                break  # Exit inner loop
        
            logger.print('\n')


    ### Evaluate with test dataset
    model_path = os.path.join(config.dest_dir, '_best_teacher.pth')
    load_checkpoint(model, model_path)

    metric_meter.reset(mode='test')
    val(test_loader, model, metric_meter, device, config)
    scores = metric_meter.epoch_get_scores()
    logger.print('Test Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Test Acc: {:5f}%'.format(
            scores['total_loss'], scores['class_loss'], scores['metric_loss'], scores['accuracy']*100.0))
    
    
    metric_meter.dump(path=os.path.join(config.dest_dir, f'teacher_metric_log.json'))
    os.rename(
        os.path.join(config.dest_dir, '_best_teacher.pth'),
        os.path.join(config.dest_dir, f'best_teacher.pth')
    )

    logger.print('\n'*5)

    return scores


def set_random_seed(seed):
    #print('Random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)


def train(train_loader, model, optimizer,metric_meter, device, config):

    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader, start=1):
            
        input_ids = batch['input_ids'].to(device) 
        token_type_ids = batch['token_type_ids'].to(device) 
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)

        output = model(input_ids, token_type_ids, attention_mask)                
        logits = output['logits']

        
        ### Calculate class loss
        class_loss = F.cross_entropy(logits, label)
        
        ### Calculate metric loss
        metric_loss = torch.tensor(0.0).to(device)
        protos = output['hidden_state']
        unique_label = torch.unique(label)
        if config.coef_metric == 0.0:
            protos = protos.detach()

        for l in unique_label:
            target_protos = protos[label == l] # (-1, hidden size)
            centroid = torch.mean(target_protos, axis=0) #(hidden_size)
            distance = torch.sum(((target_protos - centroid)**2), axis=1)
            metric_loss += torch.mean(distance, axis=0)
        metric_loss = metric_loss/protos.size(-1) #/len(unique_label)

        loss = class_loss + config.coef_metric*metric_loss
        loss /= len(train_loader)
        loss.backward() 


        metric_meter.update('total_loss', loss.item()*len(train_loader), total=1)
        metric_meter.update('class_loss', class_loss.item(), total=1)
        metric_meter.update('metric_loss', metric_loss.item(), total=1)
        metric_meter.update('accuracy', 
            correct=(logits.argmax(1) == label).sum().item(), total=len(label))

    optimizer.step() 


def val(val_loader, model, metric_meter, device, config):

    model.eval()

    with torch.no_grad():    
        for i, batch in enumerate(val_loader, start=1):
                    
            input_ids = batch['input_ids'].to(device) 
            token_type_ids = batch['token_type_ids'].to(device) 
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            output = model(input_ids, token_type_ids, attention_mask)                
            logits = output['logits']

            
            ### Calculate class loss
            class_loss = F.cross_entropy(logits, label)
            
            ### Calculate metric loss
            metric_loss = torch.tensor(0.0).to(device)
            protos = output['hidden_state']
            unique_label = torch.unique(label)
            if config.coef_metric == 0.0:
                protos = protos.detach()

            for l in unique_label:
                target_protos = protos[label == l] # (-1, hidden size)
                centroid = torch.mean(target_protos, axis=0) #(hidden_size)
                distance = torch.sum(((target_protos - centroid)**2), axis=1)
                metric_loss += torch.mean(distance, axis=0)
            metric_loss = metric_loss/protos.size(-1) 

            loss = class_loss + config.coef_metric*metric_loss

            metric_meter.update('total_loss', loss.item(), total=1)
            metric_meter.update('class_loss', class_loss.item(), total=1)
            metric_meter.update('metric_loss', metric_loss.item(), total=1)
            metric_meter.update('accuracy', 
                correct=(logits.argmax(1) == label).sum().item(), total=len(label))

            print(f'Evaluating {i}/{len(val_loader)}...', end='\r')


def save_checkpoint(model, path):

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path):

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    pass