import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from logger import TrainingLogger, MetricMeters

from dataset import STTextDataset
from model import BaseModel
from sampler import AugmentedDataSelector, IndicesSampler
from stopper import EarlyStopping
from config import Config
from loss import PHuberCrossEntropy, TaylorCrossEntropyLoss
from utils import Timer, ensure_path_exists

from transformers import logging
import pandas as pd
logging.set_verbosity_error()
pd.options.mode.chained_assignment = None  # default='warn'

os.environ['TOKENIZERS_PARALLELISM'] = 'False'

mean = lambda x: sum(x)/len(x)
logger = None

def train_student(config):

    ### Set random seed
    set_random_seed(config.seed)  

    ### Set up logger
    global logger
    logger = TrainingLogger(dest_path=os.path.join(config.dest_dir, f'student_std_log.txt'))
                

    ### Get data loader    
    train_dset = STTextDataset(f'{config.ssl_data_path}/{config.dataset_name}/train.pkl', config.transformers_model_name, config.max_seq_len, mode='train')
    val_dset = STTextDataset(f'{config.ssl_data_path}/{config.dataset_name}/val.pkl', config.transformers_model_name, config.max_seq_len)
    unlabeled_dset = STTextDataset(f'{config.ssl_data_path}/{config.dataset_name}/unsupervised.pkl', config.transformers_model_name, config.max_seq_len)
    test_dset = STTextDataset(f'{config.ssl_data_path}/{config.dataset_name}/test.pkl', config.transformers_model_name, config.max_seq_len)
    num_classes = train_dset.num_classes

    train_loader = DataLoader(train_dset, batch_size=config.teacher_train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=config.eval_batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size=config.eval_batch_size, pin_memory=True, num_workers=4)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BaseModel(config.transformers_model_name, num_classes=num_classes, 
        emb_dim=config.emb_dim, output_hidden_states=True)
    load_checkpoint(model, config.teacher_model_path)
 
    optimizer = get_optimizer(model, opt_type='AdamW', lr=config.finetune_lr)

    if config.use_data_parallel:
        model = nn.DataParallel(model)
     

    unlabeled_data_sampler = AugmentedDataSelector(
        unlabeled_dset, 
        num_evaluate_pool_data=config.unlabeled_sample_size*4,
        num_augmented_data=config.unlabeled_sample_size, 
        eval_batch_size=config.eval_batch_size, 
        mc_dropout_iters=config.mc_dropout_iters, 
        sampling_scheme=config.sampling_scheme, 
        majority_votes=config.majority_votes
    )  
    
    
    logger.print('Number of classes: {}'.format(num_classes))
    logger.print('Length of train/val/unlabeled/test datasets: {}, {}, {}, {}'.format(
        len(train_dset), len(val_dset), len(unlabeled_dset), len(test_dset)))
    logger.print('Length of train/val/unlabeled/test dataloaders: {}, {}, {}, {}'.format(
        len(train_loader), len(val_loader), int(len(unlabeled_dset)/config.unlabeled_sample_size)+1, len(test_loader)))
    logger.print(f'Using {torch.cuda.device_count()} GPUs:'
        + ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]) )
    logger.print('Config:\n' + str(config))
    logger.print('\n')   


    #------------- Start of Self-training -------------#
    model = model.to(device)
    model.device = device
    self_training_early_stopper = EarlyStopping(model, patience=5, delta=0.008, print_fn=logger.print)
    best_model_path = os.path.join(config.dest_dir, 'best_student.pth')
    metric_meter = MetricMeters()
    timer = Timer()
    best_acc = 0.0
    epoch = 1
    
    
    metric_meter.reset(mode=f'teacher_val')
    val(val_loader, model, metric_meter, config) 
    teacher_scores = metric_meter.epoch_get_scores()
    logger.print('Before training: Test Loss: {:5f} | Test Acc: {:5f}%'.format(
            teacher_scores['total_loss'], teacher_scores['accuracy']*100.0))
    
    
    while epoch < config.num_epoch + 1:

        config.epoch = epoch
        logger.print('----- Start of Self Training Epoch {:2d} -----'.format(epoch))
        

        #------------- Select samples from unlabeled dataset -------------#
        
        ### Load best model
        if os.path.exists(best_model_path):
            load_checkpoint(model, best_model_path)

        selected_ids, pseudo_labels, conf_scores, y_var = unlabeled_data_sampler.select_samples(model)

        if len(selected_ids) == 0:            
            logger.print('No sample is qualified for student training.')
            break
        else:
            logger.print(f'{len(selected_ids)} samples are selected. ')
        
        logger.print(f'length of finished ids: {len(unlabeled_data_sampler.finished_ids)}')
        logger.print('Selected uncertainty (max/min/avg): ' + 
                f'{conf_scores.max():.4f}/{conf_scores.min():.4f}/{conf_scores.mean():.4f}')


        # Update dataset with pseudo-labels and predicted variance
        correct, total = 0, 0
        for index, label, var in zip(selected_ids, pseudo_labels, y_var):     
            true_label = unlabeled_dset[index]['label']       
            if true_label != -1:
                total += 1
                if true_label == label: correct += 1

            unlabeled_dset.update_data(index, content={'pseudo_label': label}) #, 'variance': var[label]})

        logger.print(f'Pseudo label accuracy: {correct/total*100.0:.3f}%')

        unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_dset, 
            batch_size=config.train_batch_size, sampler=IndicesSampler(selected_ids, shuffle=True))
            
        
        
        ### Reset model and optimizer each epoch 
        optimizer = get_optimizer(model, lr=config.finetune_lr)
        if os.path.exists(best_model_path): # No best model at first iteration
            load_checkpoint(model, best_model_path)
        else:
            load_checkpoint(model, config.teacher_model_path)
            


        #------------- Training with the selected unlabeled dataset -------------# 
        epoch_early_stopper = EarlyStopping(model, patience=10, print_fn=logger.print)
        verbose = True
        train_steps = 0
        while True:
            
            metric_meter.reset(mode=f'train_epoch{epoch}')
            train(unlabeled_data_loader, model, optimizer, metric_meter, config)
            train_steps += 1

            scores = metric_meter.epoch_get_scores()
            logger.print('Steps {:2d} | Train | Loss(total/cls/ctr): {:5f}, {:5f}, {:5f} | Acc: {:5f}%'.format(
                train_steps, scores['total_loss'], scores['class_loss'], scores['contrastive_loss'], scores['accuracy']*100.0), show=verbose)
    

            ### Validating 

            metric_meter.reset(mode=f'val_epoch{epoch}')
            val(val_loader, model, metric_meter, config)    
            
            scores = metric_meter.epoch_get_scores()
            logger.print('Steps {:2d} |  Val  | Loss(total/cls/ctr): {:5f}, {:5f}, {:5f} | Acc: {:5f}%'.format(
                train_steps, scores['total_loss'], scores['class_loss'], scores['contrastive_loss'], scores['accuracy']*100.0), show=verbose)

        
            save_ckpt = epoch_early_stopper(scores['total_loss'], scores['accuracy'])

            if save_ckpt:
                save_checkpoint(model, path=os.path.join(config.dest_dir, '_best_student_epoch.pth'))
                logger.print(f'Best loss = {scores["total_loss"]:.5f} achieved. Checkpoint saved.')

            if epoch_early_stopper.early_stop == True:
                break  # Exit inner loop
                
    
        save_ckpt = self_training_early_stopper(epoch_early_stopper.min_loss, epoch_early_stopper.best_acc)

        if save_ckpt and self_training_early_stopper.counter == 0:
            
            if os.path.exists(best_model_path):
                os.remove(best_model_path)

            os.rename(
                os.path.join(config.dest_dir, '_best_student_epoch.pth'),
                best_model_path
            )
            logger.print(f'Save model to {best_model_path}.')

        
        logger.print('----- End of Self Training Epoch {:2d} -----'.format(epoch))
        logger.print('Self Training Epoch {:2d}: min_loss = {:.5f}, best_acc = {:.5f}'.format(
            epoch, epoch_early_stopper.min_loss, epoch_early_stopper.best_acc*100.0))
        logger.print('Current best: min_loss = {:.5f}, best_acc = {:.5f}'.format(
            self_training_early_stopper.min_loss, self_training_early_stopper.best_acc*100.0))
        logger.print('Self Training Epoch time: {}/{}'.format(timer.measure(p=1), 
            timer.measure(p=epoch/config.num_epoch)))
        logger.print('\n\n')


        epoch += 1
        
        if self_training_early_stopper.early_stop == True:
            break  # Exit outer(self-training) loop


    logger.print('\n\n-------- Summary of Self Training --------')
    logger.print('Teacher model: Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Acc: {:5f}%'.format(
        teacher_scores['total_loss'], teacher_scores['class_loss'], teacher_scores['contrastive_loss'], teacher_scores['accuracy']*100.0))
    logger.print('Best performance so far: min_loss = {:.5f}, best_acc = {:.5f}'.format(
        self_training_early_stopper.min_loss, self_training_early_stopper.best_acc*100.0))
    logger.print('Self Training Total time: {}/{}\n\n'.format(timer.measure(p=1), 
        timer.measure(p=epoch/config.num_epoch)))
    
    if os.path.exists(os.path.join(config.dest_dir, '_best_student_epoch.pth')):
        os.remove(os.path.join(config.dest_dir, '_best_student_epoch.pth'))


    #------------- Evaluate with the test dataset -------------# 
    load_checkpoint(model, best_model_path)
    metric_meter.reset(mode='test', clear=True)
    val(test_loader, model, metric_meter, config)
        
    scores = metric_meter.epoch_get_scores()
    logger.print('Test | Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Acc: {:5f}%'.format(
            scores['total_loss'], scores['class_loss'], scores['contrastive_loss'], scores['accuracy']*100.0))
    
    
    metric_meter.dump(path=os.path.join(config.dest_dir, f'student_metric_log.json'))
    os.rename(
        best_model_path,
        os.path.join(config.dest_dir, f'best_student.pth')
    )

    return scores


def train(train_loader, model, optimizer, metric_meter, config):

    device = model.device
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader, start=1):
            
        input_ids = batch['input_ids'].to(device) 
        token_type_ids = batch['token_type_ids'].to(device) 
        attention_mask = batch['attention_mask'].to(device)
        label = batch['pseudo_label'].to(device)
        
        selected_ids = []
        
        output = model(input_ids, token_type_ids, attention_mask)  
        logits = output['logits']    

        if config.loss_opt == 'PHuberCrossEntropy':
            class_loss = PHuberCrossEntropy(logits, label, tau=config.tau, reduction='none')
        if config.loss_opt == 'TaylorCrossEntropyLoss':
            class_loss = TaylorCrossEntropyLoss(logits, label, reduction='none', n=6)
        else:
            class_loss = F.cross_entropy(logits, label, reduction='none')
        
        class_loss = class_loss.mean()

        
        ### Calculate contrastive loss
        contrastive_loss = torch.tensor(0.0).to(device)
        protos = output['hidden_state'] # (batch_size, hidden_size)
        if config.coef_contrastive  == 0.0:
            protos = protos.detach()

        else:

            if config.contrastive_type == 'SNTG':
                margin = 1.0
                batch_size = protos.size(0)
                feature1 = protos[ : batch_size//2]
                feature2 = protos[batch_size//2 : (batch_size//2)*2]
                label1 = label[ : batch_size//2]
                label2 = label[batch_size//2 : (batch_size//2)*2 ]
                neighbor_mask = (label1 == label2).float()
                l2_distance = torch.mean(torch.square(feature1-feature2), axis=1)

                pos = neighbor_mask * l2_distance
                neg = (1.0 - neighbor_mask) * torch.square(torch.clamp(margin-l2_distance, min=0))
                contrastive_loss = torch.mean((pos + neg))

            elif config.contrastive_type == 'Reliable':

                margin = 1.0
                num_contrastive_samples = 2
                variance = batch['variance']
                batch_size = batch['variance'].size(0)

                for ids in range(batch_size):
                    
                    if ids in selected_ids:
                        continue

                    current_proto = protos[ids]
                    current_label = label[ids]
                    current_certainty = variance[ids]

                    positive_ids = (label == current_label)
                    negative_ids = ~positive_ids
                    positive_var = variance[positive_ids]
                    negative_var = variance[negative_ids]

                    # argsort yields ascending order
                    selected_positive_ids = torch.argsort(variance[positive_ids])[:num_contrastive_samples]
                    selected_negative_ids = torch.argsort(variance[negative_ids])[:num_contrastive_samples]
                    # if len(selected_positive_ids) != num_contrastive_samples \
                    #         or len(selected_negative_ids) != num_contrastive_samples:
                    #     print(f'{len(positive_ids)}, {len(negative_ids)}, {len(selected_positive_ids)}, {len(selected_negative_ids)}')

                    positive_samples = protos[positive_ids][selected_positive_ids]
                    negative_samples = protos[negative_ids][selected_negative_ids]
                    
                    positive_distance = torch.mean(torch.square(current_proto - positive_samples), dim=1).sum()
                    negative_distance = torch.mean(torch.square(current_proto - negative_samples), dim=1)
                    negative_distance =  torch.square(torch.clamp(margin - torch.sqrt(negative_distance), min=0)).sum()

                    # Up sampling
                    if len(selected_positive_ids) != 0:
                        positive_distance = positive_distance / len(selected_positive_ids)
                    if len(selected_negative_ids) != 0:
                        negative_distance = negative_distance / len(selected_negative_ids)

                    contrastive_loss += (positive_distance + negative_distance) / 2

                contrastive_loss = contrastive_loss/batch_size


                # Future TO-DO: class separate for negative samples

            elif config.contrastive_type == 'NoReliable':

                margin = 1.0
                batch_size = protos.size(0)

                for ids1 in range(batch_size):                    
                    for ids2 in range(ids1+1, batch_size):

                        dist = torch.sqrt(torch.mean(torch.square(protos[ids1] - protos[ids2])))

                        if label[ids1] == label[ids2]:
                            contrastive_loss += torch.square(dist)
                        else:
                            contrastive_loss += torch.square(torch.clamp(margin - dist, min=0))
                    
                contrastive_loss = contrastive_loss/batch_size/batch_size

        loss = class_loss + config.coef_contrastive*contrastive_loss
        loss = loss / len(train_loader)
        loss.backward() 

        metric_meter.update('total_loss', loss.item()*len(train_loader), total=1)
        metric_meter.update('class_loss', class_loss.item(), total=1)
        metric_meter.update('contrastive_loss', contrastive_loss.item(), total=1)
        metric_meter.update('accuracy', 
            correct=(logits.argmax(1) == label).sum().item(), total=len(label))

    optimizer.step() 


def val(val_loader, model, metric_meter, config):

    device = model.device
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
            if config.loss_opt == 'PHuberCrossEntropy':
                class_loss = PHuberCrossEntropy(logits, label, tau=config.tau, reduction='mean')
            else:
                class_loss = F.cross_entropy(logits, label, reduction='mean')
            
            ### Calculate metric loss
            contrastive_loss = torch.tensor(0.0).to(device)
            protos = output['hidden_state']
            if config.coef_contrastive  == 0.0:
                protos = protos.detach()

            loss = class_loss + config.coef_contrastive*contrastive_loss

            metric_meter.update('total_loss', loss.item(), total=1)
            metric_meter.update('class_loss', class_loss.item(), total=1)
            metric_meter.update('contrastive_loss', contrastive_loss.item(), total=1)
            metric_meter.update('accuracy', 
                correct=(logits.argmax(1) == label).sum().item(), total=len(label))


def set_random_seed(seed):
    #print('Random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)


def get_optimizer(model, opt_type='AdamW', lr=1e-5):

    if isinstance(model, nn.DataParallel):
        optimizer = getattr(optim, opt_type)(model.module.parameters(), lr)
    else:
        optimizer = getattr(optim, opt_type)(model.parameters(), lr)
    
    return optimizer


def save_checkpoint(model, path):

    global logger
    logger.print(f'Save model to {path}.')

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path):

    global logger
    logger.print(f'Load model from {path}.')

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    pass