import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.sampler import Sampler

class AugmentedDataSelector:    
    ''' Select proper samples from the unlabeled data '''

    def __init__(self, unlabeled_dataset, num_evaluate_pool_data, num_augmented_data, eval_batch_size, 
        sampling_scheme='threshold', threshold=0.0, mc_dropout_iters=30, majority_votes=False):
        
        self.unlabeled_dataset = unlabeled_dataset
        self.num_evaluate_pool_data = num_evaluate_pool_data
        self.num_augmented_data = num_augmented_data
        self.eval_batch_size = eval_batch_size

        # Sampling config
        self.sampling_scheme = sampling_scheme
        self.threshold = threshold        
        self.mc_dropout_iters = mc_dropout_iters
        self.majority_votes = majority_votes

        self.selected_avg_confidence = []

        if self.sampling_scheme == 'threshold' or self.sampling_scheme == 'uniform':
            self.mc_dropout_iters = 1

        self.finished_ids = np.array([])


    def select_samples(self, model):

        if self.mc_dropout_iters == 0:         
            return None
        
        eval_pool_size = min(len(self.unlabeled_dataset), self.num_evaluate_pool_data)   
        eval_pool_ids = np.random.choice(len(self.unlabeled_dataset), eval_pool_size, replace=False)       
        

        ### Evaluate uncertainty for the selected evaluation pool
        unlabeled_data_loader = torch.utils.data.DataLoader(self.unlabeled_dataset, 
            batch_size=self.eval_batch_size, sampler=IndicesSampler(eval_pool_ids))              
        y_T, y_pred = self._mc_dropout_evaluate(model, unlabeled_data_loader)


        ### Sampling according to the sampling_scheme
        selected_pool_ids, conf_scores = None, None
        if 'uniform' in self.sampling_scheme:
            selected_pool_ids, conf_scores = self._uniform_sampling(y_T)

        elif 'threshold' in self.sampling_scheme:
            
            # Select measures
            # IG: information gain
            if 'confidence' in self.sampling_scheme:
                target = 'confidence'
            elif 'PV' in self.sampling_scheme:
                target = 'PV'
            elif 'IG' in self.sampling_scheme:
                target = 'IG'
            elif 'NE' in self.sampling_scheme:
                target = 'NE'

            selected_pool_ids, conf_scores = self._BNN_threshold_sampling(y_T, y_pred, eval_pool_ids, target)

        elif 'IG' in self.sampling_scheme:

            selected_pool_ids, conf_scores = self._IG_sampling(y_T, y_pred, eval_pool_ids)  
        
        
        selected_ids = eval_pool_ids[selected_pool_ids]
        pseudo_labels = y_pred[selected_pool_ids]
        conf_scores = conf_scores[selected_pool_ids]
        # Update the selected indices pool
        # self.finished_ids = np.union1d(self.finished_ids, selected_ids[:len(selected_ids)//5])


        # if len(selected_pool_ids) != 0:
        #     print("Selected uncertainty (max/min/avg): " + 
        #         f"{conf_scores.max():.4f}/{conf_scores.min():.4f}/{conf_scores.mean():.4f}")
        #     self.selected_avg_confidence = conf_scores.mean()
        
        y_var = np.var(y_T, axis=0)  # (len_unlabeled_data, num_classes) 

        return selected_ids, pseudo_labels, conf_scores, y_var


    def _uniform_sampling(self, y_T):
        
        y_mean = np.mean(y_T, axis=0)
        selected_size = min(len(y_mean), self.num_augmented_data)
        selected_pool_ids = np.random.choice(len(y_mean), self.num_augmented_data, replace=False)
        conf_scores = y_mean.max(1)

        return selected_pool_ids, conf_scores


    def _conf_threshold_sampling(self, y_T):
        '''
        eval_pool_ids, y_mean: pool indices for mc dropout evaluation
        selected_pool_ids: ids selected from eval_pool_ids
        selected_ids: global ids (subsetting from selected_pool_ids)
        shape of y_mean: (len_unlabeled_data, num_classes)
        '''
        y_mean = np.mean(y_T, axis=0)
        confident_ids = np.argwhere(y_mean.max(1) > self.threshold).squeeze(1)

        ### Randomly select confident samples
        selected_size = min(len(confident_ids), self.num_augmented_data)
        selected_pool_ids = confident_ids[np.random.choice(len(confident_ids), selected_size, replace=False)]
        

        ### select top-k confident samples        
        # if len(confident_ids) > self.num_augmented_data: # enough samples        
        #     kth = len(confident_ids) - sampled_size
        #     topk_ids = np.argpartition(y_mean.max(1)[confident_ids], kth=kth)[kth:]
        #     selected_pool_ids = confident_ids[topk_ids]
        # else:
        #     selected_pool_ids = confident_ids
        

        conf_scores = y_mean.max(1)

        return selected_pool_ids, conf_scores


    def _IG_sampling(self, y_T, y_pred, eval_pool_ids):
        ''' y_pred: pseudo-labels (np.array of shape (len_unlabeled_data,)) '''

        IG_acq = get_IG_acquisition(y_T)

        def real_to_pool(real_ids):
            converter = {real_ids: pool_ids 
                for pool_ids, real_ids in enumerate(eval_pool_ids)}
            pool_ids = np.array([int(converter[ids]) for ids in real_ids])
            return pool_ids

        selected_pool_ids = []
        
        if 'class' in self.sampling_scheme:

            selected_pool_ids = []
            classes = np.unique(y_pred)   
            sample_size = self.num_augmented_data // len(classes)

            for class_id in classes:

                class_ids = np.argwhere(y_pred == class_id).squeeze(1)
                if len(class_ids) == 0: 
                    print(f"No instances are selected for class {class_id}")
                    selected_pool_ids = []
                    break

                real_ids = eval_pool_ids[class_ids]
                unused_real_ids = np.setdiff1d(real_ids, self.finished_ids, assume_unique=True) 
                unused_pool_ids = real_to_pool(unused_real_ids)
                
                IG_acq_class = IG_acq[unused_pool_ids]
                prob_norm = np.maximum(np.zeros(len(IG_acq_class)), (1. - IG_acq_class)/np.sum(1. - IG_acq_class))
                # prob_norm = np.maximum(np.zeros(len(IG_acq_class)), prob_norm) 
                prob_norm = prob_norm / np.sum(prob_norm)

                selected_class_ids = unused_pool_ids[
                    np.random.choice(len(unused_pool_ids), min(len(unused_pool_ids), sample_size), p=prob_norm)]
                selected_pool_ids.append(selected_class_ids)
                
            selected_pool_ids = np.concatenate(selected_pool_ids)
                    

        else:
            unused_real_ids = np.setdiff1d(eval_pool_ids, self.finished_ids, assume_unique=True) 
            unused_pool_ids = real_to_pool(unused_real_ids)


            unused_IG_acq = IG_acq[unused_pool_ids]
            prob_norm = np.maximum(np.zeros(len(unused_IG_acq)), (1. - unused_IG_acq)/np.sum(1. - unused_IG_acq)) 
            prob_norm = prob_norm / np.sum(prob_norm)
            selected_pool_ids = unused_pool_ids[np.random.choice(len(unused_pool_ids), 
                min(len(unused_pool_ids), self.num_augmented_data), p=prob_norm)]
            

        if 'replacement' not in self.sampling_scheme:
            self.finished_ids = np.union1d(self.finished_ids, eval_pool_ids[selected_pool_ids])

        conf_scores = IG_acq # np.mean(y_T, axis=0).max(1)
        # selected_ids = eval_pool_ids[selected_pool_ids]
        # conf_scores = IG_acq[selected_pool_ids]
        #print("conf_scores:", conf_scores)
        
        return selected_pool_ids, conf_scores


    def _BNN_threshold_sampling(self, y_T, y_pred, eval_pool_ids, target='IG'):
        ''' y_pred: pseudo-labels (np.array of shape (len_unlabeled_data,)) '''
        
        # Lower means model is well-learned on it
        num_classes = y_T.shape[-1]
        target_measure = get_acquisitions(y_T)[target]

        def real_to_pool(real_ids):
            converter = {real_ids: pool_ids 
                for pool_ids, real_ids in enumerate(eval_pool_ids)}
            pool_ids = np.array([int(converter[ids]) for ids in real_ids])
            return pool_ids

        selected_pool_ids = None
        
        # class-separated
        if 'class' in self.sampling_scheme: 

            selected_pool_ids = []  
            sample_size = self.num_augmented_data // num_classes
            
            for class_id in range(num_classes):

                class_ids = np.argwhere(y_pred == class_id).squeeze(1)
                target_measure_class = np.argsort(target_measure[class_ids]) # ascending order
                # target_measure_class = (target_measure < 0.05)
                
                # Without replacement: ids in self.finished_ids will not be sampled
                real_ids = eval_pool_ids[target_measure_class]
                selected_class_ids = np.setdiff1d(real_ids, self.finished_ids, assume_unique=True) 
                selected_class_ids = selected_class_ids[:sample_size]
                # selected_class_ids = selected_class_ids[np.random.choice(len(selected_class_ids), 
                #     sample_size, p=to_prob(target_measure_class[:len(selected_class_ids)]))]  # hard sampling strategy
                    
                # self.finished_ids = np.union1d(self.finished_ids, selected_class_ids) 
                selected_pool_ids.append(real_to_pool(selected_class_ids))
                
            selected_pool_ids = np.concatenate(selected_pool_ids).astype(int)

        else:

            target_measure_ids = np.argsort(target_measure) # ascending order
            # target_measure_ids = (target_measure < 0.02) 
            sample_size = self.num_augmented_data

            # Without replacement: in target_measure_ids, but not in self.finished_ids
            real_ids = eval_pool_ids[target_measure_ids]
            selected_ids = np.setdiff1d(real_ids, self.finished_ids)#[:sample_size]
            # selected_ids = selected_ids[np.random.choice(len(selected_ids), 
            #         sample_size, p=to_prob(target_measure_class[:len(selected_class_ids)]))]

            # Update the selected indices pool
            # self.finished_ids = np.union1d(self.finished_ids, selected_ids[:sample_size])
            selected_pool_ids.append(real_to_pool(selected_ids))

        if 'replacement' not in self.sampling_scheme:
            self.finished_ids = np.union1d(self.finished_ids, eval_pool_ids[selected_pool_ids])


        conf_scores = target_measure # np.mean(y_T, axis=0).max(1)
        # selected_ids = eval_pool_ids[selected_pool_ids]
        # conf_scores = target_measure[selected_pool_ids]
        #print("conf_scores:", conf_scores)
        
        return selected_pool_ids, conf_scores


    def select_useful_samples(self, model, unlabeled_data_loader):
        
        if self.mc_dropout_iters == 0:         
            return None
        
        eval_pool_size = min(len(self.unlabeled_dataset), self.num_evaluate_pool_data)   
        eval_pool_ids = np.random.choice(len(self.unlabeled_dataset), eval_pool_size, replace=False)               

        ### Evaluate uncertainty for the selected evaluation pool
        unlabeled_data_loader = torch.utils.data.DataLoader(self.unlabeled_dataset, 
            batch_size=self.eval_batch_size, sampler=IndicesSampler(eval_pool_ids))              
        y_T, y_pred = self._mc_dropout_evaluate(model, unlabeled_data_loader)


        device = next(model.parameters()).device
        model.eval()
        y_pred = []
        hidden_states = []
        trange = tqdm(unlabeled_data_loader, desc="Evaluating by Usefulness", ncols=100)
        with torch.no_grad():       

            for batch in trange:     

                input_ids = batch["input_ids"].to(device) 
                token_type_ids = batch["token_type_ids"].to(device) 
                attention_mask = batch["attention_mask"].to(device)

                # pred = model(input_ids, token_type_ids, attention_mask) # pred: (batch_size, num_classes)
                output = model(input_ids, token_type_ids, attention_mask) 
                pred = output['logits']
                protos = output['hidden_state']
                y_pred.append(pred.cpu().argmax(1))
                hidden_states.append(protos.cpu())

        y_pred = np.concatenate(y_pred, axis=0)
        hidden_states = np.concatenate(hidden_states, axis=0)
        print(hidden_states.shape)
                
        total_len = [i for i in range(len(y_pred))]
        selected_ids = []
        selected_edges = []
        for i in range(len(y_pred)):
            for j in range(i, len(y_pred)):
                distance = np.sum((hidden_states[i] - hidden_states[j])**2)
                
                if y_pred[i] == y_pred[j] and distance > 0.0:
                    selected_ids.append(i)
                    selected_ids.append(j)
                    selected_edges.append([i, j])
                    
                if y_pred[i] != y_pred[j] and distance < 1.0:
                    selected_ids.append(i)
                    selected_ids.append(j)
                    selected_edges.append([i, j])
        
        selected_ids = eval_pool_ids[np.unique(np.array(selected_ids))]
        selected_edges = np.array(selected_edges)
        y_pred = y_pred[selected_ids]
        variance = y_var = np.var(y_T, axis=0)[selected_ids]  
        
        return selected_ids, selected_edges, pseudo_labels, variance


    def _mc_dropout_evaluate(self, model, unlabeled_data_loader):
        
        print("Evaluate by MC Dropout...", end="\r")
        device = next(model.parameters()).device

        if self.sampling_scheme == 'threshold':
            model.eval()
        else:
            model.train()
        
        y_T = []
        trange = tqdm(range(self.mc_dropout_iters), desc="Evaluating by MC Dropout", ncols=100)
        for i in trange:

            y_pred = torch.tensor([])

            with torch.no_grad():       

                for i, batch in enumerate(unlabeled_data_loader, start=1):     

                    input_ids = batch["input_ids"].to(device) 
                    token_type_ids = batch["token_type_ids"].to(device) 
                    attention_mask = batch["attention_mask"].to(device)

                    # pred = model(input_ids, token_type_ids, attention_mask) # pred: (batch_size, num_classes)
                    output = model(input_ids, token_type_ids, attention_mask) 
                    pred = output['logits']
                    y_pred = torch.cat((y_pred, pred.cpu()), dim=0)

                    trange.set_postfix_str(f"{i/len(unlabeled_data_loader)*100.0:.2f}% Data")

            # y_pred: (len_unlabeled_data, num_classes)
            y_T.append(torch.nn.functional.softmax(y_pred, dim=1))


        y_T = torch.stack(y_T, dim=0).numpy() # (T, len_unlabeled_data, num_classes)


        #compute majority prediction: y_pred.shape=(len_unlabeled_data,)
        if self.majority_votes:
            y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])

        else: # Use hard labels
            
            y_pred = []
            model.eval()
            with torch.no_grad():   

                for i, batch in enumerate(unlabeled_data_loader, start=1):
                    
                    input_ids = batch["input_ids"].to(device) 
                    token_type_ids = batch["token_type_ids"].to(device) 
                    attention_mask = batch["attention_mask"].to(device)
                    output = model(input_ids, token_type_ids, attention_mask) 
                    logits = output['logits']
                    y_pred += logits.cpu().argmax(1).tolist()
            
            y_pred = np.array(y_pred)


        # #compute mean: y_mean=(len_unlabeled_data, num_classes)
        # y_mean = np.mean(y_T, axis=0)
        
        # #compute variance: y_var.shape=(len_unlabeled_data, num_classes)
        # y_var = np.var(y_T, axis=0)

        return y_T, y_pred



def to_prob(vec):
    vec = np.array(vec)
    prob = vec / np.sum(vec)
    prob = np.maximum( np.zeros(len(vec)), prob)
    prob = prob / np.sum(prob)
    
    return prob


def get_IG_acquisition(y_T, eps=1e-16):
    ''' y_T: numpy array of size: (T, len_dset, num_class) '''

    expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + eps), axis=-1), axis=0) 
    expected_p = np.mean(y_T, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + eps), axis=-1)
    return (entropy_expected_p - expected_entropy)


def get_acquisitions(y_T, eps=1e-16):
    ''' 
    y_T: numpy array of size: (T, len_dset, num_class) 
    '''
    
    u_c = np.mean(y_T, axis=0) # (len_dset, num_class)
    H_t = - np.sum(y_T * np.log10(y_T + eps), axis=-1) # (T, len_dset)

    # per_class_variance = np.mean((y_T - u_c)**2, axis=0) 
    PV = np.sum(np.var(y_T, axis=0), axis=-1)
    # EV = np.var(H_t, axis=0)
    # PE = np.sum(u_c * np.log10(u_c + eps), axis=-1)
    # MI = PE - np.mean(H_t, axis=0)

    # IG
    expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + eps), axis=-1), axis=0) 
    expected_p = np.mean(y_T, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + eps), axis=-1)
    IG = (entropy_expected_p - expected_entropy)

    # Normalized entropy
    NE = np.mean(H_t, axis=0)/(np.log10(4))

    # Confidence
    CONF = u_c.max(1) # (len_dset, )

    return {
        "PV": PV,
        "NE": NE,
        "IG": IG,
        "CONF": CONF
    }


class IndicesSampler(Sampler):
    ''' Data loader will only sample specific indices '''
 
    def __init__(self, indices, shuffle=False):

        self.indices = np.array(indices)
        self.len_ = len(self.indices)
        self.shuffle = shuffle
        
    
    def __len__(self):
        return self.len_

    def __iter__(self):

        if self.shuffle:
            np.random.shuffle(self.indices) # in-place shuffle

        for index in self.indices:
            yield index
    

if __name__ == '__main__':
    pass