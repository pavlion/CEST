import os
import copy
import time
from utils import ensure_path_exists
from config import Config
from train_teacher import train_teacher, set_random_seed
from train_student import train_student


config = Config()   
config.seed = 2021
set_random_seed(config.seed)
    
### Data config
config.dataset_name = 'ag_news'
config.split = 0
config.max_seq_len = 256
config.label_per_class = 10


### Training config
config.num_epoch = 30
config.encoder_lr = 1e-5
config.fc_lr = 1e-3
config.finetune_lr = 1e-5
config.teacher_train_batch_size = 32
config.train_batch_size = 32
config.eval_batch_size = 300
config.use_data_parallel = False    

### Algo-specific config
config.emb_dim = 128 
config.coef_metric = 0.0
config.coef_contrastive = 0.75
config.contrastive_type = 'Reliable'
config.unlabeled_sample_size = 2000
config.mc_dropout_iters = 30
config.sampling_scheme = 'IG_class' 
config.majority_votes = False 
config.loss_opt = 'PHuberCrossEntropy'
config.transformers_model_name = f'ckpt/pre_training/bert-base-uncased/{config.dataset_name}'    
config.tau = 10



current_time = time.strftime("%m%d_%H-%M") #%Y%S
config.ssl_data_path = f'ssl_data/split{config.split}'     
config.dest_dir = os.path.join('ckpt', config.dataset_name, f'split{config.split}', f'{current_time}')
ensure_path_exists(config.dest_dir)


scores = train_teacher(config)
config.teacher_test_scores = scores
config.dump(os.path.join(config.dest_dir, f'config.json'))

config.teacher_model_path = os.path.join(config.dest_dir, f"best_teacher.pth")
scores = train_student(config)
config.student_test_scores = scores
config.dump(os.path.join(config.dest_dir, f'config.json'))
