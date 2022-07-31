import os
import time 

try:
    # import wandb
    raise ModuleNotFoundError # Do not use wandb

except ModuleNotFoundError:
    class wandb_fake:
        def init(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass
        
        def Image(self, *args, **kwargs):
            pass    
    #print("Fake wandb is used.")
    wandb = wandb_fake()


class Timer:

    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def epoch_time(self):
        curr_time = time.time()
        duration = int(curr_time - self.last_time)
        self.last_time = time.time()

        if duration >= 3600:
            return '{:.1f}h'.format(duration / 3600)
        if duration >= 60:
            return '{}m'.format(round(duration / 60))
        return '{:.2f}s'.format(duration)

    def measure(self, p=1):
        x = (time.time() - self.start_time) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))

        return '{:.2f}s'.format(x)



def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path



def calc_acc(pred, label, return_raw=False):
    '''
    Calculate the accuracy
    Args:
        pred  (tensor): shape=(batch_size, num_class)
        label (tensor): shape=(batch_size)
    '''
    pred_idx = pred.argmax(dim=1) #
    correct = (pred_idx == label).int().sum().item()
    total = len(label)

    if return_raw: 
        return correct/total, correct, total
        
    return correct/total