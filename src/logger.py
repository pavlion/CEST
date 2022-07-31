import json
from collections import defaultdict

class TrainingLogger:

    def __init__(self, dest_path='.'):
        self.epoch = 0
        self.logs = []
        self.dest_path = dest_path 
    
    def reset(self, dest_path='.'):
        self.logs = [] 
        self.epoch = 0
        self.dest_path = dest_path 

    def print(self, epoch_msg, end="\n", show=True, epoch_end=False):
        if show:
            print(epoch_msg + " "*20, end=end)
        self.logs.append(epoch_msg)
        self.epoch += 1 if epoch_end else 0
        self.dump_logs()
    
    def dump_logs(self):
        msg = '\n'.join(self.logs)
        with open(self.dest_path, 'w') as f:
            f.write(msg)

class MetricsLogger:

    def __init__(self, dest_path='.'):
        self.epoch = 0
        self.history = []
        self.dest_path = dest_path 
    
    def log(self, epoch_log, dest_path='.'):

        self.history.append(epoch_log)

        with open(self.dest_path, 'w') as f:
            json.dump(self.history, f, indent='    ')

class MetricMeters:
    
    def __init__(self):

        self.n = defaultdict(int)
        self.n_corrects = defaultdict(int)
        self.history = {}
        
        self.current_mode = 'train'


    def reset(self, mode='train', clear=False):        

        self.n = defaultdict(int)
        self.n_corrects = defaultdict(int)
        self.current_mode = mode

        if mode not in self.history.keys() or clear:
            self.history[mode] = defaultdict(list)
            

    def update(self, monitor, correct, total):    

        self.n_corrects[monitor] += correct    
        self.n[monitor] += total


    def epoch_get_scores(self):

        scores = {}
        for key in self.n.keys():
            n_correct = self.n_corrects[key]
            n = self.n[key]
            score = (n_correct/n) if n != 0 else 0.0
            
            scores[key] = score
            
            self.history[self.current_mode][key].append(score)

        return scores

    def dump(self, path):

        with open(path, 'w') as f:
            json.dump(self.history, f, indent='    ')
        
        

class MetricMeter:
    def __init__(self, name='', at=1):
        self.name = name
        self.at = at
        self.n = 0.0
        self.n_corrects = 0.0
        self.name = '{}@{}'.format(name, at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, correct, total):    
        self.n_corrects += correct    
        self.n += total

    def get_score(self):
        return self.n_corrects / self.n if self.n != 0 else 0.0
