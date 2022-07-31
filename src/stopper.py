
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, model, patience=7, delta=0, print_fn=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0       
        """
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.early_stop = False
        self.min_loss = 1e10
        self.best_acc = 0.0

        self.model = model
        self.print_fn = print_fn

    def __call__(self, val_loss, val_acc, path="", verbose=True):

        save_ckpt = False

        
        if val_loss > self.min_loss:
            
            self.counter += 1
            
            if val_loss < self.min_loss + self.delta:
                save_ckpt = True

            if self.counter >= self.patience:
                self.early_stop = True
                self.print_fn("Early stopped.")
            else:
                self.print_fn(f"Early stop counter {self.counter}/{self.patience}.")

        else: # val_loss <= min_loss + self.delta
            
            save_ckpt = True
            
            self.counter = 0 
            self.min_loss = val_loss
            self.best_acc = val_acc 
             
             
                
            
        return save_ckpt     
    

    def reset(self):

        self.counter = 0
        self.early_stop = False
        self.min_loss = 1e10
        self.best_acc = 0.0
        self.save_checkpoint_counter = 0
