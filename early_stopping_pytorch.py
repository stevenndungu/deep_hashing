# adopted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience # number of times to allow for no improvement before stopping the execution
        self.verbose = verbose
        self.counter = 0 # count the number of times the validation accuracy not improving
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta # the minimum change to be counted as improvement
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        #modified by SN
        #when score is not improving and is constant throughout the epochs, the EarlyStopping counter should start, and it should terminate eventually after we run out of patience ==> changed from score < self.best_score + self.delta to score <= self.best_score + self.delta
        if score <= self.best_score + self.delta: 
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0 # reset the counter if validation loss decreased at least by min_delta
         else: #(if self.best_score is None)
            self.best_score = score
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss