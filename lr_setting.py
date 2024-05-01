import numpy as np

class LR_choice:
    def __init__(self, initial_lr, lr_choices='exponential', **kwargs):
        self.lr=initial_lr
        self.lr_choices = lr_choices
        self.kwargs = kwargs

    
    def get_lr(self, epoch, epoch_num=None):
        if self.lr_choices == 'exponential':
            return self.exponential_decay_lr(epoch)
        elif self.lr_choices == 'step':
            return self.step_decay_lr(epoch)
        elif self.lr_choices == 'cosine':
            return self.cosine_decay_lr(epoch, epoch_num)
    
    def exponential_decay_lr(self, epoch):
        decay_rate=self.kwargs.get('decay_rate',0.5)
        learning_rate=self.lr*(decay_rate ** epoch)
        return learning_rate
    
    def step_decay_lr(self, epoch):
        drop = self.kwargs.get('drop', 0.8)
        epochs_drop = self.kwargs.get('epochs_drop', 10)
        learning_rate=self.lr * (drop ** (epoch // epochs_drop))
        return learning_rate
    
    
    def cosine_decay_lr(self, epoch,epoch_num):
        learning_rate=self.lr*(1 + np.cos(np.pi * epoch / epoch_num)) / 2
        return learning_rate