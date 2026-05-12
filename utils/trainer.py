import torch
import torch.nn as nn

class Trainer:
    def __init__(self):
        pass


    def _build_criterion(self):
        pass
    
    def _build_optimizer(self):
        pass

    def _build_scheduler(self):
        pass

    def train_one_epoch(self):
        pass

    @torch.no_grad()
    def validate_one_epoch(self):
        pass


    def train(self):
        pass
    
    @torch.no_grad()
    def evaluate(self):
        pass

