"""
early_stopping.py
Early stopping mechanism to prevent overfitting and save training time.
Monitors a specific metric (QWK) and stops training when improvement stalls.
"""

import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        """
        Args:
            patience (int): How many epochs to wait after last time improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): 'max' for metrics like Kappa/Accuracy, 'min' for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = model.state_dict()
        else:
            if self.mode == 'max':
                improved = current_score > (self.best_score + self.min_delta)
            else:
                improved = current_score < (self.best_score - self.min_delta)

            if improved:
                self.best_score = current_score
                self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def get_best_state(self):
        return self.best_model_state
