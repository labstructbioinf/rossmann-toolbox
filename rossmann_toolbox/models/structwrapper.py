import argparse

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, matthews_corrcoef

from .structmodel import GAT

class GatLit(GAT):
    def __init__(self, hparams):
        if not hasattr(hparams, '__dict__'):
            hparams = argparse.Namespace(**hparams)
        super().__init__(**vars(hparams))
        self.hparams = hparams
