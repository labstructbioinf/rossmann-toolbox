import argparse

from .structmodel import GAT

class GatLit(GAT):
    def __init__(self, hparams):
        if not hasattr(hparams, '__dict__'):
            hparams = argparse.Namespace(**hparams)
        super().__init__(**vars(hparams))
        self.hparams = hparams
