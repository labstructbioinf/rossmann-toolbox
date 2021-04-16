import argparse

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from sklearn.metrics import f1_score, matthews_corrcoef

from .structmodel import GAT

class GatLit(GAT, pl.LightningModule):
    def __init__(self, hparams):
        if not hasattr(hparams, '__dict__'):
            hparams = argparse.Namespace(**hparams)
        super().__init__(**vars(hparams))
        self.hparams = hparams
        

        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = FocalLoss2(size_average=self.hparams.loss_size_avarage)(y_hat, y)
        #loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        progress_bar = {'lr' : self.hparams.learning_rate}
        return {"loss" : loss,  'progress_bar' : progress_bar }
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.reg_term,
                                     amsgrad=self.hparams.use_amsgrad)
        scheduler = { 'scheduler' : ReduceLROnPlateau(optimizer, 'max',
                                                      patience=self.hparams.plateau_patience,
                                                      factor=0.2, verbose =True),

                     'monitor' : 'f1_score',
                     'interval': 'epoch',
                     'frequency' : 1,
                     'name' : 'LRScheduler'
                    }
        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx):
    
        x, y = batch
        y_hat = self(x)
        loss = FocalLoss2(size_average=self.hparams.loss_size_avarage)(y_hat, y)
        #loss = torch.nn.CrossEntropyLoss()(y_hat, y)

        return {'val_loss' : loss, 'y_hat' : y_hat, 'y_true' : y}
    
    def validation_epoch_end(self, outputs):
        val_set_loss = [x['val_loss'] for x in outputs]
        avg_loss = torch.stack(val_set_loss).mean()
        y_hat = [x['y_hat'].view(-1, self.hparams.n_classes) for x in outputs]
        y = [x['y_true'] for x in outputs]
        
        y_hat = torch.cat(y_hat, dim=0)
        y = torch.cat(y).view(-1)
        
        y_hat = y_hat.cpu().detach().numpy()
        assert not (y_hat > 1).any(), y_hat
        y_hat = y_hat.argmax(1)
        y = y.cpu().detach().numpy()
        f1 = f1_score(y, y_hat, average=self.hparams.f1_type)
        f1 = f1.mean()
        f1_loss = f1/(avg_loss + 1e-3)
        
        #matt = matthews_corrcoef(y, y_hat)
        #print(f'{self.hparams.f1_type} val f1 {f1:.3f} f1_loss {f1_loss:.3f} loss {avg_loss:.3f} lr {self.hparams.learning_rate:.5f}')
        #self.log('val_loss',  avg_loss)
        progress_bar = {f'f1_{self.hparams.f1_type}' : f1, 'val_loss' : avg_loss}
        return {'val_loss' : avg_loss, 'f1_score' : f1, 'f1_loss' : f1_loss, 'progress_bar' : progress_bar}
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        logs = self.validation_epoch_end(outputs)
        logs_ = dict(test_f1 = logs['f1_score'],
                     test_loss = logs['val_loss'].item(),
                     test_matt = logs['matt'])
        return logs_
        

        
        

class FocalLoss2(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()