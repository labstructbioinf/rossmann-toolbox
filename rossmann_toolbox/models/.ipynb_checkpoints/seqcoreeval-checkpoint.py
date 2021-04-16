import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqCoreEvaluator(nn.Module):
    def __init__(self):
        super(SeqCoreEvaluator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=32, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)
        self.dr1 = nn.Dropout(p=0.5)
        self.bc1 = nn.BatchNorm1d(32, eps=1e-03, momentum=0.99)
        self.lin1 = nn.Linear(32, 32)
        self.lin2 = nn.Linear(61, 64)
        self.dr2 = nn.Dropout(p=0.25)
        self.lin2 = nn.Linear(32, 16)
        self.lin3 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.dr1(F.relu(self.conv1(x)))
        x = self.bc1(x)
        x = self.dr2(F.relu(self.conv2(x)))
        x = F.relu(self.lin1(x.transpose(2, 1)))
        x = torch.max(x, axis=1).values
        x = torch.sigmoid(self.lin3(F.relu(self.lin2(x))))
        return x

