import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqCoreDetector(nn.Module):
    def __init__(self):
        super(SeqCoreDetector, self).__init__()
        self.lin1 = nn.Linear(1024, 64)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.dr1 = nn.Dropout(p=0.5)
        self.dr2 = nn.Dropout(p=0.25)
        self.bc1 = nn.BatchNorm1d(32, eps=1e-03, momentum=0.99)
        self.bc2 = nn.BatchNorm1d(32, eps=1e-03, momentum=0.99)
        self.lin2 = nn.Linear(32, 16)
        self.lin3 = nn.Linear(16, 4)
        self.lin4 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.conv1(x.transpose(2, 1)))
        x = self.dr1(x)
        x = self.bc1(x)
        x = F.relu(self.conv2(x))
        x = self.dr2(x)
        x = self.bc2(x)
        x = F.relu(self.lin2(x.transpose(2, 1)))
        x = F.relu(self.lin3(x))
        x = torch.sigmoid(self.lin4(x))
        return x

