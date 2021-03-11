import torch
import torch.nn as nn

class Conv_temporal(nn.Module):

    def __init__(self,K,hidden_size, out_size = 1, nb_layers=3):

        super().__init__()

        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.input_size = K
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc1 = nn.Linear(3*49, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()


    def forward(self,seq):
        batch = seq.shape[0]
        seq = self.conv1(seq)
        seq = seq.view(batch,3*49)
        out = self.relu(self.fc1(seq))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
