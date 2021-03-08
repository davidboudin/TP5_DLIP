import torch
import torch.nn as nn

class K_temporal(nn.Module):

    def __init__(self,K,hidden_size_1,hidden_size_2, out_size = 1):

        super().__init__()

        self.hidden_size_1 = hidden_size_1

        self.hidden_size_2 = hidden_size_2

        self.input_size = K

        self.fc1 = nn.Linear(K, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, out_size)
        self.relu = nn.ReLU()


    def forward(self,seq):

        out = self.relu(self.fc1(seq))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
