import torch
import torch.nn as nn

class K_temporal(nn.Module):

    def __init__(self,K,hidden_size, out_size = 1, nb_layers=3):

        super().__init__()

        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.input_size = K
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(K, hidden_size))
        for i in range(nb_layers-2):
            self.layers.append(nn.Linear(hidden_size, hidden_size) )
        self.layers.append(nn.Linear(hidden_size, out_size))
        '''self.fc1 = nn.Linear(K, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)'''
        self.relu = nn.ReLU()


    def forward(self,seq):
        for i in range(self.nb_layers-1):
            seq = self.relu(self.layers[i](seq))
        out = self.layers[-1](seq)
        '''out = self.relu(self.fc1(seq))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)'''
        return out
