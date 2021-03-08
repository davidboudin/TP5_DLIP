import torch
import torch.nn as nn

class LSTMmodel(nn.Module):

    def __init__(self,input_size,hidden_size_1,hidden_size_2,out_size):

        super().__init__()

        self.hidden_size_1 = hidden_size_1

        self.hidden_size_2 = hidden_size_2

        self.input_size = input_size

        self.lstm_1 = nn.LSTM(input_size,hidden_size_1,num_layers=1).float()

        self.lstm_2 = nn.LSTM(hidden_size_1,hidden_size_1,num_layers=1).float()

        self.lstm_3 = nn.LSTM(hidden_size_1,hidden_size_2,num_layers=1).float()

        self.linear = nn.Linear(hidden_size_2,out_size)

        self.hidden_1 = (torch.zeros(1,1,hidden_size_1), torch.zeros(1,1,hidden_size_1))

        self.hidden_2 = (torch.zeros(1,1,hidden_size_2), torch.zeros(1,1,hidden_size_2))

        self.hidden_3 = (torch.zeros(1,1,hidden_size_2), torch.zeros(1,1,hidden_size_2))

    def forward(self,seq):
        self.lstm_1 = self.lstm_1.float()

        lstm_out_1 , self.hidden_1 = self.lstm_1(seq)

        lstm_out_2 , self.hidden_2 = self.lstm_2(lstm_out_1)

        lstm_out_3 , self.hidden_3 = self.lstm_3(lstm_out_2)

        pred = self.linear(lstm_out_3.view(len(seq),-1))

        return pred
