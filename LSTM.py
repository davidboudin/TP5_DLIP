import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from LSTM_model import LSTMmodel
from dataset import Rossler

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

df = pd.read_csv('data.csv')

y_data = df["y"]

y_train = y_data[:1600000]
y_test = y_data[1600000:]
window = 50
# Defining a batch size based on the data
train_loader = DataLoader(Rossler(df=y_train, window=window,pred_window=1, normalise=False),
                                        batch_size=100,
                                        shuffle=True)

test_loader = DataLoader(Rossler(df=y_test, window=window,pred_window=1, normalise=False),
                                        batch_size=100,
                                        shuffle=False)

#%%
model = LSTMmodel(input_size=window,hidden_size_1=32,hidden_size_2=16, hidden_size_3=8, out_size=1)
model = model.float()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=0. )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1, verbose=False)

epochs = 80

for epoch in range(epochs):

    # Running each batch separately
    epoch_loss = 0
    for idx, (seq,target) in enumerate(train_loader):

        # set the optimization gradient to zero
        optimizer.zero_grad()

        # initialize the hidden states

        model.hidden_1 = (torch.zeros(1,1,model.hidden_size_1),
                        torch.zeros(1,1,model.hidden_size_1))

        model.hidden_2 = (torch.zeros(1,1,model.hidden_size_2),
                        torch.zeros(1,1,model.hidden_size_2))

        # Make predictions on the current sequence
        seq = seq.float()
        target = target.float()

        y_pred = model(seq.unsqueeze(1))

        # Compute the lossseq.view(-1,1,self.input_size)
        #print(y_pred.shape, label.shape)



        #For L1 regularization,

        # l1_reg = torch.tensor(0.)
        # for param in model.parameters():
        #     l1_reg += torch.norm(param, 1)


        loss = criterion(y_pred, target) #+ 1e-4 * l1_reg
        # Perform back propogation and gradient descent

        loss.backward()

        optimizer.step()

        epoch_loss += loss
    scheduler.step()
    if epoch%5 == 0:

        print("Epoch: %s Loss: %f"%(epoch,epoch_loss))

torch.save(model, "LSTM_new.pth")

#%%
model.eval()
mean = y_train.mean()
std = y_train.std()

def y_to_full_state(features, y_pred, delta_t, params):
    x = features[0]
    y = features[1]
    a, b, c = params
    x_pred = (y_pred-y[-1])/delta_t - a * y_pred
    z_pred = -y_pred - (x_pred-x[-1])/delta_t
    return x_pred, y_pred, z_pred

#Predictor
window = 50
n_preds = 3000
start = 1000
features = y_test[start:window+start].values
target = []
for i in range(n_preds):
    #seq = (features[i:]-mean)/std
    seq = features[i:]
    seq = torch.tensor(seq).float().view(1,-1)
    y_pred = model(seq.unsqueeze(1))
    y_pred = y_pred.detach().numpy()
    features = np.append(features, y_pred)
    target.append(y_test.iloc[start+window+i])

print(target)
print(features[1:])


plt.plot(target)
plt.plot(features[window:])
plt.savefig("prediction_50.png")

plt.show()

#%%
model.eval()
mean = y_train.mean()
std = y_train.std()


#Predictor
window = 1000
n_preds = 1250
start = 0
input = torch.zeros(1000)
input[-1] = 0.2
torch.cat((input,torch.ones((1))))

pred
for i in range(window):
    pred = model(input)
    input = torch.cat((input,pred))
    break


features = y_test[start:window+start].values
target = []
for i in range(n_preds):
    seq = (features[i:]-mean)/std
    seq = torch.tensor(seq).float().view(1,-1)
    y_pred = model(seq.unsqueeze(1))
    y_pred = y_pred.detach().numpy()
    features = np.append(features, y_pred)
    target.append(y_test.iloc[window+i])

print(target)
print(features[1:])


plt.plot(target)plt.plot(y_test.values[:5000])

np.linspace(0, 2000000 *0.01, 2000000)

plt.plot(features[window:])
plt.savefig("prediction_1000.png")

plt.show()

#%%
