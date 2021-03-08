import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from K_temporal_model import K_temporal
from dataset import Rossler

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

df = pd.read_csv('data.csv')

y_data = df["y"]

y_train = y_data[:1600000]
y_test = y_data[1600000:2000000]
window = 100
# Defining a batch size based on the data
train_loader = DataLoader(Rossler(df=y_train, window=window,pred_window=1, normalise=False),
                                        batch_size=100,
                                        shuffle=True)

test_loader = DataLoader(Rossler(df=y_test, window=window,pred_window=1, normalise=False),
                                        batch_size=100,
                                        shuffle=False)

model = K_temporal(K=window,hidden_size_1=window,hidden_size_2=window,out_size=1)
model = model.float()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1, verbose=False)

epochs = 200
print(model)
for epoch in range(epochs):

    # Running each batch separately
    epoch_loss = 0
    for idx, (seq,target) in enumerate(train_loader):

        # set the optimization gradient to zero
        optimizer.zero_grad()

        # Make predictions on the current sequence
        seq = seq.float()
        target = target.float()
        y_pred = model(seq)
        # Compute the lossseq.view(-1,1,self.input_size)
        #print(y_pred.shape, label.shape)
        deriv_pred = (y_pred - seq[-1])/1e-2

        deriv_pred_2 = (y_pred-2*seq[-1] +seq[-2])/1e-4

        deriv_target = (target-seq[-1])/1e-2

        loss = criterion(y_pred, target)#+torch.norm(deriv_pred_2)*1e-6 #+ 0.0005*criterion(deriv_pred, deriv_target)

        # Perform back propogation and gradient descent

        loss.backward()

        optimizer.step()

        epoch_loss += loss
    scheduler.step()
    if epoch%20 == 0:

        print("Epoch: %s Loss: %f"%(epoch,epoch_loss))

model.eval()
mean = 0 #y_train.mean()
std = 1 #y_train.std()


#Predictor
n_preds = 5000
start = 0
features = y_test[start:window+start].values
target = []
for i in range(n_preds):
    seq = (features[i:]-mean)/std
    seq = torch.tensor(seq).float().view(1,-1)
    y_pred = model(seq)
    y_pred = y_pred.detach().numpy()
    features = np.append(features, y_pred)
    target.append(y_test.iloc[window+i])

#print(target)
#print(features[1:])


plt.plot(target)
plt.plot(features[window:])
plt.savefig("prediction_K_Temp_1000.png")

plt.show()

'''plt.plot(y_test.values[:5000])

np.linspace(0, 2000000 *0.01, 2000000)'''
