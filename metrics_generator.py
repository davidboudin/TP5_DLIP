import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from LSTM_model import LSTMmodel
from dataset import Rossler
from rossler_map import RosslerMap
from scipy import stats
from scipy.fft import fft, fftfreq, fftshift
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

#Importing training data
df = pd.read_csv('data.csv')

#Generating uniformaly sampled initial values
x_random = np.random.uniform(low=df["x"].min(), high=df["x"].max(), size=5)
y_random = np.random.uniform(low=df["y"].min(), high=df["y"].max(), size=5)
z_random = np.random.uniform(low=df["z"].min(), high=df["z"].max(), size=5)

random_init = np.array((x_random, y_random, z_random))

#Preparing temporal series generator
params = (0.2, 0.2, 0.54)
delta_t = 1e-2
ROSSLER_MAP = RosslerMap(delta_t=delta_t)
n_preds = 3000
window = 50

#Loading models
model_lstm = torch.load("LSTM_new.pth")
model_lstm = model_lstm.float()
model_lstm.eval()

model_temp = torch.load("K_temporal.pth")
model_temp.eval()

LSTM_results = []
TEMP_results = []

def y_to_full_state(features, y_pred, delta_t, params):
    """
    Estimates x and z from y using the Rossler system formula
    """
    x = features[:,0]
    y = features[:,1]
    a, b, c = params
    x_pred = (y_pred-y[-1])/delta_t - a * y_pred
    z_pred = -y_pred - (x_pred-x[-1])/delta_t
    return x_pred, y_pred, z_pred

#%% Time series prediction
#For each sampled initial values
for i in range(len(x_random)):
    #Generate a ground_truth trajectory
    features,t = ROSSLER_MAP.full_traj(3050, random_init[:,i])
    ground_truth = features.copy()
    temp_features = features.copy()[:50]
    features = features[:50]

    #Based on a history of 50 y-values
    for j in range(n_preds):

        #Predict the next values using both models
        seq = features[j:j+50, 1]
        seq = torch.tensor(seq).float().view(1,-1)
        y_pred = model_lstm(seq.unsqueeze(1))
        y_pred = y_pred.detach().numpy()
        x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
        features = np.append(features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

        seq = temp_features[j:j+50, 1]
        seq = torch.tensor(seq).float().view(1,-1)
        y_pred = model_temp(seq.unsqueeze(1))
        y_pred = y_pred.detach().numpy()
        x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
        temp_features = np.append(temp_features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

    #RMSE per prediction for both sets of TS
    LSTM_RMSE = np.sum((features[50:]-ground_truth[50:])**2, axis=1)**0.5
    TEMP_RMSE = np.sum((temp_features[50:]-ground_truth[50:])**2, axis=1)**0.5
    LSTM_results.append(LSTM_RMSE)
    TEMP_results.append(TEMP_RMSE)

#Plot of the cumulative RMSE
plt.plot(np.cumsum(np.mean(LSTM_results,axis=0)), label="LSTM")
plt.plot(np.cumsum(np.mean(TEMP_results,axis=0)), label="Temporal")
plt.legend()
plt.title("Cumulative RMSE of predictions based on a history of 50 states")
plt.xlabel("Time")
plt.ylabel("Cumulative RMSE")
plt.savefig("RMSE.png")
plt.show()


#%% Making predictions
#Generating a ground-truth
features,t = ROSSLER_MAP.full_traj(3050, random_init[:,2])
ground_truth = features.copy()
temp_features = features.copy()[:50]
features = features[:50]


for j in range(n_preds):
    #Make predictions with both models

    seq = features[j:j+50, 1]
    seq = torch.tensor(seq).float().view(1,-1)
    y_pred = model_lstm(seq.unsqueeze(1))
    y_pred = y_pred.detach().numpy()
    x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
    features = np.append(features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

    seq = temp_features[j:j+50, 1]
    seq = torch.tensor(seq).float().view(1,-1)
    y_pred = model_temp(seq.unsqueeze(1))
    y_pred = y_pred.detach().numpy()
    x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
    temp_features = np.append(temp_features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

#PLotting the y_value
for i in [1]:
    plt.plot(ground_truth[:,i], label="GT")
    plt.plot(temp_features[:,i], label ="Temporal")
    plt.plot(features[:,i], label="LSTM")
    plt.legend()
    plt.title("Predictions of the y-value based on a history of 50 states")
    plt.xlabel("Time")
    plt.ylabel("Predictions")
    plt.savefig("predictions.png")
    plt.show()

#%% Checking that the approximation of x and z from y is accurate
y = ground_truth[:,1]
features = features[:50]

for j in range(n_preds):
    #Make predictions with both models
    x_pred, y_pred, z_pred = y_to_full_state(features, y[50+j], delta_t, params)
    features = np.append(features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)


plt.plot(features[:,2])
plt.plot(ground_truth[:,2])
plt.show()

 #%%
i=1
x_eval = np.linspace(ground_truth[:,i].min()-1, ground_truth[:,i].max()+1, num=200)

kde_gt_y = stats.gaussian_kde(ground_truth[:,i],bw_method=0.01)
kde_temp_y = stats.gaussian_kde(temp_features[:,i],bw_method=0.01)
kde_lstm_y = stats.gaussian_kde(features[:,i],bw_method=0.01)

plt.title("Gaussian KDE of predictions")
plt.plot(x_eval, kde_gt_y(x_eval), label="Ground Truth")
plt.plot(x_eval, kde_temp_y(x_eval), label="Temp")
#plt.plot(x_eval, kde_lstm_y(x_eval), label="LSTM")
plt.ylabel("Density")
plt.xlabel("y-value")
plt.legend()
plt.savefig("PDF.png")
plt.show()

#%% Calculating the Fourier transform of the predictions
# number of signal points
N = ground_truth[:,1].size

# sample spacing
T = 1.0 / ground_truth[:,1].size

x = np.linspace(0.0, N*T, N, endpoint=False)

y = ground_truth[:,1]
z = temp_features[:,1]

zf = fft(z)
zplot =fftshift(zf)
yf = fft(y)
xf = fftfreq(N, T)
xf = fftshift(xf)
yplot = fftshift(yf)

plt.title("Fourier transform of the predictions")
plt.plot(xf, 1.0/N * np.abs(yplot),label="Ground Truth")
plt.plot(xf, 1.0/N * np.abs(zplot),label="Temp")
plt.xlim(-50,50)
plt.xlabel("z")
plt.ylabel("Magnitude")
plt.legend()
plt.savefig("Fourier.png")
plt.show()
