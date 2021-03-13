import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft, fftfreq, fftshift

from dataset import Rossler
from rossler_map import RosslerMap

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from tempconv_model import Conv_temporal, Linear_temporal

graph_titles = False

#Importing training data
df = pd.read_csv('data.csv')

#Generating uniformaly sampled initial values
x_random = np.random.uniform(low=df["x"].min(), high=df["x"].max(), size=10)
y_random = np.random.uniform(low=df["y"].min(), high=df["y"].max(), size=10)
z_random = np.random.uniform(low=df["z"].min(), high=df["z"].max(), size=10)

random_init = np.array((x_random, y_random, z_random))

#Preparing temporal series generator
params = (0.2, 0.2, 5.7)
delta_t = 1e-2
ROSSLER_MAP = RosslerMap(delta_t=delta_t)
n_preds = 3000
window = 100

#Loading models
model_conv = torch.load("K_temporal_conv.pth",  map_location=torch.device('cpu'))
model_linear = torch.load("K_temporal_linear.pth")
model_conv.eval()
model_linear.eval()

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

#%%Time series prediction
TEMP_results = []
TEMP_linear_results = []
#For each sampled initial values
for i in range(len(x_random)):
    #Generate a ground_truth trajectory
    features,t = ROSSLER_MAP.full_traj(3100, random_init[:,i])
    ground_truth = features.copy()
    features = features[:window]
    features_linear = features.copy()

    #Based on a history of 50 y-values
    for j in range(n_preds):

        #Predict the next values using the trained model
        seq = features[j:j+window, 1]
        seq = torch.tensor(seq).float().view(1,-1)
        y_pred = model_conv(seq)
        y_pred = y_pred.detach().numpy()
        x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
        features = np.append(features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

        seq = features_linear[j:j+window, 1]
        seq = torch.tensor(seq).float().view(1,-1)
        y_pred = model_linear(seq.unsqueeze(1))
        y_pred = y_pred.detach().numpy()
        x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
        features_linear = np.append(features_linear, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

    #RMSE per prediction for both sets of TS
    TEMP_RMSE = np.sum((features[window:]-ground_truth[window:])**2, axis=1)**0.5
    TEMP_results.append(TEMP_RMSE)

    TEMP_RMSE = np.sum((features_linear[window:]-ground_truth[window:])**2, axis=1)**0.5
    TEMP_linear_results.append(TEMP_RMSE)


#Plot of the cumulative RMSE
plt.figure(facecolor='white')
plt.plot(np.cumsum(np.mean(TEMP_results,axis=0)), label="ConvTemp Model")
plt.plot(np.cumsum(np.mean(TEMP_linear_results,axis=0)), label="Linear Model")
plt.legend()
if graph_titles:
    plt.title("Average cumulative RMSE of predictions based on a history of 100 states over 10 runs")
plt.xlabel("Time")
plt.ylabel("Cumulative RMSE")
plt.savefig("RMSE.png")
plt.show()

#%% Making predictions
#Generating a ground-truth
n_preds = 50000

features,t = ROSSLER_MAP.full_traj(50100, np.array([-5.75, -1.6,  0.02]))
ground_truth = features.copy()
features = features[:window]


for j in range(n_preds):
    #Make predictions with both models
    seq = features[j:j+window, 1]
    seq = torch.tensor(seq).float().view(1,-1)
    y_pred = model_conv(seq)
    y_pred = y_pred.detach().numpy()
    x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
    features = np.append(features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

#PLotting the y_value
for i in [1]:
    plt.figure(facecolor='white')
    plt.plot(features[100:3100,i], label ="ConvTemp Model")
    plt.plot(ground_truth[100:3100,i], label="Ground truth")

    plt.legend(loc='lower left')
    if graph_titles:
        plt.title("Predictions of the y-value based on a history of 100 states")
    plt.xlabel("Time")
    plt.ylabel("Predictions")
    plt.savefig("predictions.png")
    plt.show()

 #%% PDF of trajectories
i=1
x_eval = np.linspace(ground_truth[100:,i].min()-1, ground_truth[100:,i].max()+1, num=200)

kde_gt_y = stats.gaussian_kde(ground_truth[100:,i],bw_method=0.01)
kde_temp_y = stats.gaussian_kde(features[100:,i],bw_method=0.01)

if graph_titles:
    plt.title("Gaussian KDE of predictions")
plt.figure(facecolor='white')
plt.plot(x_eval, kde_temp_y(x_eval), label="ConvTemp Model")
plt.plot(x_eval, kde_gt_y(x_eval), label="Ground Truth")

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
z = features[:,1]

zf = fft(z)
zplot =fftshift(zf)
yf = fft(y)
xf = fftfreq(N, T)
xf = fftshift(xf)
yplot = fftshift(yf)
plt.figure(facecolor='white')
if graph_titles:
    plt.title("Fourier transform of the predictions")
plt.plot(xf, 1.0/N * np.abs(zplot),label="ConvTemp Model")
plt.plot(xf, 1.0/N * np.abs(yplot),label="Ground Truth")

plt.xlim(-50,50)
plt.ylim(0,1.5)
plt.xlabel("xi")
plt.ylabel("Fourier transform magnitude")
plt.legend()
plt.savefig("Fourier.png")
plt.show()

#%%
start = 102
fig = plt.figure(facecolor='white')
ax = fig.gca(projection='3d')
ax.plot(features[start:3000,0], features[start:3000,1], features[start:3000,2], label="ConvTemp Model")
ax.plot(ground_truth[:3000,0], ground_truth[:3000,1], ground_truth[:3000,2], label="Ground Truth")
fig.legend()
plt.xlabel("x")
plt.ylabel("y")

if graph_titles:
    plt.title("Trajectories in 3D")
plt.savefig("3D_traj.png")
plt.show()

#%% 2D trajectories in the x-y plane
fig = plt.figure(figsize=(10,8), facecolor='white')
plt.plot(features[start:3000,0], features[start:3000,1], label="ConvTemp Model")
plt.plot(ground_truth[start:3000,0], ground_truth[start:3000,1], label="Ground Truth")
plt.scatter(-3.1529e-9,-0.0515, label="Predicted trajectory equilibrium point")
plt.scatter(7.0210e-3,-0.0351, label="Theoretical equilibrium point")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
if graph_titles:
    plt.title("Trajectories in the x-y plane")
plt.savefig("2D_traj.png")
plt.show()

#%%
