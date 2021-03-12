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
from tempconv_model import Conv_temporal, Linear_temporal
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
window = 100

#Loading models
model_temp = torch.load("K_temporal.pth",  map_location=torch.device('cpu'))
model_linear = torch.load("K_temporal_linear.pth")
model_temp.eval()
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
        y_pred = model_temp(seq)
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

len(TEMP_results)
#Plot of the cumulative RMSE
plt.plot(np.cumsum(np.mean(TEMP_results,axis=0)), label="Conv")
plt.plot(np.cumsum(np.mean(TEMP_linear_results,axis=0)), label="Linear")
plt.legend()
plt.title("Cumulative RMSE of predictions based on a history of 50 states")
plt.xlabel("Time")
plt.ylabel("Cumulative RMSE")
plt.savefig("RMSE.png")
plt.show()

#%%
i=1
#plt.plot(ground_truth[window:,1], label="GT")
#plt.plot(features_linear[window:,1], label ="Linear")
plt.plot(features[window:,1], label="Conv")
plt.legend()
plt.title("Predictions of the y-value based on a history of 50 states")
plt.xlabel("Time")
plt.ylabel("Predictions")
plt.savefig("predictions.png")
plt.show()
#%% Making predictions
#Generating a ground-truth
features,t = ROSSLER_MAP.full_traj(3100, random_init[:,0])
ground_truth = features.copy()
features = features[:window]


for j in range(n_preds):
    #Make predictions with both models
    seq = features[j:j+window, 1]
    seq = torch.tensor(seq).float().view(1,-1)
    y_pred = model_temp(seq)
    y_pred = y_pred.detach().numpy()
    x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
    features = np.append(features, np.array([x_pred, y_pred, z_pred]).reshape(1,3),axis=0)

#PLotting the y_value
for i in [2]:
    plt.plot(ground_truth[:,i], label="GT")
    #plt.plot(features[:,i], label ="Temporal")
    #plt.plot(features[:,i], label="LSTM")
    plt.legend()
    plt.title("Predictions of the y-value based on a history of 50 states")
    plt.xlabel("Time")
    plt.ylabel("Predictions")
    plt.savefig("predictions.png")
    plt.show()

 #%%
i=1
x_eval = np.linspace(ground_truth[:,i].min()-1, ground_truth[:,i].max()+1, num=200)

kde_gt_y = stats.gaussian_kde(ground_truth[:,i],bw_method=0.01)
kde_temp_y = stats.gaussian_kde(features[:,i],bw_method=0.01)

plt.title("Gaussian KDE of predictions")
plt.plot(x_eval, kde_gt_y(x_eval), label="Ground Truth")
plt.plot(x_eval, kde_temp_y(x_eval), label="Temp")
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

plt.title("Fourier transform of the predictions")
plt.plot(xf, 1.0/N * np.abs(yplot),label="Ground Truth")
plt.plot(xf, 1.0/N * np.abs(zplot),label="Temp")
plt.xlim(-50,50)
plt.xlabel("z")
plt.ylabel("Magnitude")
plt.legend()
plt.savefig("Fourier.png")
plt.show()

#%% Joint law - multiple Kolgomorov Smirnov Tests
T = 20
y = ground_truth[:-20,1]
y_T = ground_truth[20:,1]

y_pred = features[:-20,1]
y_pred_T = features[20:,1]


print(stats.ks_2samp(y, y_pred))
print(stats.ks_2samp(y, y_pred_T))
print(stats.ks_2samp(y_T, y_pred))
print(stats.ks_2samp(y_T, y_pred_T))

print(stats.ks_2samp(y_pred,y))
print(stats.ks_2samp(y_pred_T,y))
print(stats.ks_2samp(y_pred,y_T))
print(stats.ks_2samp(y_pred_T, y_T))




#%%
start = 102
fig = plt.figure(figsize=(30,30))
ax = fig.gca(projection='3d')
ax.plot(features[start:,0], features[start:,1], features[start:,2], label="Temp")
ax.plot(ground_truth[:,0], ground_truth[:,1], ground_truth[:,2], label="GT")
fig.legend()
plt.savefig("plot.png")

#%%

plt.plot(features[start:,0], features[start:,1], label="temp")
plt.plot(ground_truth[start:,0], ground_truth[start:,1], label="GT")
plt.scatter(7.0210e-3,-0.0351, label="Theoretical Equilibrium point")
plt.legend()
plt.savefig("2d_curves.png")

#%%
from numpy.linalg import qr, solve, norm

def master(y):

    y_pred = model_temp(y)
    #x_pred, y_pred, z_pred = y_to_full_state(features, y_pred, delta_t, params)
    y_pred = torch.cat((y[0,0,1:],y_pred[0]))
    return y_pred.unsqueeze(0).unsqueeze(0) #x_pred, y_pred, z_pred

def f(x):

    return torch.autograd.functional.jacobian(master, x, create_graph=False, strict=False)#-np.eye

x = torch.rand(50).unsqueeze(0).unsqueeze(0)
x = torch.tensor(ground_truth[50:100,1]).unsqueeze(0).unsqueeze(0).float()
tol =1
i=1
while tol>1e-3:
    #WARNING this is true for the jacobian of the continuous system!
    tol = x
    J = f(x).squeeze(0).squeeze(0).squeeze(1).squeeze(1)
    b = master(x).squeeze(0).squeeze(0)
    x = x-0.1*torch.linalg.solve(J, b)
    tol = torch.norm(tol-x)
    #print(tol)
    i+=1
    print(i, tol)
    if i ==1000:
        break
print(master(x))
