from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np

from tempconv_model import Conv_temporal, Linear_temporal
from rossler_map import RosslerMap

#Loading model
model_conv = torch.load("K_temporal_conv.pth",  map_location=torch.device('cpu'))

#Generating history
delta_t = 1e-2
ROSSLER_MAP = RosslerMap(delta_t=delta_t)
INIT = np.array([-5.75, -1.6,  0.02])
ground_truth,t = ROSSLER_MAP.full_traj(3100, INIT)

#%%
def model(y):
    y_pred = model_conv(y)

    y_pred = torch.cat((y[0,1:],y_pred[0]))
    return y_pred.unsqueeze(0).unsqueeze(0)

def model_jacobian(x):
    return torch.autograd.functional.jacobian(model, x, create_graph=False, strict=False)#-np.eye

#Initialising the algorithm
y = torch.tensor(ground_truth[0:100,1]).unsqueeze(0).float()


tol =1
i=1
while tol>1e-3:
    tol = y
    J = f(y).squeeze(0).squeeze(0).squeeze(1)
    b = model(y).squeeze(0).squeeze(0)
    y = y-0.001*torch.linalg.solve(J, b)
    tol = torch.norm(tol-y)
    i+=1
    print(i, tol, model(y)[:,:,-1])
    if i ==20000:
        break

#EP y-coordinate
y_EP = model(y)[:,:,-1]
y_previous = model(y)[:,:,-2]

#EP x-coordinate
x_EP = (y_EP-y_previous)/delta_t - 0.2 * y_EP
x_previous = (y_previous-model(y)[:,:,-3])/delta_t - 0.2 * y_previous

#EP z-coordinate
z_EP = -y_EP - (x_EP-x_previous)/delta_t

print("equilibrium point:", x_EP, y_EP, z_EP)
