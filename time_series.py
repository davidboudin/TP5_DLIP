import argparse
from scipy.interpolate import interp1d
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = 10000//self.delta_t

        self.rosler_nn = torch.load("K_temporal.pth")
        self.initial_condition = np.array(value.init)

    def full_traj(self):
        # run your model to generate the time series with nb_steps
        # just the y coordinate is necessary.
        initial_condition=self.initial_condition
        print(initial_condition)
        self.rosler_nn.eval()

        y = 0
        #if your delta_t is different to 1e-2 then interpolate y
        #in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        return y

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)

        np.save('traj.npy',y)

    
if __name__ == '__main__':

    delta_t = 1e-2
    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj()

    ROSSLER.save_traj(y)
