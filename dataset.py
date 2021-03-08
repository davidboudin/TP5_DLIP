from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Rossler(Dataset):
    def __init__(self, df, window=200, pred_window=1, normalise=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.window = window
        self.pred_window = pred_window
        self.length = int(len(df) / (window+pred_window))
        self.sample_len = (window+pred_window)
        self.df = df.iloc[:self.length*(window+pred_window)]
        self.normalise = normalise

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.df.iloc[idx*self.window:idx*self.window+self.window].values
        #print(idx*self.sample_len, idx*self.sample_len+self.window)
        target = self.df.iloc[idx*self.window+self.window:idx*self.window+self.window+self.pred_window].values

        if self.normalise == True:
            mean = self.df.mean()
            std = self.df.std()
            features = (features-mean)/std

        return torch.tensor(features), torch.tensor(target)

if __name__ == "__main__":

    df = pd.read_csv('data.csv')
    scaler = MinMaxScaler
    y_data = df["y"]

    data = Rossler(df=y_data, window=200,pred_window=1, normalise=False)

    print(data[1])

    data = Rossler(df=y_data, window=200,pred_window=1, normalise=True)

    print(data[1])
