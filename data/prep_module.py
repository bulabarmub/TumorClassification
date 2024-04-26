import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew, kurtosis
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, activations):
        super(Net, self).__init__()
        self.activations = activations
        
        self.linear1 = nn.Linear(29, 256)
        self.linear2 = nn.Linear(29 + 256, 256)
        self.linear3 = nn.Linear(29 + 256 + 256, 2) 
        # "почему только 3 слоя ?" - добавление параметров в модель не улучшает качество,
        # так как модель ограничена зависимостью ответа от входных данных,
        # то есть она уже выделила максимум информации из поданных признаков.
        
        self.softmax = nn.Softmax()

    def forward(self, x):
        input_1 = x
        x_1 = self.activations[0](self.linear1(input_1))

        input_2 = torch.cat((x, x_1), dim=1)
        x_2 = self.activations[1](self.linear2(input_2))

        input_3 = torch.cat((x, x_1, x_2), dim=1)
        x_3 = self.softmax(self.linear3(input_3))
        
        return x_3

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.linear1 = nn.Linear(29, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.linear1(x))
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.linear2(x))
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.linear3(x))
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv5(x))
        x = self.softmax(x)
        
        return x

class Prep:
    def __init__(self):
        self.log_reg_model = joblib.load('data/LogisticRegressionModel.sav')
        
        self.neural_net_model = Net([torch.relu, torch.relu])
        self.neural_net_model.load_state_dict(torch.load('data/NeuralNetwork.pth'))
        self.scaler = joblib.load('data/StandartScaler.sav')
        
        self.neural_net_2D = FCN()
        self.neural_net_2D.load_state_dict(torch.load('data/NeuralNetwork2D.pth'))

        self.neural_net_2D_2 = FCN()
        self.neural_net_2D_2.load_state_dict(torch.load('data/NeuralNetwork2D2.pth'))
        
        self.scaler_2D = joblib.load('data/StandartScaler2D.sav')
    
    def draw_pies(self, log_reg_probas, nn_probas):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        labels = ['healthy', 'sick']
        axs[0].set_title("Logistic Regression model predictions")
        axs[0].pie(log_reg_probas, labels=labels, autopct='%1.6f%%')
        axs[1].set_title("Neural Network model predictions")
        axs[1].pie(nn_probas, labels=labels, autopct='%1.6f%%')
        
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.show()
    
    def get_features(self, x, y):
        return [np.min(x), np.max(x), np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75), np.ptp(x, 0), np.mean(x), np.median(x),
                    np.std(x), np.var(x), skew(x.flatten()), kurtosis(x.flatten()),
                    np.min(y), np.max(y), np.percentile(y, 25), np.percentile(y, 50), np.percentile(y, 75), np.ptp(y, 0), np.mean(y), np.median(y),
                    np.std(y), np.var(y), skew(y.flatten()), kurtosis(y.flatten()),
                    np.correlate(x, y, mode='valid')[0], np.corrcoef([x, y])[0, 1], np.cov(np.array([x, y]))[0, 0], np.cov(np.array([x, y]))[0, 1],
                    np.cov(np.array([x, y]))[1, 1]]
    
    def get_1D_results(self, path_to_sample):
        # read data
        df = pd.read_csv(
            path_to_sample, sep='\t', skiprows=[0],
            header=None, names=['Wave', 'Intensity'])
        waves = df["Wave"].to_numpy()
        intensities = df["Intensity"].to_numpy()
        
        # extract features
        features = self.get_features(waves, intensities)
        features_scaled = self.scaler.transform([features])[0]
        
        # get predictions
        log_reg_probas = self.log_reg_model.predict_proba([features_scaled])[0]
        
        tensor = torch.from_numpy(np.array([features_scaled])).to(torch.float32)
        nn_probas = list(self.neural_net_model(tensor)[0].detach().numpy())
        
        # draw results
        self.draw_pies(log_reg_probas, nn_probas)
    
    def get_2D_results(self, path_to_sample):
        data = None
        with open(path_to_sample, 'r') as fin:
            data = fin.readlines()
        mapped_xy = dict()
        xs = []
        ys = []
        for line in data[1:]:
            x, y, wave, intensity = map(float, line.split('\t'))
            if (x, y) not in mapped_xy.keys():
                mapped_xy[(x, y)] = {
                    'wave': [],
                    'intensity': []
                }
                xs.append(x)
                ys.append(y)
            mapped_xy[(x, y)]['wave'].append(wave)
            mapped_xy[(x, y)]['intensity'].append(intensity)
        
        xs = sorted(list(set(xs)))
        ys = sorted(list(set(ys)))
        
        width = len(xs)
        height = len(ys)
        
        table = [[[] for _ in range(20)] for __ in range(20)]
        table_formed = [[[[] for _ in range(20)] for __ in range(20)] for channel in range(29)]
        
        for i in range(20):
            for j in range(20):
                x_index = int((i / 20) * width)
                y_index = int((j / 20) * height)
                d = mapped_xy[(xs[x_index], ys[y_index])]
                features = self.get_features(np.asarray(d['wave']), np.asarray(d['intensity']))
                table[i][j] = self.scaler_2D.transform([features])[0]
                for channel in range(29):
                    table_formed[channel][i][j] = table[i][j][channel]
        
        tensor = torch.from_numpy(np.array([table_formed])).to(torch.float32)
        
        nn_probas = self.neural_net_2D(tensor)[0].detach().numpy()
        nn_probas_2 = self.neural_net_2D_2(tensor)[0].detach().numpy()
        
        x = []
        y = []
        labels = []
        labels2 = []
        for i in range(20):
            for j in range(20):
                x_index = int((i / 20) * width)
                y_index = int((j / 20) * height)
                x.append(xs[x_index])
                y.append(ys[y_index])
                labels.append([min(1.0, nn_probas[1][i][j] + nn_probas[2][i][j]), min(1.0, nn_probas[0][i][j] + nn_probas[2][i][j]), 0])
                labels2.append([min(1.0, nn_probas_2[1][i][j] + nn_probas_2[2][i][j]), min(1.0, nn_probas_2[0][i][j] + nn_probas_2[2][i][j]), 0])
        
        color_map = {0: 'green', 1: 'red', 2: 'yellow'}

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
        axs[0].scatter(x, y, c=[label for label in labels])
        axs[0].set_title('Train Pool based predictions')
        
        axs[1].scatter(x, y, c=[label for label in labels2])
        axs[1].set_title('Custom marking based predictions')
        
        plt.subplots_adjust(wspace=0.6, hspace=0.3)
        plt.show()
    
    def get_results(self, path_to_sample):
        data = None
        with open(path_to_sample, 'r') as fin:
            data = fin.readlines()[0]
        columns = len(data.split('\t\t'))
        if columns == 2:
            self.get_1D_results(path_to_sample)
        elif columns == 4:
            print('green for healthy\nred for sick\nyellow for near-tumor')
            self.get_2D_results(path_to_sample)