import torch, torchvision
import math
import utils
import progressbar
import numpy as np
import gc

torch.manual_seed(0)

class MLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size = 100):
        super(MLP, self).__init__()
    
        self.f_theta = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_size, 1),
        )
    
    def forward(self, X, Y):

        Z = torch.cat((X,Y),1)
        return self.f_theta(Z)

class CNN(torch.nn.Module):

    def __init__(self, dimX, dimY):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,5,5,1,padding=1),
            torch.nn.MaxPool2d(5,2,2),
            torch.nn.ReLU6(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(5,50,5),
            torch.nn.MaxPool2d(5,2,2),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(50*5*5 + 1,100),
            torch.nn.ReLU6(inplace=True),
        )
        self.fc2 = torch.nn.Linear(100 + 1,1)
    
    def forward(self, X, Y):

        Y = Y.view(-1,1)
        Z = self.conv1(X + Y.view((-1,) + (1,)*(len(X.shape)-1)))
        Z = self.conv2(Z + Y.view((-1,) + (1,)*(len(Z.shape)-1)))
        Z = self.fc1(torch.cat([Z.flatten(1),Y],1))
        return self.fc2(torch.cat([Z,Y],1))

class MINE(torch.nn.Module):

    def __init__(self, dimX, dimY, moving_average_rate = 0.01, hidden_size = 100, network_type = 'mlp'):
        super(MINE, self).__init__()

        if network_type == 'mlp':
            self.network = MLP(dimX + dimY, hidden_size)
        elif network_type == 'cnn':
            self.network = CNN(dimX, dimY)
        self.network.apply(utils.weight_init)
        self.moving_average_rate = moving_average_rate
    
    def get_mi(self, X, Y, Y_tilde):

        T = self.network(X, Y).mean()
        expT = torch.exp(self.network(X, Y_tilde)).mean()
        mi = (T - torch.log(expT)).item() / math.log(2)
        return mi, T, expT
    
    def train(self, dataset, learning_rate = 1e-3, batch_size = 256, n_iterations = int(5e3), n_verbose = 1000, n_window = 100, decay_rate = 0.9, n_decay = -1):

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        if n_decay > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,n_decay,decay_rate)

        iteration = 0
        moving_average_expT = 1
        mi = np.empty(n_window)

        for iteration in progressbar.progressbar(range(n_iterations)):

            X, Y, Y_tilde = dataset.sample_batch(batch_size)
            X = torch.autograd.Variable(X)
            Y = torch.autograd.Variable(Y)
            Y_tilde = torch.autograd.Variable(Y_tilde)

            mi_lb, T, expT = self.get_mi(X, Y, Y_tilde)
            moving_average_expT = ((1-self.moving_average_rate) * moving_average_expT + self.moving_average_rate * expT).item()
            loss = -1.0 * (T - expT / moving_average_expT)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mi[iteration % n_window] = mi_lb
                if iteration >= n_window and iteration % n_verbose == n_verbose - 1:
                    print(f'Iteration {iteration+1}: {np.mean(mi)}')
            
                if iteration % n_decay == n_decay - 1:
                    scheduler.step()

        return np.mean(mi)
