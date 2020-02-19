import torch, torchvision
import numpy as np
import argparse

np.random.seed(0)

class BiVariateGaussianDatasetForMI(torch.utils.data.Dataset):

    def __init__(self, d, rho, N):
        super(BiVariateGaussianDatasetForMI, self).__init__()

        cov = torch.eye(2*d) 
        cov[d:2*d, 0:d] = rho * torch.eye(d) 
        cov[0:d, d:2*d] = rho * torch.eye(d)
        f = torch.distributions.MultivariateNormal(torch.zeros(2*d), cov)
        Z = f.sample((N,))
        self.X, self.Y = Z[:,:d], Z[:,d:2*d]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def sample_batch(self, batch_size):

        index_joint = np.random.choice(range(self.__len__()),size=batch_size,replace=False)
        index_marginal = np.random.choice(range(self.__len__()),size=batch_size,replace=False)
        return self.X[index_joint], self.Y[index_joint], self.Y[index_marginal]

class MNISTForMI(torch.utils.data.Dataset):

    def __init__(self):
        super(MNISTForMI, self).__init__() 

        dataset = torchvision.datasets.MNIST('../data', train=True, download=True)
        X = dataset.data.float()
        self.X = ((X - X.mean()) / X.std()).unsqueeze(1)
        self.Y = dataset.targets.float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def sample_batch(self, batch_size):

        index_joint = np.random.choice(range(self.__len__()),size=batch_size,replace=False)
        index_marginal = np.random.choice(range(self.__len__()),size=batch_size,replace=False)
        return self.X[index_joint], self.Y[index_joint], self.Y[index_marginal]

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def get_args():

    parser = argparse.ArgumentParser(
        description = 'Run the experiments of MINE',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--figs_dir', default = '../figures/',
        help = 'folder to output the resulting images')
    parser.add_argument('--n_iterations', type = int, default = int(5e3),
        help = 'number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'mini-batch size for the SGD')
    parser.add_argument('--learning_rate', type = float, default = 1e-3,
        help = 'initial learning rate')
    parser.add_argument('--n_verbose', type = int, default = -1,
        help = 'number of iterations for showing the current MI, if -1, then never')
    parser.add_argument('--n_window', type = int, default = 100,
        help = 'number of iterations taken into consideration for the averaging the MI (moving average)')
    parser.add_argument('--save_progress', type = int, default = -1, 
        help = 'sampling rate of the MI, if -1, nothing is saved')    
    parser.add_argument('--d', type = int, default = 1,
        help = 'dimensionality of the Gaussians in the example')
    parser.add_argument('--n_rhos', type = int, default = 19,
        help = 'number of rhos for the Gaussian experiment')
    parser.add_argument('--example', choices = ['Gaussian', 'MNIST'], default = 'Gaussian',
        help = 'example to run')    

    return parser.parse_args()



