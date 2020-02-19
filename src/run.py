import mine
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

example = 'Gaussian'

if example == 'Gaussian':

    n_rhos = 19
    d = 10
    MI_real = np.empty(n_rhos)
    MI_mine = np.empty(n_rhos)

    rhos = np.linspace(-0.95, 0.95, n_rhos)

    for i, rho in enumerate(rhos):

        print(f'Computing MI for rho {rho:.2f}')
        
        # Compute the real MI
        cov = np.eye(2*d)
        cov[d:2*d, 0:d] = rho * np.eye(d) 
        cov[0:d, d:2*d] = rho * np.eye(d)
        MI_real[i] = - 0.5 * np.log(np.linalg.det(cov)) / np.log(2)
        
        # Compute the estimation of the MI 
        dataset = utils.BiVariateGaussianDatasetForMI(d, rho, 5000)
        mine_network = mine.MINE(d,d, hidden_size=100, moving_average_rate=0.1).to(device)
        MI_mine[i] = mine_network.train(dataset, batch_size = 4096, n_iterations=int(5e3), n_verbose=-1, n_window=10, save_progress=-1)

        print(f'Rho {rho:.2f}: Real ({MI_real[i]:.3f}) - Estimated ({MI_mine[i]:.3f})')

    plt.plot(rhos, MI_real,'black',label='Real MI')
    plt.plot(rhos, MI_mine,'orange',label='Estimated MI')
    plt.legend()
    plt.savefig(f'../figures/{d}-dimensional_gaussian_mi.png')

if example == 'MNIST':

    n_iterations = int(25e3) 
    n_window = 100
    save_progress = 125

    dataset = utils.MNISTForMI()
    mine_network = mine.MINE((28,28),1,network_type='cnn').to(device)
    MI_mine = mine_network.train(dataset, batch_size=1024, n_iterations=n_iterations, n_verbose=1000, n_window=n_window, learning_rate=1e-3, decay_rate=0.9, n_decay=-1, save_progress=save_progress)
    iterations = np.linspace(save_progress,n_iterations,int(n_iterations/save_progress))
    plt.plot(iterations, np.log2(10)*np.ones_like(iterations) ,'black',label='Real MI')
    plt.plot(iterations, MI_mine, 'orange', label='Estimated MI')
    plt.legend() 
    plt.savefig(f'../figures/MNIST_mi.png')
    


