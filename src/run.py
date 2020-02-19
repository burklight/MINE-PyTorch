import mine
import utils
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

example = 'MNIST'

if example == 'Gaussian':

    n_rhos = 19
    d = int(28*28/2)
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
        mine_network = mine.MINE(d,d, hidden_size=100)
        MI_mine[i] = mine_network.train(dataset, batch_size = 128, n_iterations=int(5e3), n_verbose=-1, n_window=250)

        print(f'Rho {rho:.2f}: Real ({MI_real[i]:.3f}) - Estimated ({MI_mine[i]:.3f})')

    plt.plot(rhos, MI_real,'black',label='Real MI')
    plt.plot(rhos, MI_mine,'orange',label='Estimated MI')
    plt.legend()
    plt.savefig(f'../figures/{d}-dimensional_gaussian_mi_.png')

if example == 'MNIST':

    dataset = utils.MNISTForMI()
    mine_network = mine.MINE((28,28),1,network_type='cnn')
    MI_mine =mine_network.train(dataset, batch_size=128, n_iterations=int(5e4), n_verbose=1000, n_window=250, learning_rate=1e-3, decay_rate=0.9, n_decay=-1)
    print(MI_mine)
    


