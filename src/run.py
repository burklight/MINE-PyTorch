import mine
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = utils.get_args()

if args.example == 'Gaussian':

    n_rhos = args.n_rhos
    d = args.d
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
        mine_network = mine.MINE(d,d, hidden_size=100).to(device)
        MI_mine[i] = mine_network.train(dataset, batch_size = args.batch_size, n_iterations=args.n_iterations, n_verbose=args.n_verbose, n_window=args.n_window, save_progress=args.save_progress)

        print(f'Rho {rho:.2f}: Real ({MI_real[i]:.3f}) - Estimated ({MI_mine[i]:.3f})')

    plt.plot(rhos, MI_real,'black',label='Real MI')
    plt.plot(rhos, MI_mine,'orange',label='Estimated MI')
    plt.legend()
    plt.savefig(f'../figures/{d}-dimensional_gaussian_mi.png')

if args.example == 'MNIST':

    dataset = utils.MNISTForMI()
    mine_network = mine.MINE((28,28),1,network_type='cnn').to(device)
    MI_mine = mine_network.train(dataset, batch_size = args.batch_size, n_iterations=args.n_iterations, n_verbose=args.n_verbose, n_window=args.n_window, save_progress=args.save_progress)
    iterations = np.linspace(args.save_progress,args.n_iterations,int(args.n_iterations/args.save_progress))
    plt.plot(iterations, np.log2(10)*np.ones_like(iterations) ,'black',label='Real MI')
    plt.plot(iterations, MI_mine, 'orange', label='Estimated MI')
    plt.legend() 
    plt.savefig(f'../figures/MNIST_mi.png')
    


