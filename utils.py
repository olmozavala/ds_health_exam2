import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def generate_model1_data():
    torch.manual_seed(10)
    n = 500
    x = torch.linspace(- np.pi , np.pi, n)
    return generate_data(x, 'harmonic', 0.2, n)

def generate_model2_data():
    torch.manual_seed(33)
    n = 1000
    x = torch.linspace(0 , 4*np.pi, n)
    return generate_data(x, 'harmonic', 2.5, n)

def generate_data(x, obs_type='linear', error_size=2, n = 100):
    '''Generates data for linear, quadratic or harmonic function'''
    if obs_type == 'linear':
        m = 1*torch.rand(1) -1
        b = 1*torch.rand(1) -1
        y = x*m + b + torch.rand(x.shape)*error_size
    elif obs_type == 'quadratic':
        a = 2*torch.rand(1) -1
        b = 2*torch.rand(1) -1
        c = 2*torch.rand(1) -1
        y =  a*x**2 + b*x + c + torch.rand(x.shape)*error_size
    elif obs_type == 'harmonic':
        a = 2 * torch.rand(1) - 1
        b = 8 * torch.rand(1) - 1
        c = 5 * torch.rand(1) - 1
        d = 2 * torch.rand(1) - 1
        y = a*x**2 + b*np.sin(x) + c*np.cos(3*x) + d + torch.rand(x.shape)*error_size

    idxs = torch.randperm(x.shape[0])
    return x[idxs], y[idxs]

def split_data(x,y, train_percentage=0.8, validation_percentage=0.1):
    '''Splits data into train, validation and test'''
    train_size = int(x.shape[0]*train_percentage)
    validation_size = int(x.shape[0]*validation_percentage)
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_validation = x[train_size:train_size + validation_size]
    y_validation = y[train_size:train_size + validation_size]
    x_test = x[train_size + validation_size:]
    y_test = y[train_size + validation_size:]
    return x_train, y_train, x_validation, y_validation, x_test, y_test

def plot_data(x, y, x_val, y_val, model, loss_history, loss_history_val, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    X = torch.reshape(x, (x.shape[0], 1)).to(device)
    model_y = model(X).cpu().detach().numpy()
    axs[0].plot(x, y, '.')
    axs[0].plot(x_val, y_val, 'y.')
    axs[0].plot(x, model_y, color='r')
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].semilogy(range(len(loss_history)), loss_history, color='r', label='Training')
    axs[1].semilogy(range(len(loss_history_val)), loss_history_val, color='g', label='Validation')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("MSE")
    axs[1].legend()
    plt.suptitle(title, fontsize=17)
    plt.tight_layout()
    plt.show()