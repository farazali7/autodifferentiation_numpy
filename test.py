import pickle
import numpy as np
import matplotlib.pyplot as plt
from example_network import ExampleNetwork
from auto_diff import gradient, comp_graph as cg

EPOCHS = 5
LR = 1./100.


def print_layer1_grads(nn, x, y):
    '''
    Print gradients of layer 1 for first pair of training data.
    '''
    first_pair_outputs = nn.forward_pass(cg.reshape(x[0], [1, -1]))
    first_pair_loss = nn.compute_loss(first_pair_outputs, cg.reshape(y[0], [1, -1]))
    first_pair_grads = gradient(first_pair_loss)

    # Print first layer gradients
    print(f'First layer weight gradients: \n {first_pair_grads["w1"]} \n')
    print(f'First layer bias gradients: \n {first_pair_grads["b1"]}')


def train_five_epochs(nn, x, y):
    '''
    Train a network for 5 epochs, while calculating avereage gradient w.r.t all parameters each epoch, and updating
    the parameters with LR = 1/100. Plot a training curve of average loss over dataset before and after each update.
    '''

    losses = []
    for epoch_num in range(EPOCHS+1):

        # Perform forward pass
        nn_output = nn.forward_pass(x)

        # Compute loss
        loss = nn.compute_loss(nn_output, y)
        mean_loss = cg.mean(loss)
        losses.append(mean_loss)

        # Compute gradients
        grads = gradient(mean_loss)

        # Calculate mean gradient
        total_grad_value = np.sum([np.sum(x) for x in grads.values()])
        total_grads = np.sum([x.size for x in grads.values()])
        mean_grad = total_grad_value / total_grads

        print(f'EPOCH: {epoch_num}, AVG LOSS: {mean_loss}, AVG GRAD: {mean_grad}')

        new_params = {}
        # Perform parameter updates
        for param_name in grads.keys():
            new_params[param_name] = nn.params[param_name] - grads[param_name]*LR
        nn.params = nn.update_parameters(**new_params)

    # Plot avg loss over epochs
    x_d = range(EPOCHS+1)
    plt.plot(x_d, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid()
    plt.show()


def train_five_epochs_binary_cls(nn, x, y):
    '''
    Train a network for 5 epochs, while calculating avereage gradient w.r.t all parameters each epoch, and updating
    the parameters with LR = 1/100. Plot a training curve of average loss over dataset before and after each update.
    This test uses a binary classification loss instead of a regression loss.
    '''

    def binary_cls_loss(preds, actual):
        # k = 1 with the network
        return -1*(actual*cg.log(preds) + (1-actual)*cg.log(1-preds))

    # Binarize the labels
    y_binary = np.where(y < 0, 0, 1)

    losses = []
    for epoch_num in range(EPOCHS+1):

        # Perform forward pass
        nn_output = nn.forward_pass(x)

        # Compute loss
        nn_output_binary_cls = 1./(1 + cg.exp(-1*nn_output))
        loss = binary_cls_loss(nn_output_binary_cls, y_binary)
        mean_loss = cg.mean(loss)
        losses.append(mean_loss)

        # Compute gradients
        grads = gradient(mean_loss)

        # Calculate mean gradient
        total_grad_value = np.sum([np.sum(x) for x in grads.values()])
        total_grads = np.sum([x.size for x in grads.values()])
        mean_grad = total_grad_value / total_grads

        print(f'EPOCH: {epoch_num}, AVG LOSS: {mean_loss}, AVG GRAD: {mean_grad}')

        new_params = {}
        # Perform parameter updates
        for param_name in grads.keys():
            new_params[param_name] = nn.params[param_name] - grads[param_name]*(LR/10.)
        nn.params = nn.update_parameters(**new_params)

    # Plot avg loss over epochs
    x_d = range(EPOCHS+1)
    plt.plot(x_d, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid()
    plt.show()


with open('data/assignment-one-test-parameters.pkl', 'rb') as f:
    x = pickle.load(f)

x_train = x['inputs']
y_train = x['targets'][..., np.newaxis]

w1 = x['w1']
w1_pt = w1.T

w2 = x['w2']
w2_pt = w2.T

w3 = x['w3']
w3_pt = w3.T

b1 = x['b1']
b2 = x['b2']
b3 = x['b3']

network = ExampleNetwork(seed=7, **{'w1': w1_pt, 'w2': w2_pt, 'w3': w3_pt,
                                    'b1': b1, 'b2': b2, 'b3': b3})

print_layer1_grads(network, x_train, y_train)
train_five_epochs(network, x_train, y_train)
train_five_epochs_binary_cls(network, x_train, y_train)
