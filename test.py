import pickle
import numpy as np
from example_network import ExampleNetwork
from auto_diff import gradient, comp_graph as cg


def print_layer1_grads_test(nn, x, y):
    '''
    Print gradients of layer 1 for first pair of training data.
    '''
    first_pair_outputs = nn.forward_pass(cg.reshape(x[0], [1, -1]))
    first_pair_loss = nn.compute_loss(first_pair_outputs, cg.reshape(y[0], [1, -1]))
    first_pair_grads = gradient(first_pair_loss)

    # Print first layer gradients
    print(f'First layer weight gradients: \n {first_pair_grads["w1"]} \n')
    print(f'First layer bias gradients: \n {first_pair_grads["b1"]}')


def train_five_epochs_test(nn, x, y):
    '''
    Train a network for 5 epochs, while calculating avereage gradient w.r.t all parameters each epoch, and updating
    the parameters with LR = 1/100. Plot a training curve of average loss over dataset before and after each update.
    '''
    EPOCHS = 5
    LR = 1./100.

    losses = []
    for epoch_num in range(1, EPOCHS+1):
        # Reset grads?

        # Perform forward pass
        nn_output = nn.forward_pass(x)

        # Compute loss
        loss = nn.compute_loss(nn_output, y)
        mean_loss = np.mean(loss)
        losses.append(mean_loss)
        print(f'EPOCH: {epoch_num}, AVG LOSS: {mean_loss}')

        # Compute gradients
        grads = gradient(loss)

        # Calculate mean gradient
        # mean_grad = np.array(grads)

        new_params = {}
        # Perform parameter updates
        for param_name in grads.keys():
            new_params[param_name] = nn.params[param_name] - grads[param_name]*LR
        nn.params = nn.update_parameters(**new_params)



with open('data/assignment-one-test-parameters.pkl', 'rb') as f:
    x = pickle.load(f)

x_train = x['inputs']
y_train = x['targets'][..., np.newaxis]
w1 = x['w1'].T
w2 = x['w2'].T
w3 = x['w3'].T
b1 = x['b1']
b2 = x['b2']
b3 = x['b3']

network = ExampleNetwork(seed=7, **{'w1': w1, 'w2': w2, 'w3': w3,
                                    'b1': b1, 'b2': b2, 'b3': b3})

print_layer1_grads_test(network, x_train, y_train)
# train_five_epochs_test(network, x_train, y_train)
