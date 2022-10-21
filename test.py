import pickle
import numpy as np
import matplotlib.pyplot as plt
from example_network import ExampleNetwork
from auto_diff import gradient, comp_graph as cg
from pytorch_network import Network


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

    x = x[:201]
    y = y[:201]
    losses = []
    for epoch_num in range(EPOCHS+1):
        # Reset grads?

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
            # new_params[param_name] = nn.params[param_name] - mean_grad*LR
        nn.params = nn.update_parameters(**new_params)

    # Plot avg loss over epochs
    x_d = range(EPOCHS+1)
    plt.plot(x_d, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()


def pytorch_test(x, y, w1, w2, w3, b1, b2, b3):
    network = Network(weights=[w1.T, b1.T, w2.T, b2.T, w3.T, b3.T])
    EPOCHS = 5
    LR = 1./100.

    test_1, all_non_leaf_1 = network.one_forward(x, y)

    network_2 = ExampleNetwork(seed=7, **{'w1': w1_pt, 'w2': w2_pt, 'w3': w3_pt,
                                          'b1': b1, 'b2': b2, 'b3': b3})
    test_2, all_non_leaf_2 = network_2.one_forward(x, y)

    print('done testing')



    network.train(x, y, epochs=EPOCHS, learning_rate=LR)
    print('done')



with open('data/assignment-one-test-parameters.pkl', 'rb') as f:
    x = pickle.load(f)

x_train = x['inputs']
y_train = x['targets'][..., np.newaxis]
w1 = x['w1']
w1_pt = w1.T
w1_our = np.reshape(w1, [w1.shape[1], w1.shape[0]], order='F')
w2 = x['w2']
w2_pt = w2.T
w2_our = np.reshape(w2, [w2.shape[1], w2.shape[0]], order='F')
w3 = x['w3']
w3_pt = w3.T
w3_our = np.reshape(w3, [w3.shape[1], w3.shape[0]], order='F')
b1 = x['b1']
b2 = x['b2']
b3 = x['b3']

network = ExampleNetwork(seed=7, **{'w1': w1_pt, 'w2': w2_pt, 'w3': w3_pt,
                                    'b1': b1, 'b2': b2, 'b3': b3})

# print_layer1_grads_test(network, x_train, y_train)
# train_five_epochs_test(network, x_train, y_train)
pytorch_test(x_train, y_train, w1_pt, w2_pt, w3_pt, b1, b2, b3)
