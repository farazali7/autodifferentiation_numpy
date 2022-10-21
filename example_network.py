import numpy as np
import auto_diff.comp_graph as cg
from auto_diff import gradient


class ExampleNetwork:
    '''
    A neural network with:
    - 2 inputs
    - 2 hidden layers with 10 neurons each & ReLU non-linearity
    - 1 output layer with 1 neuron
    '''
    def __init__(self, seed, **kwargs):
        '''
        Initialize example network.
        :param seed: Seed for reproducibility of parameter inits.
        '''
        np.random.seed(seed)

        # Initialize the parameters (weights and biases) of the network.
        layer_sizes = [2, 10, 10, 1]
        input_size = layer_sizes[0]
        hidden_1_size = layer_sizes[1]
        hidden_2_size = layer_sizes[2]
        output_size = layer_sizes[3]

        if kwargs is not None:
            self.params = self.update_parameters(**kwargs)
        else:
            self.params = self.default_init(input_size, hidden_1_size, hidden_2_size, output_size)

    def default_init(self, input_size, hidden_1_size, hidden_2_size, output_size):
        '''
        Default weights and biases initialization using Kaiming initialization.
        :param input_size: Int, neurons in input layer
        :param hidden_1_size: Int, neurons in first hidden layer
        :param hidden_2_size: Int, neurons in second hidden layer
        :param output_size: Int, neurons in output layer
        :return: Dictionary of networks parameters
        '''
        # Weights initialized with Kaiming initialization, biases as zero
        params = {
            'W1': cg.data_node(np.random.normal(loc=0.0,
                                                scale=np.sqrt(2. / input_size * hidden_1_size),
                                                size=(hidden_1_size, input_size)),
                               name=''),
            'W2': cg.data_node(np.random.normal(loc=0.0,
                                                scale=np.sqrt(2. / hidden_1_size * hidden_2_size),
                                                size=(hidden_2_size, hidden_1_size))),
            'W3': cg.data_node(np.random.normal(loc=0.0,
                                                scale=np.sqrt(2. / hidden_2_size * output_size),
                                                size=(output_size, hidden_2_size))),
            'b1': cg.data_node(np.zeros((1, hidden_1_size))),
            'b2': cg.data_node(np.zeros((1, hidden_2_size))),
            'b3': cg.data_node(np.zeros((1, output_size)))
        }

        return params

    def one_forward(self, input_data, y):
        params = self.params

        # Input layer activations
        params['A0'] = input_data

        # Input layer to hidden layer 1
        params['Z1'] = cg.dot(params['A0'], params['w1']) + params['b1']
        params['A1'] = cg.relu(params['Z1'])

        # Hidden layer 1 to hidden layer 2
        params['Z2'] = cg.dot(params['A1'], params['w2']) + params['b2']
        params['A2'] = cg.relu(params['Z2'])

        # Hidden layer 2 to output layer
        params['Z3'] = cg.dot(params['A2'], params['w3']) + params['b3']
        params['A3'] = params['Z3']

        loss = self.compute_loss(params['A3'], y)
        mean_loss = cg.mean(loss)

        grads, all_g = gradient(mean_loss)

        old_params = self.params.copy()
        new_params = {}
        # Perform parameter updates
        for param_name in grads.keys():
            new_params[param_name] = self.params[param_name].T - grads[param_name].T*(1./100.)
            # new_params[param_name] = new_params[param_name].T
            # new_params[param_name] = nn.params[param_name] - mean_grad*LR
        self.params = self.update_parameters(**new_params)

        return grads, all_g, old_params


    def forward_pass(self, input_data):
        params = self.params

        print(f'ALL NODES DATA?: {np.all([p.node_type == "data" for p in self.params.values()])}')
        # print(f'w1 Params: {self.params["w1"]}')
        # Input layer activations
        params['A0'] = input_data

        # Input layer to hidden layer 1
        params['Z1'] = cg.dot(params['A0'], params['w1']) + params['b1']
        params['A1'] = cg.relu(params['Z1'])

        # Hidden layer 1 to hidden layer 2
        params['Z2'] = cg.dot(params['A1'], params['w2']) + params['b2']
        params['A2'] = cg.relu(params['Z2'])

        # Hidden layer 2 to output layer
        params['Z3'] = cg.dot(params['A2'], params['w3']) + params['b3']
        params['A3'] = params['Z3']

        return params['A3']

    def compute_loss(self, preds, actual):
        # Regression loss if no. of output neurons, k = 1 in network
        return (preds - actual) ** 2 / 2

    def update_parameters(self, **new_params):
        params = {
            'w1': cg.data_node(new_params['w1'], name='w1'),
            'w2': cg.data_node(new_params['w2'], name='w2'),
            'w3': cg.data_node(new_params['w3'], name='w3'),
            'b1': cg.data_node(new_params['b1'], name='b1'),
            'b2': cg.data_node(new_params['b2'], name='b2'),
            'b3': cg.data_node(new_params['b3'], name='b3'),
        }
        return params




