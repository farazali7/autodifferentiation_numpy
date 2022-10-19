import numpy as np
import auto_diff.comp_graph as cg
from .auto_diff import gradient


# Create an example network
class ExampleNetwork:
    '''
    A neural network with:
    - 2 inputs
    - 2 hidden layers with 10 neurons each & ReLU non-linearity
    - 1 output layer with 1 neuron
    '''
    def __init__(self, seed):
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

        # Weights initialized with Kaiming initialization, biases as zero
        self.params = {
            'W1': cg.data_node(np.random.normal(loc=0.0,
                                                scale=np.sqrt(2. / input_size*hidden_1_size),
                                                size=(hidden_1_size, input_size)),
                               name=''),
            'W2': cg.data_node(np.random.normal(loc=0.0,
                                                scale=np.sqrt(2. / hidden_1_size*hidden_2_size),
                                                size=(hidden_2_size, hidden_1_size))),
            'W3': cg.data_node(np.random.normal(loc=0.0,
                                                scale=np.sqrt(2. / hidden_2_size * output_size),
                                                size=(output_size, hidden_2_size))),
            'b1': cg.data_node(np.zeros((1, hidden_1_size))),
            'b2': cg.data_node(np.zeros((1, hidden_2_size))),
            'b3': cg.data_node(np.zeros((1, output_size)))
        }

    def forward_pass(self, input_data):
        params = self.params

        # Input layer activations
        params['A0'] = input_data

        # Input layer to hidden layer 1
        params['Z1'] = cg.dot(params['W1'], params['A0'])
        params['A1'] = cg.relu(params['Z1'])

        # Hidden layer 1 to hidden layer 2
        params['Z2'] = cg.dot(params['W2'], params['A1'])
        params['A2'] = cg.relu(params['Z2'])

        # Hidden layer 2 to output layer
        params['Z3'] = cg.dot(params['W3'], params['A2'])
        params['A3'] = params['Z3']

        return params['A3']

    def backward_pass(self, preds, actual):
        params = self.params
        grads = {}




