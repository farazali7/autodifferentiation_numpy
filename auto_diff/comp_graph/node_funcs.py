import numpy as np
from .node import Node

'''
This file contains common functions that are Node-class friendly. Common array operations
in numpy should be replaced by the ones defined here to enable creation of accurate of computational graph.
'''


def data_node(value, name=None, order='F'):
    '''
    Create a data node in the computational graph.
    :param value: np.ndarray or Number
    :param name: String, optional name for Node
    :param order: String, one of {'C' for row-major, or 'F' for column-major order}
    :return: Node containing input data
    '''
    return Node.create_data_node(value, name=name, order=order)


def const_node(value, name=None):
    '''
    Create a constant node in the computational graph.
    :param value: np.ndarray or Number
    :param name: String, optional name for Node
    :return: Node containing input data
    '''
    return Node.create_const_node(value, name=name) if not isinstance(value, Node) else value


def sum(arr, axis=None, keepdims=False, name=None):
    '''
    Creates Node representing a sum.
    :param arr: Node or nd.array or Number, array on which to sum
    :param axis: Int, axis to sum on
    :param keepdims: Boolean, should dimensions of arr be kept
    :param name: String, optional name for Node
    :return: Node representing this sum operation
    '''

    arr = const_node(arr)
    op_res = np.sum(arr, axis=axis, keepdims=keepdims)

    return Node.create_op_node(op_res, 'sum', arr, name=name)


def mean(arr, axis=None, name=None):
    '''
    Creates Node representing a mean operation.
    :param arr: Node or nd.array or Number
    :param name: String, optional name for Node
    :return: Node representing this operation
    '''

    arr = const_node(arr)
    op_res = np.mean(arr, axis=axis)

    return Node.create_op_node(op_res, 'mean', arr, name=name)


def exp(arr, name=None):
    '''
    Creates Node representing a exp (raise all values in given array by exp, Euler's number).
    :param arr: Node or nd.array or Number
    :param name: String, optional name for Node
    :return: Node representing this operation
    '''

    arr = const_node(arr)
    op_res = np.exp(arr)

    return Node.create_op_node(op_res, 'exp', arr, name=name)


def log(arr, name=None):
    '''
    Creates Node representing a log.
    :param arr: Node or nd.array or Number
    :param name: String, optional name for Node
    :return: Node representing this operation
    '''

    arr = const_node(arr)
    op_res = np.log(arr)

    return Node.create_op_node(op_res, 'log', arr, name=name)


def dot(arr_a, arr_b, name=None):
    '''
    Creates Node representing a dot product.
    :param arr_a: Node or nd.array or Number
    :param arr_b: Node or nd.array or Number
    :param name: String, optional name for Node
    :return: Node representing this operation
    '''

    arr_a = const_node(arr_a)
    arr_b = const_node(arr_b)
    op_res = np.dot(arr_a, arr_b)

    return Node.create_op_node(op_res, 'dot', arr_a, arr_b, name=name)


def where(cond, arr_a, arr_b, name=None):
    '''
    Creates Node representing a where-clause filtering operation.
    :param cond: ndarray of Boolean representing filtering conditions
    :param arr_a: Node or nd.array or Number
    :param arr_b: Node or nd.array or Number
    :param name: String, optional name for Node
    :return: Node representing this operation
    '''

    if not isinstance(arr_a, Node):
        nd_arr_a = np.full_like(cond, arr_a)  # Match dimensions
        arr_a = const_node(nd_arr_a)
    if not isinstance(arr_b, Node):
        nd_arr_b = np.full_like(cond, arr_b)  # Match dimensions
        arr_b = const_node(nd_arr_b)

    op_res = np.where(cond, arr_a, arr_b)
    op_node = Node.create_op_node(op_res, 'where', arr_a, arr_b, name=name)

    # Store condition for grad computation
    op_node.cond = cond

    return op_node

def reshape(arr, new_shape, name=None):
    '''
    Creates Node representing reshape operation.
    :param arr: Node or nd.array or Number
    :param new_shape: Tuple or List, new shape for array
    :param name: String, optional name for Node
    :return: Node representing this operation
    '''

    arr = const_node(arr)
    op_res = np.reshape(arr, new_shape)

    return Node.create_op_node(op_res, 'reshape', arr, name=name)


def relu(arr, name=None):
    '''
    Creates Node representing ReLU operation.
    :param arr: Node or nd.array or Number
    :param name: String, optional name for Node
    :return: Node representing this operation
    '''

    arr = const_node(arr)
    arr_b = const_node(np.zeros(shape=arr.shape))
    cond = arr > 0
    op_res = np.where(cond, arr, arr_b)

    op_node = Node.create_op_node(op_res, 'relu', arr, arr_b, name=name)
    op_node.cond = cond

    return op_node


#TODO: Remove, just for gradient checking right now
def add(arr1, arr2):
    return arr1 + arr2

def sub(arr1, arr2):
    return arr1 - arr2

def mul(arr1, arr2):
    return arr1 * arr2

def div(arr1, arr2):
    return arr1 / arr2

def pow(arr1, arr2):
    return arr1 ** arr2