import numpy as np
import auto_diff.comp_graph as cg


def correct_deriv(node, previous_deriv):
    '''
    This utility fn enables shape correction of a previously computed derivative tensor that has incompatible
    shape with the current node. Typically, this occurs at nodes where a bias term with size d is added to a
    dot product of weights and inputs with size nxd, since the dot product also has derivatives with size nxd.
    :param node: Node, node used to check if previous derivative requires correction
    :param previous_deriv: ndarray, previously computed derivatives feeding into the given 'node'
    :return: Unbroadcasted previous derivative array
    '''
    corrected_deriv = previous_deriv

    if node.shape != previous_deriv.shape:
        diff_ndim = np.abs(node.ndim - previous_deriv.ndim)
        if diff_ndim != 0:
            # print(node.node_type)
            # if node.node_type == 'op':
            #     print(node.op_name)
            # if node.node_type == 'const':
            #     print(node)
            # print(node.name)
            sum_dims = tuple(range(diff_ndim))
            corrected_deriv = cg.sum(previous_deriv, axis=sum_dims)

            axes_with_ones = tuple([axis for axis, size in enumerate(node.shape) if size == 1])
            if len(axes_with_ones) != 0:
                corrected_deriv = cg.sum(corrected_deriv, axis=axes_with_ones, keepdims=True)

    return corrected_deriv


# TODO: Change
def check_gradient(fx, args, suspect):
    """
    checks the correctness of the suspect derivative value against
    the value of the numerical approximation of the derivative
    Parameters:
    ----------
    fx: callable
        The function to check its derivative
    wrt: int
        0-based index of the variable to differntiate with respect to
    args: list
        the values of the function variables at the derivative point
    suspect: float
        the the suspected value of the derivative to check
    """
    h = 1.e-7
    approx_grad = []

    for i in range(len(args)):
        shifted_args = args[:]
        shifted_args[i] = shifted_args[i] + h
        approx_grad.append((fx(*shifted_args) - fx(*args)) / h)

    return np.allclose(approx_grad, suspect)
