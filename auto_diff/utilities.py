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
            sum_dims = tuple(range(diff_ndim))
            corrected_deriv = cg.sum(previous_deriv, axis=sum_dims)

            axes_with_ones = tuple([axis for axis, size in enumerate(node.shape) if size == 1])
            if len(axes_with_ones) != 0:
                corrected_deriv = cg.sum(corrected_deriv, axis=axes_with_ones, keepdims=True)

    return corrected_deriv
