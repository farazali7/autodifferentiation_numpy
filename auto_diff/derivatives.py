import auto_diff.comp_graph as cg
import numpy as np

'''
This file contains the derivatives for common binary and unary arithmetic operations and ndarray operations.
In general, derivatives for function f with operands a and b are expressed as [df/d_operand_a, df/d_operand_b].
'''


def add_deriv(previous_deriv, node):
    return [previous_deriv, previous_deriv]


def sub_deriv(previous_deriv, node):
    return [previous_deriv, -1*previous_deriv]


def mul_deriv(previous_deriv, node):
    return [previous_deriv * node.operand_b,
            previous_deriv * node.operand_a]


def div_deriv(previous_deriv, node):
    return [previous_deriv / node.operand_b,
            -1 * previous_deriv * (node.operand_a / node.operand_b**2)]


def pow_deriv(previous_deriv, node):
    return [previous_deriv * node.operand_b * (node.operand_a ** (node.operand_b - 1)),
            previous_deriv * node * cg.log(node.operand_a)]


def transpose_deriv(previous_deriv, node):
    return [previous_deriv.T, None]


def sum_deriv(previous_deriv, node):
    return [previous_deriv * np.ones_like(node.operand_a), None]


def mean_deriv(previous_deriv, node):
    return [previous_deriv * np.ones_like(node.operand_a) / node.operand_a.shape[0], None]


def exp_deriv(previous_deriv, node):
    return [previous_deriv * node, None]


def log_deriv(previous_deriv, node):
    return [previous_deriv * (1. / node.operand_a), None]


def dot_deriv(previous_deriv, node):
    prev_deriv = previous_deriv  # Placeholder
    operand_a = node.operand_a
    operand_b = node.operand_b

    # Dot operation needs 2D arrays
    if previous_deriv.ndim == 1:
        prev_deriv = cg.reshape(prev_deriv, (1, -1))

    if node.operand_a.ndim == 1:
        operand_a = cg.reshape(operand_a, (1, -1))

    if node.operand_b.ndim == 1:
        operand_b = cg.reshape(operand_b, (-1, 1))

    # Dot product version of mul_deriv
    return [cg.dot(prev_deriv, operand_b.T), cg.dot(operand_a.T, prev_deriv)]


def where_deriv(previous_deriv, node):
    op_a_cond_idxs = np.zeros_like(node.operand_a)  # Operand a was True
    op_b_cond_idxs = np.ones_like(node.operand_b)  # Operand b was False

    op_a_cond_idxs[node.cond] = 1
    op_b_cond_idxs[node.cond] = 0

    return [previous_deriv * op_a_cond_idxs, previous_deriv * op_b_cond_idxs]


def relu_deriv(previous_deriv, node):
    return where_deriv(previous_deriv, node)  # ReLU was implemented via 'where' fn


def reshape_deriv(previous_deriv, node):
    return [cg.reshape(previous_deriv, node.operand_a.shape), None]

