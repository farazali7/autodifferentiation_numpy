from collections import defaultdict
import numpy as np
import auto_diff.derivatives as derivs
from auto_diff.utilities import correct_deriv, check_gradient
from auto_diff.comp_graph import Node, NodeQueue
import auto_diff.comp_graph as cg


def gradient2(node):
    """
    computes and returns the gradient of the given node wrt to VariableNodes
    the function implements a breadth-first-search (BFS) to traverse the
    computational graph from the gievn node back to VariableNodes
    Parameters:
    ----------
    node: Node
        the node to compute its gradient
    """

    adjoint = defaultdict(int)
    grad = {}
    queue = NodeQueue()

    # put the given node in the queue and set its adjoint to one
    adjoint[node.name] = Node.create_const_node(np.ones(node.shape))
    queue.push(node)

    while len(queue) > 0:
        current_node = queue.pop()

        if current_node.node_type == 'const':
            continue
        if current_node.node_type == 'data':
            grad[current_node.name] = adjoint[current_node.name]
            continue

        current_adjoint = adjoint[current_node.name]
        current_op = current_node.op_name

        op_grad = getattr(derivs, '{}_deriv'.format(current_op))
        next_adjoints = op_grad(current_adjoint, current_node)

        adjoint[current_node.operand_a.name] = correct_deriv(
            current_node.operand_a,
            adjoint[current_node.operand_a.name] + next_adjoints[0]
        )
        if current_node.operand_a not in queue:
            queue.push(current_node.operand_a)

        if current_node.operand_b is not None:
            adjoint[current_node.operand_b.name] = correct_deriv(
                current_node.operand_b,
                adjoint[current_node.operand_b.name] + next_adjoints[1]
            )
            if current_node.operand_b not in queue:
                queue.push(current_node.operand_b)

    return grad


def gradient(node):
    '''
    Compute the gradient of the given node
    :param node: Node, the node to compute gradient of
    :return: Gradient of node w.r.t to data nodes of this node
    '''

    queue = NodeQueue()
    curr_grads = defaultdict(int)  # Create dict for storing computed grads
    all_grads = {}  # All gradients in graph

    # Gradient with respect to node itself is 1
    curr_grads[node.name] = Node.create_const_node(np.ones(node.shape))

    # Push node into queue
    queue.push(node)

    # Perform Breadth-First Search
    while len(queue) > 0:
        curr_node = queue.pop()

        if curr_node.node_type == 'const':
            continue
        if curr_node.node_type == 'data':
            all_grads[curr_node.name] = curr_grads[curr_node.name]
            continue

        curr_grad = curr_grads[curr_node.name]
        curr_op = curr_node.op_name

        op_grad_fn = getattr(derivs, '{}_deriv'.format(curr_op))
        # Note: 'next_' here means backwards through a graph
        next_grads = op_grad_fn(curr_grad, curr_node)

        # Check grad
        # op_fn = getattr(cg, curr_op)
        # ch_grad = next_grads/curr_grad
        # is_grad_correct = check_gradient(op_fn, [curr_node.operand_a, curr_node.operand_b], ch_grad)
        # if not is_grad_correct:
        #     print('STOP')
        #     is_grad_correct = check_gradient(op_fn, [curr_node.operand_a, curr_node.operand_b], ch_grad)
        # print(f'Gradient correct?: {is_grad_correct}')

        curr_grads[curr_node.operand_a.name] = correct_deriv(curr_node.operand_a,
                                                             curr_grads[curr_node.operand_a.name] + next_grads[0])
        # curr_grads[curr_node.operand_a.name] = curr_grads[curr_node.operand_a.name] + next_grads[0]

        if curr_node.operand_a not in queue:
            queue.push(curr_node.operand_a)

        if curr_node.operand_b is not None and curr_node.operand_b.node_type != 'const':
            curr_grads[curr_node.operand_b.name] = correct_deriv(curr_node.operand_b,
                                                                 curr_grads[curr_node.operand_b.name]
                                                                 + next_grads[1])
            # curr_grads[curr_node.operand_b.name] = curr_grads[curr_node.operand_b.name] + next_grads[1]
            if curr_node.operand_b not in queue:
                queue.push(curr_node.operand_b)

    return all_grads, curr_grads



