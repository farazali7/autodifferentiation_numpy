from collections import defaultdict
import numpy as np
import auto_diff.derivatives as derivs
from auto_diff.utilities import correct_deriv
from auto_diff.comp_graph import Node, NodeQueue


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

        curr_grads[curr_node.operand_a.name] = correct_deriv(curr_node.operand_a,
                                                             curr_grads[curr_node.operand_a.name] + next_grads[0])

        if curr_node.operand_a not in queue:
            queue.push(curr_node.operand_a)

        if curr_node.operand_b is not None:
            curr_grads[curr_node.operand_b.name] = correct_deriv(curr_node.operand_b,
                                                                 curr_grads[curr_node.operand_b.name]
                                                                 + next_grads[1])
            if curr_node.operand_b not in queue:
                queue.push(curr_node.operand_b)

    return all_grads



