import numpy as np
import uuid


class Node(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, node_type='data', name=None):
        '''
        Create a new Node object that acts as a wrapper around a NumPy array.
        :param shape: List of int representing shape of Node
        :param dtype: np.dtype representing data type of Node value
        :param buffer: None or np.ndarray, contains initialization data
        :param node_type: String, one of {'data' for data/vars, 'const' for constants, 'op' for operations}
        :return: new Node object
        '''
        node_obj = np.ndarray.__new__(cls, shape=shape, dtype=dtype, buffer=buffer)
        node_obj.node_type = node_type
        name = name if name is not None else str(uuid.uuid4().hex)
        node_obj.name = name

        return node_obj

    @staticmethod
    def create_op_node(op_res, op_name, operand_a, operand_b=None, name=None):
        '''
        Create a computation graph node for an operation
        :param op_res: np.ndarray, result of the operation
        :param op_name: String, name of operation
        :param operand_a: Node, first operand to operation
        :param operand_b: Node, second operand to operation
        :param name: String, optional name of node, otherwise uuid given as name
        :return: Node containing pertinent operation information
        '''

        name = name if name is not None else f"{op_name}_{str(uuid.uuid4().hex)}"
        op_node = Node(shape=op_res.shape, dtype=op_res.dtype, buffer=np.copy(op_res), node_type='op', name=name)
        op_node.op_name = op_name
        op_node.operand_a = operand_a
        op_node.operand_b = operand_b

        return op_node

    @staticmethod
    def create_data_node(data, name=None):
        '''
        Creates a computation graph node for a variable.
        :param data: np.ndarray or Number
        :return: Node containing pertinent variable information
        '''
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)

        name = name if name is not None else f"data_{str(uuid.uuid4().hex)}"
        return Node(shape=data.shape, dtype=data.dtype, buffer=data, node_type='data', name=name)

    @staticmethod
    def create_const_node(data, name=None):
        '''
        Creates a computation graph node for a constant.
        :param data: np.ndarray or Number
        :return: Node containing pertinent constant information
        '''
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)

        name = name if name is not None else f"const_{str(uuid.uuid4().hex)}"
        return Node(shape=data.shape, dtype=data.dtype, buffer=data, node_type='const', name=name)

    def _override_arith_ops(self, method_name, other_operand, op_name, is_this_node_first=True):
        '''
        Override a given array arithmetic method by returning a new OpNode with the result.
        :param method_name: String, name of arithmetic method
        :param other_operand: Node or nd.array or Number, the second operand in operation
        :param op_name: Name of the new OpNode
        :param is_this_node_first: Boolean, indicates ordering: if True, this Node is first operand, else second
        :return: OpNode with result of arithmetic method
        '''

        # Handle non-node objects
        if not isinstance(other_operand, Node):
            other_operand = Node.create_const_node(other_operand)

        op_value = getattr(np.ndarray, method_name)(self, other_operand)

        return Node.create_op_node(op_value, op_name,
                                   self if is_this_node_first else other_operand,
                                   other_operand if is_this_node_first else self)

    # Override basic array arithmetic ops from np.ndarray (add, subtract, multiply, divide)
    def __add__(self, other_operand):
        return self._override_arith_ops('__add__', other_operand, 'add')

    def __radd__(self, other_operand):
        return self._override_arith_ops('__radd__', other_operand, 'add')

    def __sub__(self, other_operand):
        return self._override_arith_ops('__sub__', other_operand, 'sub')

    def __rsub__(self, other_operand):
        return self._override_arith_ops('__rsub__', other_operand, 'sub', False)

    def __mul__(self, other_operand):
        return self._override_arith_ops('__mul__', other_operand, 'mul')

    def __rmul__(self, other_operand):
        return self._override_arith_ops('__rmul__', other_operand, 'mul')

    def __div__(self, other_operand):
        return self._override_arith_ops('__div__', other_operand, 'div')

    def __rdiv__(self, other_operand):
        return self._override_arith_ops('__rdiv__', other_operand, 'div', False)

    def __truediv__(self, other_operand):
        return self._override_arith_ops('__truediv__', other_operand, 'div')

    def __rtruediv__(self, other_operand):
        return self._override_arith_ops('__rtruediv__', other_operand, 'div', False)

    def __pow__(self, other_operand):
        return self._override_arith_ops('__pow__', other_operand, 'pow')

    def __rpow__(self, other_operand):
        return self._override_arith_ops('__rpow__', other_operand, 'pow', False)

    @property
    def T(self):
        '''
        Override transpose property on normal np.ndarray by returning OpNode capturing this op
        :return: OpNode holding transposed node data
        '''
        return Node.create_op_node(np.transpose(self), 'transpose', self)
