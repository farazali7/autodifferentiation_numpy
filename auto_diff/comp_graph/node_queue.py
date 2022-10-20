from collections import deque


class NodeQueue:
    def __init__(self):
        '''
        Create a queue data structure containing Node objects.
        '''
        self.nodes = deque()
        self.node_names = deque()

    def push(self, node):
        '''
        Push given node into queue.
        :param node: Node, new node to insert
        '''
        self.nodes.append(node)
        self.node_names.append(node.name)

    def pop(self):
        '''
        Pop oldest node from queue.
        :return: Popped Node object
        '''
        node = self.nodes.popleft()
        self.node_names.popleft()
        return node

    def __len__(self):
        '''
        Return length of current node queue
        :return: Int, length of current queue
        '''
        return len(self.nodes)

    def __contains__(self, node):
        '''
        Check if this queue contains a node (useful when doing BFS for gradient computation).
        :param node: Node, node object to check for
        :return: Boolean, indicating whether node is in this queue already
        '''
        return node.name in self.node_names
