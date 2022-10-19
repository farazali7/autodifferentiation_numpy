from collections import deque


class NodeQueue:
    def __init__(self):
        '''
        Create a queue data structure containing Node objects.
        '''
        self.nodes = deque()

    def push(self, node):
        '''
        Push given node into queue.
        :param node: Node, new node to insert
        '''
        self.nodes.append(node)

    def pop(self):
        '''
        Pop oldest node from queue.
        :return: Popped Node object
        '''
        node = self.nodes.popleft()
        return node

    def __len__(self):
        '''
        Return length of current node queue
        :return: Int, length of current queue
        '''
        return len(self.nodes)
