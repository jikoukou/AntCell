from Node import Node
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Add, ZeroPadding2D, Add, Layer
from tensorflow.keras.layers import Concatenate, Flatten, Dense, concatenate, Input
from tensorflow.keras.layers import BatchNormalization, Activation, DepthwiseConv2D, SeparableConv2D


class Graph:
    # Creates the graph for the ants.
    def __init__(self, available_CNN, available_Pool, maxBlocks, one_input_bef, two_inputs_bef, pherormone):
        self.allNodes = []

        for i in range(maxBlocks):
            newNode = Node(Add(), [], [], [], i, 2, 0)
            self.allNodes.append(newNode)
        nodes_only_add = self.allNodes.copy()

        for i in range(len(available_CNN)):

            newNode = Node(available_CNN[i], [], [], [], i+maxBlocks)

            for j in nodes_only_add:
                j.addNeighbor(newNode, pherormone, 1)
                newNode.addNeighbor(j, pherormone, 1)
            self.allNodes.append(newNode)

        for k in range(len(available_Pool)):
            newNode = Node(available_Pool[k], [], [], [], k+i+maxBlocks)

            for j in nodes_only_add:
                j.addNeighbor(newNode, pherormone, 1)
                newNode.addNeighbor(j, pherormone, 1)
            self.allNodes.append(newNode)

        for i in range(len(self.allNodes)):
            self.allNodes[i].addNeighbor(one_input_bef, pherormone, 1)
            one_input_bef.addNeighbor(self.allNodes[i], pherormone, 1)
            self.allNodes[i].addNeighbor(two_inputs_bef, pherormone, 1)
            two_inputs_bef.addNeighbor(self.allNodes[i], pherormone, 1)

        self.allNodes.append(one_input_bef)
        self.allNodes.append(two_inputs_bef)

        startNode = Node(None, [], [], [], -3, 0, 0)
        one_input_bef.addNeighbor(startNode, pherormone, 1)
        startNode.addNeighbor(one_input_bef, pherormone, 1)

        two_inputs_bef.addNeighbor(startNode, pherormone, 1)
        startNode.addNeighbor(two_inputs_bef, pherormone, 1)

        self.allNodes.append(startNode)

        del nodes_only_add[:]

    def port_recovery(self):
        for i in self.allNodes:
            i.local_inputs = 0
            i.outputs = 0
