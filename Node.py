import time
import copy
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Add, ZeroPadding2D, Add, Layer
from tensorflow.keras.layers import Concatenate, Flatten, Dense, concatenate, Input
from tensorflow.keras.layers import BatchNormalization, Activation, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
import math
import random
from tensorflow.keras.datasets import mnist
import sys
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecay


class Node:

    # Layer: Type of keras layer added
    # neighbors: Type list of Nodes -> The node is adjasted to each one of them
    # pherormones: Type list of numbers: The pherormones of each ant
    # heuristic: Type list: Represents the heuritic function

    # special_number: Defines a distinct number of all the layers

    # Max inputs: Specifies how many of them can pass through the node
    # Add layer: 2
    # Conv/Pool: 1

    def __init__(self, layer, neighbors, pherormones, heuristic, special_number, max_inputs=1, local_inputs=0):
        self.layer = layer
        self.neighbors = neighbors
        self.pherormones = pherormones
        self.initial_pheromones = []
        self.heuristic = heuristic
        self.num = special_number
        self.max_inputs = max_inputs
        self.local_inputs = local_inputs
        self.outputs = 0

        if (layer.__class__.__name__ == 'Conv2D' or
           layer.__class__.__name__ == 'DepthwiseConv2D' or
           layer.__class__.__name__ == 'SeparableConv2D'):
            self.has_batch = True
        else:
            self.has_batch = False

    # Updates the pheromone value of the node's ith neighbour

    # rule = Local/Global
    def updatePheromone(self, node, rule, decay_factor, evaporation, cgb):

        for i in range(len(self.neighbors)):
            if (self.neighbors[i].num == node.num):
                if (rule == 'local'):
                    newPheromone = (1 - decay_factor) * self.pherormones[i] + \
                        (decay_factor) * self.initial_pheromones[i]

                else:
                    newPheromone = (1 - evaporation) * self.pherormones[i] + \
                        evaporation * cgb

                self.pherormones[i] = newPheromone
#                 break
                if (newPheromone < 0):
                    newPheromone = 0
                    self.pherormones[i] = newPheromone
#                 break

        if ((cgb != 0 and rule == 'global') or rule == 'local'):
            for i in range(len(node.neighbors)):
                if (node.neighbors[i].num == self.num):
                    node.pherormones[i] = newPheromone
                    break

    # Add Node neighbor and set their path pherormone
    def addNeighbor(self, Node, pathPheromone, pathHeuristic):
        self.initial_pheromones.append(pathPheromone)
        # Add the necessary information
        (self.neighbors).append(Node)
        (self.pherormones).append(pathPheromone)
        (self.heuristic).append(pathHeuristic)

    def PrintNode(self):
        if (self.layer.__class__.__name__ == 'Conv2D'):
            print1 = 'kernel_size'
        elif (self.layer.__class__.__name__ == 'MaxPooling2D'):
            print1 = 'pool_size'
        elif (self.layer.__class__.__name__ == 'AveragePooling2D'):
            print1 = 'pool_size'
        elif (self.layer.__class__.__name__ == 'Add' or self.num < 0):
            print1 = ''

        print(self.layer.__class__.__name__, print1)

    def increase_inputs(self):
        self.local_inputs += 1
        self.outputs = 1

    def basicPrint(self, myType, neighborType, neighborConf, selfConf, print1, print2, pherormones, neighbourNum):

        if (print1 == ''):
            print(myType, end=" ")

            if (self.num >= 0):
                print(self.num, '--', end=" ")
            else:
                if (self.num == -1):
                    print('0', '--', end=" ")
                else:
                    print('1', '--', end=" ")
        else:
            print(myType, selfConf[print1], '--', end=" ")

        if (print2 == ''):
            if (neighbourNum >= 0):
                printedNum = neighbourNum
            else:
                if (neighbourNum == -1):
                    printedNum = 0
                else:
                    printedNum = 1

            print(neighborType, printedNum, '(', pherormones, ')')
        else:
            print(neighborType, neighborConf[print2], ' (', pherormones, ')')

    # Prints the connection as self.layer -- neighbor layer (pherormone)
    def printNeighbors(self):

        if (self.layer == None):
            for i in self.neighbors:
                print('Starting Node --', i.layer)
            return

        selfConf = self.layer.get_config()

        for i in self.neighbors:
            if (i.layer == None):
                print(self.layer, '-- Starting Node')
                continue

            neighborConf = i.layer.get_config()
            print1 = ''
            print2 = ''
            if (self.layer.__class__.__name__ == 'Conv2D'):
                print1 = 'kernel_size'
            elif (self.layer.__class__.__name__ == 'MaxPooling2D'):
                print1 = 'pool_size'
            elif (self.layer.__class__.__name__ == 'AveragePooling2D'):
                print1 = 'pool_size'
            elif (self.layer.__class__.__name__ == 'Add' or self.num < 0):
                print1 = ''

            if (i.layer.__class__.__name__ == 'Conv2D'):
                print2 = 'kernel_size'
            elif (i.layer.__class__.__name__ == 'MaxPooling2D'):
                print2 = 'pool_size'
            elif (i.layer.__class__.__name__ == 'AveragePooling2D'):
                print2 = 'pool_size'
            elif (i.layer.__class__.__name__ == 'Add' or i.num < 0):
                print2 = ''

            self.basicPrint(self.layer.__class__.__name__, i.layer.__class__.__name__, neighborConf, selfConf,
                            print1, print2, self.pherormones[self.neighbors.index(i)], i.num)

    # Gets a list of all the neighbor nodes with the path pherormones and heuristic values, but not those already visited
    def available_nodes(self, visited):

        pherormones = []
        nodes = []
        heuristics = []

        for i in self.neighbors:
            node_found = False
            for j in visited:
                if (j.num == i.num):
                    node_found = True

            if (not node_found):
                nodes.append(i)

                path_pherormone = 0
                for j in i.neighbors:
                    if (j.num == self.num):
                        pherormones.append(j.pherormones[path_pherormone])
                        heuristics.append(j.heuristic[path_pherormone])
                        path_pherormone = path_pherormone + 1

        return nodes, pherormones, heuristics

    def create_layer(self):

        if (self.num < 0):
            return tf.keras.layers.Layer()

        config = self.layer.get_config()
        if self.layer.__class__.__name__ == 'Conv2D':
            return tf.keras.layers.Conv2D(filters=config['filters'], kernel_size=config['kernel_size'],
                                          padding=config['padding'], activation='relu')

        elif self.layer.__class__.__name__ == 'MaxPooling2D':
            return tf.keras.layers.MaxPool2D(strides=config['strides'], padding=config['padding'])

        elif self.layer.__class__.__name__ == 'AveragePooling2D':
            return AveragePooling2D(strides=config['strides'], padding=config['padding'])

        elif self.layer.__class__.__name__ == 'DepthwiseConv2D':
            return tf.keras.layers.DepthwiseConv2D(strides=config['strides'],
                                                   kernel_size=config['kernel_size'], padding=config['padding'],
                                                   activation='relu')

        elif self.layer.__class__.__name__ == 'SeparableConv2D':
            return tf.keras.layers.SeparableConv2D(filters=config['filters'],
                                                   strides=config['strides'],
                                                   kernel_size=config['kernel_size'],
                                                   padding=config['padding'], activation='relu')
