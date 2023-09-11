from preprocess import get_channel_axis
import sys
from Ant import Ant
from Node import Node
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Add, ZeroPadding2D, Add, Layer
from tensorflow.keras.layers import Concatenate, Flatten, Dense, concatenate, Input
from tensorflow.keras.layers import BatchNormalization, Activation, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.metrics import categorical_accuracy
import random, math

initial_dropout = 0.015

class ACO:
    def __init__(self, maxBlocks, graph, ant_walk = [], evaporation = 0.2, \
                 max_greediness=0.8, min_greediness = 0.1, step=initial_dropout):
        self.maxBlocks = maxBlocks
        self.graph = graph
        self.ant_walk = ant_walk
        self.max_greediness = max_greediness
        self.min_greediness = min_greediness
        self.greediness = self.min_greediness
        self.ant = Ant(self.graph.allNodes[len(self.graph.allNodes) - 1], [], self.greediness)
        self.evaporation = evaporation
        
        # Used in ScheduledDropPath
        self.dropout = initial_dropout
        self.step = step
        
        # A list composed of two other lists
        # The first saves the number of the used layer
        # The second stores the layer itself
        self.layers = [[], []]
        
    
    def copy(self):
        
        
        aco = ACO(self.maxBlocks, self.graph, self.ant_walk, self.evaporation, self.max_greediness, \
                  self.min_greediness, self.step)
        
        aco.dropout = self.dropout
        aco.layers = self.layers
        return aco
    
    def global_update(self, cgb):
        
        # A list made of lists, which stores all the available nodes
        all_transitions = []
        
        for i in range(len(self.graph.allNodes)):
            all_transitions.append([])
        
        # Position 0: Start node
        # Position 1: Layer 1 ...
        
        for i in self.ant_walk:
            for j in range(len(i)-1):
                all_transitions[i[j].num + 3].append(i[j+1].num)
            if(i[len(i) - 1].num == -1 or i[len(i) - 1].num == -2):
                for j in range(len(i[len(i) - 1].neighbors)):
                    if(i[len(i) - 1].neighbors[j].num == -3):
                        all_transitions[i[len(i) - 1].num + 3].append(i[len(i) - 1].neighbors[j].num)

                        

        for i in self.graph.allNodes:
            for j in range(len(i.neighbors)):
                
                if(i.neighbors[j].num in all_transitions[i.num + 3]):
                    i.updatePheromone(i.neighbors[j], 'global', None, self.evaporation, cgb)
                    
                else:
                    # If the node is in the 
                    if(i.num in all_transitions[i.neighbors[j].num + 3]):
                        continue
                        
                    i.updatePheromone(i.neighbors[j], 'global', None, self.evaporation, 0)
                
    def reset_aco(self, new_best = True, step = 0.1):
        
        self.dropout = initial_dropout
        
        if(new_best):
            self.greediness -= step
        else:
            self.greediness += step
            
        if(self.greediness > self.max_greediness):
            self.greediness = self.max_greediness

        if(self.greediness < self.min_greediness):
            self.greediness = self.min_greediness
            
        self.ant_walk = []
        
        for i in self.ant.visited:
            del i
        
        
        del self.ant.currentNode
        self.ant.set_pos(self.graph.allNodes[len(self.graph.allNodes) - 1])
        self.ant.update_greediness(self.greediness)
        self.dropout = initial_dropout
        
    def create_graph(self):
        # First node visited is none
        iteration = 0
        add_ops = 0
        
        
        while(add_ops < self.maxBlocks):
            # Ant saves the points it reached, in order other ants not to do the same

            ant_points = []
            found = 0
            flag = True
            
            while True:
                
                
                ant_points.append(self.ant.currentNode)
                flag = self.ant.choose_node(self.ant_walk)
                
                # Ant is trapped in a node. Break
                if(not flag):
                    break
                
                if(self.ant.currentNode.num < 0 and found == 1):
                    ant_points.append(self.ant.currentNode)
                    break

                elif(found == 0):
                    if(self.ant.currentNode.layer.__class__.__name__ == 'Layer'):
                        found = 1
                    
            ant_was_correct = False
            for i in ant_points:
                if(i.num < self.maxBlocks and i.num >= 0):
                    self.ant_walk.append(ant_points)
                    ant_was_correct = True
                    break
            if(not ant_was_correct):
                for i in ant_points:
                    if(i.num > 0):
                        i.local_inputs -= 1
                        break
            
#             del ant_points
            # Ant starts on new position
            for i in self.ant.visited:
                del i
                
            del self.ant.currentNode
            self.ant.set_pos(self.graph.allNodes[len(self.graph.allNodes) - 1])

            add_ops = 0
            add = []
            # Detect which add nodes are completed
            for i in self.ant_walk:
                for j in i:
                    if(j.num >= 0 and j.num <= self.maxBlocks):
                        add.append(j)

            for i in list(set(add)):
                if(i.local_inputs == 2):
                    add_ops += 1
                elif(i.local_inputs == 3):
                    # Delete from ant_points the remaining elements
                    
                    add_ops = add_ops + 1
                    
                    j = len(ant_points) - 1
                    
                    while (ant_points[j] != i):
                        del ant_points[j]
#                         ant_points.pop(j)
                        j = len(ant_points) - 1
                    ant_points[j].local_inputs = 2
        
#         for i in self.ant_walk:
#             for j in i:

#                 if(j.layer != None):
                    
#                     if(j.layer.__class__.__name__== 'Conv2D' or \
#                     j.layer.__class__.__name__ == 'DepthwiseConv2D' or \
#                     j.layer.__class__.__name__ == 'SeparableConv2D'):
#                         dropout_prob = 1.0
#                         choice = random.choices([0, 1], weights=[1-dropout_prob, dropout_prob], k=1)
                        
#                         if(choice[0] == 1):
#                             j.has_batch = True
                        
    
    def print_graph(self):
        for i in self.ant_walk:
            for j in i:
                print(j.layer.__class__.__name__, end = "")

                if(j.layer.__class__.__name__ == 'Add'):
                    print('(', j.num,')', end = "->")
                else:
                    if(j.layer.__class__.__name__ == 'Conv2D' or \
                      j.layer.__class__.__name__ == 'SeparableConv2D'):
                        print(j.layer.kernel_size, j.layer.filters, end = "")
                    elif(j.layer.__class__.__name__ == 'MaxPooling2D' or \
                        j.layer.__class__.__name__ == 'AveragePooling2D' ):
                        print(j.layer.strides, end="")
                        
                    elif(j.layer.__class__.__name__ == 'DepthwiseConv2D'):
                        print(j.layer.kernel_size, end = "")
                        
                    else:
                        print('(', j.num, ')', end = "")
                    print('->', end = "")
            print('NoneType( -3 )\n\n')
                
    
    def generate_cell(self, inp, local, add_batch):
#         inp1 = BatchNormalization(axis=get_channel_axis())(inp1)
#         inp2 = BatchNormalization(axis=get_channel_axis())(inp2)
        # List with size maxBlocks. Stores in pos i True if the final output will be concatenated
#         concat_outputs = []
#         self.layers = [[], []]

        towers = self.towers
        outputs = []
        
        for i in range(self.maxBlocks):
#             towers.append([])
            outputs.append(None)
#             concat_outputs.append(False)
    
                                
        counter = 0
        while(counter < self.maxBlocks):
            # Counts how many outputs were given 
            
            for i in range(self.maxBlocks):
                if(len(towers[i]) == 2 and outputs[i] == None):

                    # ScheduledDropPath
                    
                    first_prob = random.choices([0, 1], weights=[1-self.dropout, self.dropout], k=1)
                    second_prob = random.choices([0, 1], weights=[1-self.dropout, self.dropout], k=1)
                    
                    if(first_prob[0] == 1 and second_prob[0] == 1):
                        rnd = random.randint(1, 2)
                        
                        if(rnd == 1):
                            second_prob[0] = sys.maxsize
                            
                        else:
                            first_prob[0] = sys.maxsize
                    
                    
                    if(towers[i][0].shape[3] > towers[i][1].shape[3]):
                        convolutional = Conv2D(towers[i][1].shape[3], (1, 1), padding='same')
                        self.layers[0].append(convolutional)
                        
#                         batch_norm = BatchNormalization()
#                         self.layers[0].append(batch_norm)
                        
                        towers[i][0] = ((convolutional)(towers[i][0]))
                    if(towers[i][0].shape[3] < towers[i][1].shape[3]):
                        convolutional = Conv2D(towers[i][0].shape[3], (1, 1), padding='same')
                        self.layers[0].append(convolutional)
                        
#                         batch_norm = BatchNormalization()
#                         self.layers[0].append(batch_norm)
                        towers[i][1] = ((convolutional)(towers[i][1]))

                    self.pad_inputs(towers[i])
                    
                    if(first_prob[0] == 1):
                
                        outputs[i] = (towers[i][1])
                            
                    else:
                        if(second_prob[0] == 1):
                            outputs[i] = (towers[i][0])
                            
                        else:
                            outputs[i] = Add()(towers[i])
                            
                    counter += 1
                    
                # Find the int and create the correct new layer
                else:
                    for j in range(len(towers[i])):
                        if(isinstance(towers[i][j], int)):
                            if(outputs[towers[i][j]] != None):
                                
                                layer = towers[i][j+1].create_layer()
                                self.layers[0].append(layer)
                                self.layers[1].append(towers[i][j+1].num)
                                
                                for f in self.ant_walk:
                                    for l in range(len(f)):
                                        if(f[l].num == towers[i][j+1].num):
                                            
                                            has_batch = f[l].has_batch
                                    
                                
                                if(has_batch):
                                    batch_norm = BatchNormalization()
                                    self.layers[0].append(batch_norm)
                                    towers[i].append((batch_norm)\
                                                     (layer(outputs[towers[i][j]])))
                                    
                                else:
                                    towers[i].append(layer(outputs[towers[i][j]]))
                                del towers[i][j]
                                del towers[i][j]
                                break
            
        # Final outputs-> These result of the add operations are going to be concatenated
        final_outputs = []
        for i in range(self.maxBlocks):  
            
            if(self.concat_outputs[i] == True):
                
#                 rnd = random.random()
#                 if(rnd < self.dropout):
#                     if(i == self.maxBlocks-1 and len(final_outputs) == 0):
#                         final_outputs.append(outputs[i])
#                     else:
#                         final_outputs.append(tf.keras.backend.zeros_like(outputs[i]))
#                 else:
                final_outputs.append(outputs[i])
        
        
        if(len(final_outputs) == 1):
            return (final_outputs[0])
        else:
            self.pad_inputs(final_outputs)
            concat = concatenate(final_outputs, axis=get_channel_axis())
            
            return (concat)
            
#             batch_norm = BatchNormalization()
#             x = (batch_norm)(concat)
            
#             self.layers[0].append(batch_norm)
            
#             if(add_batch):
#                 return x
            
#             else:
#                 return concat
#         return final_outputs
                                
    def pad_inputs(self, towers):
        max_0 = 0
        
        max_1 = 0
        max_2 = 0
        
        
        for j in towers:

            if(j.shape[1] > max_1):
                max_1 = j.shape[1]

            if(j.shape[2] > max_2):
                max_2 = j.shape[2]
        
        
        for j in range(len(towers)):
            
            
            added_1 = 0
            added_2 = 0

            if(towers[j].shape[1] < max_1):
                added_1 = math.floor(((max_1 - towers[j].shape[1]))/2)

            if(towers[j].shape[2] < max_2):
                added_2 = math.floor(((max_2 - towers[j].shape[2]))/2)
            
            if(towers[j].shape[1] < max_1 or towers[j].shape[2] < max_2):
                towers[j] = ZeroPadding2D(padding = ((added_1, abs(max_1 - towers[j].shape[1] - added_1)), \
                                                     (added_2, abs(max_2 - towers[j].shape[2] - added_2))))\
                (towers[j])
    
    def initialize_layers(self, inp1, inp2, is_reduction = False):
        concat_outputs = []
        self.layers = [[], []]
        towers = []
        
        for i in range(self.maxBlocks):
            towers.append([])
            concat_outputs.append(False)
        
        for i in self.ant_walk:
            for j in range(len(i)):
                
                # Append to list the layers that can be calculated immediately
                if(i[j].num >= 0 and i[j].num < self.maxBlocks):
                    
                    if(j - 1 >= 0):
                        layer = i[j - 1].create_layer()
                        self.layers[0].append(layer)
                        self.layers[1].append(i[j - 1].num)
                        
                        if(i[j - 1].num == -1):
                            
                            towers[i[j].num].append(layer(inp1))
                            if(is_reduction):
                                if(layer.__class__.__name__== 'Conv2D' or \
                                layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                layer.__class__.__name__ == 'SeparableConv2D'):
                                    layer.strides = (2, 2)

                            
                        elif(i[j - 1].num == -2):
                            towers[i[j].num].append(layer(inp2))
                            if(is_reduction):
                                if(layer.__class__.__name__== 'Conv2D' or \
                                layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                layer.__class__.__name__ == 'SeparableConv2D'):
                                    layer.strides = (2, 2)
                            
                            
                        elif(i[j - 1].num >= 0 and i[j - 1].num < self.maxBlocks): 
                            towers[i[j].num].append(i[j - 1].num)
                            towers[i[j].num].append(None)
                            
                        if(j - 2 >= 0):
                            layer = i[j - 1].create_layer()
                            self.layers[0].append(layer)
                            self.layers[1].append(i[j - 1].num)
                            
                            if(i[j - 2].num == -1):
                                if(is_reduction):
                                    if(layer.__class__.__name__== 'Conv2D' or \
                                    layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                    layer.__class__.__name__ == 'SeparableConv2D'):
                                        layer.strides = (2, 2)
                                if(i[j-1].has_batch):
                                    batch_norm = BatchNormalization()
                                    self.layers[0].append(batch_norm)
                                    towers[i[j].num].append(batch_norm(layer(inp1)))
                                else:
                                    towers[i[j].num].append(layer(inp1))
                            elif(i[j - 2].num == -2):
                    
                                if(is_reduction):
                                    if(layer.__class__.__name__== 'Conv2D' or \
                                    layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                    layer.__class__.__name__ == 'SeparableConv2D'):
                                        layer.strides = (2, 2)
                                if(i[j-1].has_batch):
                                    batch_norm = BatchNormalization()
                                    self.layers[0].append(batch_norm)
                                    towers[i[j].num].append((batch_norm)(layer(inp2)))
                                else:
                                    towers[i[j].num].append(layer(inp2))

                            # Save a tuple (add_op_num, layer) in the list
                            elif(i[j - 2].num >= 0 and i[j - 2].num < self.maxBlocks): 
                                towers[i[j].num].append(i[j - 2].num)
                                towers[i[j].num].append(i[j - 1])
                            
                    if(j + 1 < len(i)):
                        layer = i[j + 1].create_layer()
                        self.layers[0].append(layer)
                        self.layers[1].append(i[j + 1].num)
                        if(i[j + 1].num == -1):
                            concat_outputs[i[j].num] = True
                            
                            if(is_reduction):
                                if(layer.__class__.__name__== 'Conv2D' or \
                                layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                layer.__class__.__name__ == 'SeparableConv2D'):
                                    layer.strides = (2, 2)
                            if(i[j].has_batch):
                                batch_norm = BatchNormalization()
                                self.layers[0].append(batch_norm)
                                towers[i[j].num].append(batch_norm(layer(inp1)))
                            else:
                                towers[i[j].num].append(layer(inp1))
                                
                        elif(i[j + 1].num == -2):
                            concat_outputs[i[j].num] = True
                            if(is_reduction):
                                if(layer.__class__.__name__== 'Conv2D' or \
                                layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                layer.__class__.__name__ == 'SeparableConv2D'):
                                    layer.strides = (2, 2)
                            
                            towers[i[j].num].append(layer(inp2))
    
                        
                        if(j + 2 < len(i)):
                
                            layer = i[j + 1].create_layer()
                            self.layers[0].append(layer)
                            self.layers[1].append(i[j + 1].num)
                            if(i[j + 2].num == -1):
                                concat_outputs[i[j].num] = True
                                
                                if(is_reduction):
                                    if(layer.__class__.__name__== 'Conv2D' or \
                                    layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                    layer.__class__.__name__ == 'SeparableConv2D'):
                                        layer.strides = (2, 2)                           
                                if(i[j+1].has_batch):
                                    batch_norm = BatchNormalization()
                                    self.layers[0].append(batch_norm)
                                    towers[i[j].num].append(batch_norm(layer(inp1)))
                                else:
                                    towers[i[j].num].append(layer(inp1))
                                                        
                            elif(i[j + 2].num == -2):
                        
                                if(is_reduction):
                                    if(layer.__class__.__name__== 'Conv2D' or \
                                    layer.__class__.__name__ == 'DepthwiseConv2D' or \
                                    layer.__class__.__name__ == 'SeparableConv2D'):
                                        layer.strides = (2, 2)                             
                                concat_outputs[i[j].num] = True
                                
                                                        
                                if(i[j+1].has_batch):
                                    batch_norm = BatchNormalization()
                                    self.layers[0].append(batch_norm)
                                    towers[i[j].num].append(batch_norm(layer(inp2)))
                                else:
                                    towers[i[j].num].append(layer(inp2))
                                
        self.towers = towers
        self.concat_outputs = concat_outputs
    
                    
    def set_weights(self, new_layers):
        
        for i in range(len(new_layers)):
            
                
            self.layers[0][i].set_weights(new_layers[i].get_weights())
        
                    
                    
