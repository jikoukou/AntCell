from Node import Node
import random

# Class responsible for each ant


class Ant:
    def __init__(self, currentNode, visited, greediness, decay_factor=0.1):
        self.currentNode = currentNode
        self.visited = visited
        self.decay_factor = decay_factor
        self.greediness = greediness

    def update_greediness(self, new_greediness):
        self.greediness = new_greediness

    def local_update(self, node, newNode):
        node.updatePheromone(newNode, 'local', self.decay_factor, None, None)

    def visit(self, newNode):

        if ((newNode.num == -1 or newNode.num == -2) and len(self.visited) >= 2):
            if (self.visited[len(self.visited) - 2].layer.__class__.__name__ == 'Add'):
                self.visited[len(self.visited) - 2].increase_inputs()

        self.visited.append(newNode)

        # Local update
        self.local_update(self.currentNode, newNode)

        newNode.increase_inputs()
        self.currentNode = newNode

    # Set current node for the ant. Deletes all the visited nodes
    def set_pos(self, newPos):
        del self.visited[:]
        del self.visited
        self.visited = []
        self.currentNode = newPos

    # Remove a visited node. Used when a node can be revisited
    def remove_visited(self, node):
        for i in self.visited:
            if (i.num == node.num):
                del self.visited[i]
#                 self.visited.remove(i)

    # Chooses next node based on the rules of Ant Colony System

    def choose_node(self, ant_walk):
        # All probabillities of node
        probabilities = []
        denominator = 0.0

        first = False
        available = []
        pathCounter = 0

        for i in self.currentNode.neighbors:
            isAvailable = True

            if (i.layer == None):
                continue

            # Cannot access this specific node
            if (i.local_inputs >= i.max_inputs and i.num >= 0):
                continue

            for j in self.visited:
                if (j.num == i.num):
                    isAvailable = False
                    break

#             if(i.layer.__class__.__name__ == 'MaxPooling2D' or\
#                i.layer.__class__.__name__ == 'AveragePooling2D'):
#                 if(self.currentNode.layer.__class__.__name__ == 'Add' and self.currentNode.local_inputs > 1):
#                     isAvailable = True
#                 else:
#                     isAvailable = False

            # Check paths of previous ants

            for j in range(len(ant_walk)):

                for k in range(len(ant_walk[j])-1):

                    if (i.layer.__class__.__name__ == 'Add'):
                        if (ant_walk[j][k].num == i.num):

                            # We can travel only in the first node of an add row

                            if (i.local_inputs < 2):
                                if (not first):
                                    #                                     first = True
                                    if (self.visited[len(self.visited) - 1].num != -1 and
                                       self.visited[len(self.visited) - 1].num != -2):
                                        isAvailable = False
                                    else:
                                        first = True
                                else:
                                    isAvailable = False

                    if (self.currentNode.layer.__class__.__name__ == 'Add'):
                        prev = k - 1
                        # Check if there is a connection between multiple adds. In this case the ant
                        # cannot travel there
                        while (prev >= 0):
                            if (ant_walk[j][k].layer.__class__.__name__ == 'Add'):
                                if (i.num == ant_walk[j][k].num):
                                    isAvailable = False
                            prev -= 1

                    if (ant_walk[j][k].num == self.currentNode.num):
                        if (i.num == ant_walk[j][k+1]):
                            isAvailable = False
                            break
                        else:
                            if (k != 0 and ant_walk[j][k - 1].num == i.num):
                                isAvailable = False
                                break

            if (i.num == -1 or i.num == -2):
                if (len(self.visited) > 0 and
                   self.visited[len(self.visited) - 1].layer.__class__.__name__ == 'Add'):
                    isAvailable = False
                else:
                    isAvailable = True

            if (isAvailable):
                available.append(i)
                probabilities.append(
                    self.currentNode.pherormones[pathCounter]*self.currentNode.heuristic[pathCounter])
                denominator = denominator + \
                    probabilities[len(probabilities) - 1]
            pathCounter = pathCounter + 1

        # Choose at random: Exploitation vs exploration
        random_variable = random.uniform(0, 1)

        if (len(available) == 0):
            del available
            return False

        # Code from deepswarm
        if (random_variable <= self.greediness):
            # Find max probability
            max_probability = max(probabilities)

            # Gather the indices of probabilities that are equal to the max probability
            max_indices = [i for i, j in enumerate(
                probabilities) if j == max_probability]

            # From those max indices select random index
            neighbour_index = random.choice(max_indices)
            self.visit(available[neighbour_index])

            for i in max_indices:
                del i
            del max_indices

        else:
            probabilities = [x / denominator for x in probabilities]

            # Choose based on the probabillities calculated
            neighbour = random.choices(available, weights=probabilities, k=1)
            self.visit(neighbour[0])

            for i in neighbour:
                del i
            del neighbour[:]
            del neighbour

        for i in probabilities:
            del i
        del probabilities
        for i in available:
            del i
        del available[:]
        del available
        return True
