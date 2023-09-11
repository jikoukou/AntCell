import tensorflow as tf
from ACO import ACO
from Node import Node
from Graph import Graph
from preprocess import load_CIFAR_batch
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Add, ZeroPadding2D, Add, Layer
from tensorflow.keras.layers import Concatenate, Flatten, Dense, concatenate, Input
from tensorflow.keras.layers import BatchNormalization, Activation, DepthwiseConv2D, SeparableConv2D
import copy
import time
from tensorflow.keras.models import Model

btch_size = 64
initial_dropout = 0.015
flatten_dropout = 0.5


sgd = tf.keras.optimizers.Adam()


class Algorithm:
    def __init__(self, maxBlocks, x_train, y_train, x_test, y_test, max_attempts=10):
        self.maxBlocks = maxBlocks
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.attempts = max_attempts
        self.aco = None
        self.aco_reduction = None

    def generate_model(self, is_cifar, search_epochs=10, final_epochs=10):
        maxBlocks = self.maxBlocks
        CNN = []
        max_filters = 128

        filters = 32

        CNN.append(DepthwiseConv2D((3, 3), padding='same', activation='relu'))
        CNN.append(DepthwiseConv2D((5, 5), padding='same', activation='relu'))
        CNN.append(DepthwiseConv2D((7, 7), padding='same', activation='relu'))

        while (filters <= max_filters):
            CNN.append(
                Conv2D(filters, (3, 3), padding='same', activation='relu'))
            CNN.append(
                Conv2D(filters, (5, 5), padding='same', activation='relu'))
            CNN.append(
                Conv2D(filters, (1, 1), padding='same', activation='relu'))
            CNN.append(
                Conv2D(filters, (1, 7), padding='same', activation='relu'))
            CNN.append(SeparableConv2D(filters, (3, 3),
                       padding='same', activation='relu'))

            filters = 2*filters

        available_CNN = []

        for i in CNN:
            for j in range(maxBlocks):
                available_CNN.append(i)
            for j in range(2):
                available_CNN.append(i)

        Pool = [MaxPool2D(strides=2, padding='same'), MaxPool2D(pool_size=(5, 5), padding='same'),
                MaxPool2D(pool_size=(3, 3), padding='same'),
                AveragePooling2D(pool_size=(3, 3), padding='same')]
        available_Pool = []

        for i in Pool:
            for j in range(maxBlocks):
                available_Pool.append(i)
            for j in range(2):
                available_Pool.append(i)

        one_input_bef = Node(Layer(), [], [], [], -1)
        two_inputs_bef = Node(Layer(), [], [], [], -2)
        graph = Graph(available_CNN, available_Pool, maxBlocks, one_input_bef,
                      two_inputs_bef, 0.2)

        del available_Pool[:]
        del available_CNN[:]


#         CNN1 = []

#         filters = 32

#         while(filters <= max_filters):
#             CNN1.append(Conv2D(filters, (3, 3), padding='same', activation='relu'))
#             CNN1.append(Conv2D(filters, (5, 5), padding='same', activation='relu'))
#             CNN1.append(Conv2D(filters, (1, 1), padding='same', activation='relu'))
#             CNN1.append(Conv2D(filters, (1, 7), padding='same', activation='relu'))
#             CNN1.append(DepthwiseConv2D(filters, (7, 7), padding='same', activation='relu'))
#             CNN1.append(DepthwiseConv2D(filters, (3, 3), padding='same', activation='relu'))
#             CNN1.append(SeparableConv2D(filters, (3, 3), padding='same', activation='relu'))
#             CNN1.append(DepthwiseConv2D(filters, (5, 5), padding='same', activation='relu'))
#             filters = 2*filters

#         available_CNN1 = []

#         for i in CNN1:
#             for j in range(maxBlocks):
#                 available_CNN1.append(i)
#             for j in range(2):
#                 available_CNN1.append(i)


#         Pool1 = [MaxPool2D(strides= 2, padding='same'), MaxPool2D(pool_size = (5, 5), \
#                 padding='same'), MaxPool2D(pool_size = (3, 3), padding='same'), \
#                 AveragePooling2D(pool_size = (3, 3), padding='same')]
#         available_Pool1 = []

#         for i in Pool1:
#             for j in range(maxBlocks):
#                 available_Pool1.append(i)
#             for j in range(2):
#                 available_Pool1.append(i)

#         one_input_bef = Node(Layer(), [], [], [], -1)
#         two_inputs_bef = Node(Layer(), [], [], [], -2)
#         graph_reduction = Graph(copy.deepcopy(available_CNN), available_Pool1, maxBlocks, \
#                                 one_input_bef, two_inputs_bef, random.random())

        self.aco = ACO(maxBlocks, graph=graph)
#         self.aco_reduction = ACO(maxBlocks, graph=graph_reduction)
        self.aco_reduction = copy.deepcopy(self.aco)

        if (is_cifar):
            self.x_train, self.y_train = load_CIFAR_batch(
                'cifar-10-batches-py/data_batch_'+str(1))
        inp = cur = Input(
            shape=(self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3]))

        prev = cur
        accuracy = -1

        for i in range(self.attempts):
            start = time.time()

            # Checks if the new model is the best
            isNew = False

            self.aco.create_graph()
            self.aco_reduction.create_graph()

            self.aco.print_graph()
            print('===============')
            self.aco_reduction.print_graph()

            model, loss, new_accuracy, flat, dense, previous_layers = \
                self.fit_model(inp, search_epochs, is_cifar, full_model=True)

            end = time.time()

            if (new_accuracy > accuracy):
                isNew = True
                best_prev_layers = previous_layers
                best_aco = self.aco.copy()
                best_reduction = self.aco_reduction.copy()
                accuracy = new_accuracy
                print(new_accuracy)
            else:
                best_reduction.graph = (self.aco_reduction.graph)
                best_aco.graph = (self.aco.graph)

            best_aco.global_update(accuracy)
            best_reduction.global_update(accuracy)

            self.aco.graph = best_aco.graph
            self.aco_reduction.graph = best_reduction.graph

            self.aco_reduction.graph.port_recovery()
            self.aco_reduction.reset_aco(new_best=isNew)

            self.aco.reset_aco(new_best=isNew)
            self.aco.graph.port_recovery()

            print('\n---------------------------------------\n')

        self.aco = best_aco
        self.aco_reduction = best_reduction

        self.set_dropouts(initial_dropout)

        flat = Flatten()
        dense = Dense(10, activation='softmax')

        previous_layers = self.train_model(final_epochs, best_prev_layers, inp, dense,
                                           flat, is_cifar, 1, 1)[0]
        self.set_dropouts(-0.1)

        cells_output = self.stack_cells(inp, previous_layers, 1, 1, 0)[0]

        merged = flat(cells_output)
        merged = tf.keras.layers.Dropout(flatten_dropout)(merged)
        out = dense(merged)

        model = Model(inp, out)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
#         print('Final epoch')
#         for a in range(1, 6):
#             self.x_train, self.y_train = load_CIFAR_batch('cifar-10-batches-py/data_batch_'+str(a))


#             history = model.fit(self.x_train, self.y_train, epochs = 1, batch_size = btch_size)

        loss, accuracy = model.evaluate(self.x_test, self.y_test)

#         best_model = self.fit_model(inp, final_epochs, full_model = True, \
#                                     norm_weights = best_normal_weights, \
#                                     reduc_weights=best_reduction_weights)

        with open("aco.pkl", "wb") as f:
            pickle.dump(self.aco, f, -1)

        with open("aco_reduction.pkl", "wb") as f:
            pickle.dump(self.aco_reduction, f, -1)

        return model

    # Custom fit
    # Full_model is a flag which is used for resetting the dropout rates

    def train_model(self, search_epochs, previous_layers, inp, dense, flat, cifar, num_cells, iterations):
        #         print(self.aco.dropout, self.aco_reduction.dropout)
        self.set_dropouts(initial_dropout)
        max_acc = 0
        for j in range(search_epochs):
            #             print(self.aco.dropout)

            cells_output, previous_layers = self.stack_cells(
                inp, previous_layers, num_cells, iterations, j)


#             print(self.aco.dropout)
            merged = flat(cells_output)
            merged = tf.keras.layers.Dropout(flatten_dropout)(merged)
            out = dense(merged)

            model = Model(inp, out)

#             if(j == 0):
#                 model.summary()
#             print('----------------------------------')

            model.compile(
                optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

            print('Epoch', j+1, '/', search_epochs)

            if (cifar):
                for a in range(1, 6):
                    self.x_train, self.y_train = load_CIFAR_batch(
                        'cifar-10-batches-py/data_batch_'+str(a))

                    history = model.fit(self.x_train, self.y_train, epochs=1, batch_size=btch_size,
                                        verbose=1)

            else:
                history = model.fit(self.x_train, self.y_train, epochs=1, batch_size=btch_size,
                                    verbose=1)

            loss, accuracy = model.evaluate(self.x_test, self.y_test)

            self.increase_dropouts()
            if (max_acc < accuracy):
                max_acc = accuracy

        return previous_layers, max_acc

    def fit_model(self, inp, search_epochs, is_cifar, full_model=False, norm_weights=None, reduc_weights=None):
        dense = Dense(10, activation="softmax")
        flat = Flatten()

        normal_weights = None
        reduction_weights = None

        cur = inp
        prev = inp

        if (is_cifar):
            self.x_test, self.y_test = \
                load_CIFAR_batch('cifar-10-batches-py/test_batch')

        previous_layers = [[], []]
        previous_layers, max_acc = self.train_model(search_epochs, previous_layers, inp,
                                                    dense, flat, is_cifar, 1, 1)

        # Return full model after

        self.set_dropouts(-0.1)

        cells_output = self.stack_cells(inp, previous_layers, 1, 1, 0)[0]
        self.set_dropouts(initial_dropout)

        merged = flat(cells_output)
        out = dense(merged)

        model = Model(inp, out)

        model.compile(
            optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        return model, loss, max(accuracy, max_acc), flat, dense, previous_layers
    # Stack num_cells groups of cells one after another

    def stack_cells(self, model_inp, previous_layers, num_cells, iterations, epoch):

        cur = model_inp
        prev = model_inp

        new_prev_layers = [[], []]
        counter = 0

        if (num_cells == iterations == 1):
            add_batch = False

        else:
            add_batch = True

        for j in range(iterations):
            for i in range(num_cells):

                buff = cur
                cur = self.generate_normal_cell(cur, prev, epoch, add_batch)
#                 batch_norm = BatchNormalization()
#                 self.aco.layers[0].append(batch_norm)
#                 cur = (batch_norm)(cur)
                prev = buff

                if (len(previous_layers[0]) != 0):
                    self.aco.set_weights(previous_layers[0][counter])

                new_prev_layers[0].append(self.aco.layers[0])
                counter += 1
            # Output of reduce

            if (j == iterations - 1 and iterations > 1):
                break

            buff = cur


#             conv1 = Conv2D(prev.shape[3], (2, 2), strides=(2, 2), padding='same')
#             conv2 = Conv2D(prev.shape[3], (2, 2), strides=(2, 2), padding='same')
            cur = self.generate_reduction_cell((cur), (prev), epoch, add_batch)

#             batch_norm = BatchNormalization()
#             self.aco_reduction.layers[0].append(batch_norm)
#             cur = (batch_norm)(cur)
            prev = buff


#             self.aco_reduction.layers[0].append(conv1)
#             self.aco_reduction.layers[0].append(conv2)

            if (len(previous_layers[1]) != 0):
                self.aco_reduction.set_weights(previous_layers[1][j])

            new_prev_layers[1].append(self.aco_reduction.layers[0])

#         conv = Conv2D(1, (1, 1), activation='relu')
#         self.aco.layers[0].append(conv)
#         new_prev_layers[0].append(conv)

#         relu = tf.keras.layers.Activation(tf.keras.activations.relu)(cur)
        return (cur), new_prev_layers

    def generate_normal_cell(self, cur, prev, epoch, add_batch):
        conv1 = conv2 = None

        if (epoch % 2 == 1):
            local = False
        else:
            local = True

#         if(cur.shape[3] > prev.shape[3]):
#             conv1 = Conv2D(prev.shape[3], (1, 1), padding='same')

#             cur = (conv1)(cur)


#         elif(cur.shape[3] < prev.shape[3]):
#             conv2 = Conv2D(cur.shape[3], (1, 1), padding='same')
#             prev = (conv2)(prev)

        self.aco.pad_inputs([cur, prev])
        self.aco.initialize_layers(cur, prev, is_reduction=False)

#         if(conv1 != None):
#             self.aco.layers[0].append(conv1)

#         if(conv2 != None):
#             self.aco.layers[0].append(conv2)
#         batch_norm = BatchNormalizatio
        return self.aco.generate_cell(cur, local, add_batch)

    def generate_reduction_cell(self, cur, prev, epoch, add_batch):
        conv1 = conv2 = None

        if (epoch % 2 == 1):
            local = False
        else:
            local = True

        self.aco_reduction.pad_inputs([cur, prev])
        self.aco_reduction.initialize_layers(cur, prev, is_reduction=True)

        if (conv1 != None):
            self.aco_reduction.layers[0].append(conv1)

        if (conv2 != None):
            self.aco_reduction.layers[0].append(conv2)

        return (self.aco_reduction.generate_cell(cur, local, add_batch))

    def set_dropouts(self, new_val):
        self.aco.dropout = new_val
        self.aco_reduction.dropout = new_val

    def increase_dropouts(self):

        if (self.aco.dropout < 0):
            self.aco.dropout = self.aco.step
        else:
            self.aco.dropout += self.aco.step

        if (self.aco_reduction.dropout < 0):
            self.aco_reduction.dropout = self.aco_reduction.step

        else:
            self.aco_reduction.dropout += self.aco_reduction.step
