import numpy as np
import math

def func_identity(x):
    return x

def func_sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def func_relu(x):
    return x if x > 0.0 else 0

def func_squared(x):
    return x * x

def derivative_identity(x):
    return 1

def derivative_sigmoid(x):
    return func_sigmoid(x) * (1.0 - func_sigmoid(x))

def derivative_relu(x):
    return 1 if x > 0 else 0

def derivative_squared(x):
    return 2 * x

func_identity = np.vectorize(func_identity)
func_sigmoid = np.vectorize(func_sigmoid)
func_relu = np.vectorize(func_relu)
func_skew_ramp = np.vectorize(func_squared)
derivative_identity = np.vectorize(derivative_identity)
derivative_sigmoid = np.vectorize(derivative_sigmoid)
derivative_relu = np.vectorize(derivative_relu)
derivative_skew_ramp = np.vectorize(derivative_squared)

class TF:
    identity = 1
    sigmoid = 2
    relu = 3
    squared = 4

class mlp:
    nr_layers = 0
    nr_neurons_per_layer = []
    tf_per_layer = []
    weight_matrices = []
    neuron_act_vecs = []
    neuron_out_vecs = []
    learn_rate = 0.01
    neuron_err_vecs = []

    def __init__(self):
        print("Gernerate a new empty MLP")
    
    """
    return the output of the mlp as numpy array
    """
    def get_output_vector(self):
        return self.neuron_out_vecs[len(self.neuron_out_vecs) - 1]
    
    def show_architecture(self):
        print("the MLP architecture is now: ",end = " ")
        for i in range(0,self.nr_layers):
            print(str(self.nr_neurons_per_layer[i]),end = " ")
        print("\n")

    """
    add a new layer of neurons
    """
    def add_layer(self,nr_neurons,transfer_function):
        # 1. store the number of neurons and transfer function to use
        self.nr_neurons_per_layer.append(nr_neurons)
        self.tf_per_layer.append(transfer_function)
        if self.nr_layers > 0:
            nr_neurons_before = self.nr_neurons_per_layer[-1]
            # initialize the weight from -1 to 1 
            W = np.uniform.random(low = -1.0,high = 1.0,size=(nr_neurons_before + 1,nr_neurons))
            # store the new matrix
            self.weight_matrices.append(W)
            print("Generate the new weight matrix ,the new shape is ",W.shape)
            size = W.nbytes / 1024
            print("the weight matrix 's data size is %.2f kb" % size)
        act_vec = np.zeros(nr_neurons)
        out_vec = np.zeros(nr_neurons)
        err_vec = np.zeros(nr_neurons)
        self.neuron_act_vecs.append(act_vec)
        self.neuron_err_vecs.append(err_vec)
        self.neuron_out_vecs.append(out_vec)
        # update number of layers
        self.nr_layers += 1
        # show the architectures
        self.show_architecture()

    """
    Give a input vector, we compute the output of all neurons layer by layer into the direction of the output layer
    """
    def feedfoward(self,input_vec):
        N = len(input_vec)
        self.neuron_out_vecs[0] = input_vec
        for layer_nr in range(1,self.nr_layers):
            o = self.neuron_out_vecs[layer_nr - 1]
            # add the bias input
            o = np.append([1],o)
            N = len(o)
            # vectors are one dimensional,but matrxi mulplication must be a matrix
            o_Mat = o.reshape((1,N))
            W = self.weight_matrices[layer_nr - 1]
            act_mat_this_layer = np.matmul(o_Mat,W)
            tfunction = self.tf_per_layer[layer_nr - 1]
            if tfunction == TF.identity:
                out_mat_this_layer = func_identity(act_mat_this_layer)
            elif tfunction == TF.sigmoid:
                out_mat_this_layer = func_sigmoid(act_mat_this_layer)
            elif tfunction == TF.relu:
                out_mat_this_layer = func_relu(act_mat_this_layer)
            elif tfunction == TF.squared:
                out_mat_this_layer = func_squared(act_mat_this_layer)
            self.neuron_act_vecs[layer_nr] =act_mat_this_layer.flatten()
            self.neuron_out_vecs[layer_nr] = out_mat_this_layer.flatten()

    """
    show the output of all neurons in a specific layer.
    """
    def show_output(self,layer):
        print("output value of neurons in a layer ",layer, " values: ",self.neron_out_vecs[layer])
        print("\n")

    """
    Shows some statistics about weights, e.g. what is the maximum and the minimum weight 
    in each matrix
    """
    def show_weight_statistics(self):
        for i in range(0,self.nr_layers - 1):
            print("weight from layer ",i, " to layer ",i + 1)
            W = self.weight_matrices[i]
            print("the shape of weights ",W.shape)
            print("the min value is ",np.amin(W))
            print("the max value is ",np.amax(W))
            print("the w is ",W)
        print("\n")

    """
    Show state of neurons (activity and output values)
    """
    def show_neuron_states(self):
        for i in range(0,self.nr_layers):
            print("layer ",i)
            print(" act :",self.neuron_act_vecs[i])
            print(" out :",self.neuron_out_vecs[i])
        print("\n")
    """
    set a new learn rate which is used in the weight update step
    """
    def set_learn_rate(self,new_learn_rate):
        self.learn_rate = new_learn_rate
    
    """
    Given a pair(input_vec, teacher_vec) we adapt the weights of the MLP
    such that the desired output vector (which is the teacher vector)
    is more likely to be generated the next time if the input vector is presented as input
    """
    def train(self,input_vec,teacher_vec):
        # 1. first do a feedfoward step with the input vector
        self.feedfoward(input_vec)
        # 2. first compute the error signals for the output neurons
        tf_type = self.tf_per_layer[-1]
        nr_neurons = self.nr_neurons_per_layer[-1]
        act_vec = self.neuron_act_vecs[-1]
        out_vec = self.neuron_out_vecs[-1]
        err_vec = -(out_vec - teacher_vec)
        if tf_type == TF.sigmoid:
            err_vec *= derivative_sigmoid(act_vec)
        elif tf_type == TF.identity:
            err_vec *= derivative_identity(act_vec)
        elif tf_type == TF.relu:
            err_vec *= derivative_relu(act_vec)
        elif tf_type == TF.squared:
            err_vec *= derivative_skew_ramp(act_vec)
        self.neuron_err_vecs[-1] = err_vec
        for layer_nr in range(self.nr_layers - 2, 0, -1):
            nr_neurons_this_layer = self.nr_neurons_per_layer[layer_nr]
            nr_neurons_next_layer = self.nr_neurons_per_layer[layer_nr + 1]
            W = self.weight_matrices[layer_nr]
            act_vec = self.neuron_act_vecs[layer_nr]
            tf_type = self.tf_per_layer[layer_nr]
            # run over all neurons in this year ...
            for neuron_nr in range(0,nr_neurons_this_layer):
                # compute the sum of weighted error signals from neurons in the next year
                sum_of_weighted_error_signals = 0.0
                for neuron_nr2 in range(0,nr_neurons_next_layer):
                    # get error singal for neuron_nr2 in the next layer
                    err_vec = self.neuron_err_vecs[nr_neurons_next_layer]
                    err_signal = err_vec[neuron_nr2]
                    # get weight from neuron_nr