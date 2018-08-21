'''
Minimalistic example of a MLP in TensorFlow
'''

import numpy as np
from data_generator import data_generator
import tensorflow as tf
import cv2
from timeit import default_timer as timer

# test data parameters
WINSIZE = 600
NR_CLUSTERS = 5
NR_SAMPLES_TO_GENERATE = 10000
# MLP parameters
NR_EPOCHS = 2000

# for RELU transfer function use smaller learn rate
# than for logistic trnasfer function
# Also use more hidden neurons! (e.g. 2-30-12-2)
# LEARN_RATE = 0.1
# for logistic transfer function
LEARN_RATE = 0.5
MINI_BATCH_SIZE = 100
NR_NEURONS_INPUT = 2
NR_NEURONS_HIDDEN1 = 10
NR_NEURONS_HIDDEN2 = 6
NR_NEURONS_OUTPUT = 2

# store 2d weight matrices & 1D bias vectors for all
# neuron layers in two dictionaries
weights = {
    'h1',tf.Variable(tf.random_normal([NR_NEURONS_INPUT,NR_NEURONS_HIDDEN1])),
    'h2',tf.Variable(tf.random_normal([NR_NEURONS_HIDDEN1,NR_NEURONS_HIDDEN2])),
    'out',tf.Variable(tf.random_normal([NR_NEURONS_HIDDEN2,NR_NEURONS_OUTPUT]))
}
bias = {
    'b1': tf.Variable(tf.random_normal([NR_NEURONS_HIDDEN1])),
    'b2': tf.Variable(tf.random_normal([NR_NEURONS_HIDDEN2])),
    'out': tf.Variable(tf.random_normal([NR_NEURONS_OUTPUT]))
}

# visualization parameters
RADIUS_SAMPLE = 3
COLOR_CLASS0 = (255,0,0)
COLOR_CLASS1 = (0,0,255)
NR_TEST_SAMPLES = 10000

# for saving images
image_counter = 0

'''
helper function to create a 4 layer MLP
input -layer --> 
    hidden layer @1 -->
        hidden layer @2 -->
            output layer
'''
def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    out_layer = tf.matmul(layer_2,weights['out']) + biases['out']
    return out_layer

def generate_and_show_training_data():
    my_dg = data_generator()
    data_samples = my_dg.generate_samples_two_class_problem(NR_CLUSTERS,NR_SAMPLES_TO_GENERATE)
    nr_samples = len(data_samples)
    img = np.ones((WINSIZE,WINSIZE,3),np.uint8) * 255
    for i in range(nr_samples):
        next_sample = data_samples[i]
        input_vec = next_sample[0]
        output_vec = next_sample[1]
        sample_coord = (int(input_vec[0] * WINSIZE),int(input_vec[1] * WINSIZE))
        if output_vec[0] > output_vec[1] :
            class_label = 0
        else:
            class_label = 1
        color = (0,0,0)
        if class_label == 0:
            color = COLOR_CLASS0
        else:
            color = COLOR_CLASS1
        cv2.circle(img.sample_coord,RADIUS_SAMPLE,color)
    cv2.imshow('Training data',img)
    c = cv2.waitKey(1)
    cv2.imwrite("e:/tmp/tf/training_data.png",img)
    return data_samples

def visualize_decision_boundaries(the_session,epoch_nr,x_in,mlp_output_vec):
    global image_counter
    NR_TEST_SAMPLES = 10000
    input_mat = np.zeros((NR_TEST_SAMPLES,NR_NEURONS_INPUT))
    for i in range(NR_TEST_SAMPLES):
        rnd_x = np.random.rand()
        rnd_y = np.random.rand()
        input_vec = np.array([rnd_x,rnd_y])
        input_mat[i,:] = input_vec
    res = the_session.run(mlp_output_vec,feed_dict={x_in:input_mat})
    img = np.ones((WINSIZE,WINSIZE,3),np.uint8) * 255
    for i in range(NR_TEST_SAMPLES):
        input_vec = input_mat[i,:]
        output_vec = res[i,:]
        class_label = 0 if output_vec[0] > output_vec[1] else 1
        color = COLOR_CLASS0 if class_label == 0 else COLOR_CLASS1
        sample_coord = (int(input_vec[0] * WINSIZE),int(input_vec[1] * WINSIZE))
        cv2.circle(img,sample_coord,RADIUS_SAMPLE,color)
    cv2.rectangle(img,(WINSIZE - 120,0),(WINSIZE - 1,20),(255,255,255),-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,"epoch #" + str(epoch_nr).zfill(3),
    (WINSIZE - 110,15),font,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow('Decision boundaries of trained MLP',img)
    c = cv2.waitKey(1)
    if True:
        filename = "e:/tmp/tf/img_{0:0>4}".format(image_counter)
        image_counter += 1
        cv2.imwrite(filename + ".png",img)

def build_TF_graph():
    # 1. prepare placehodlers for the input and output values
    # the input is a 2D matrix
    # in each row we store one input vector
    x_in = tf.placeholder("float")
    # the output is a 2D matrix
    # in each row we store one output vector
    y_out = tf.placeholder("float")
    
    # 2. now the use helper function defined before to generate a MLP
    mlp_output_vec = multilayer_perceptron(x_in,weights,biases)
    # 3. define a loss function
    loss = tf.reduce_mean(tf.squared_difference(mlp_output_vec,y_out))
    # 4. add an optimizer to the graph
    optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss)

    # 5. crate a  summary to track current value of loss function
    tf.summary.scalar("value-of-loss",loss)
    # 6. in case we want to track multiple summaries merge all summaries into a single operation
    summary_op = tf.summary.merge_all()
    return optimizer,mlp_output_vec,loss,x_in,y_out

def MLP_training(data_samples, optimizer,mlp_output_vec,loss,x_in,y_out):
    NR_SAMPLES = len(data_samples)
    with tf.Session() as my_session:
        # initialize all variables
        my_session.run(tf.global_variables_initializer())
        fw = tf.summary.FileWriter("e:/tmp/tf/summary",my_session.graph)
        # how many mini batches do we have to process?
        nr_batches_to_process = int(NR_SAMPLES/ MINI_BATCH_SIZE)
        # in each epoch all training samples will be presented
        for epoch_nr in range(0,NR_EPOCHS):
            print("Training MLP. Epoch nr #",epoch_nr)
            # in each mini batch some of the training samples will be feed 
            # forwarded, the weight changes for a single sample will be computed and all weight changes be accumulated
            # for all samples in the mini-batch
            # Then the weights will be updated
            start = timer()
            for mini_batch_nr in range(0,nr_batches_to_process):
                # a) generate list of indices
                sample_indices = np.arange(0,NR_SAMPLES)
                sample_indices = np.random.shuffle(sample_indices)
                input_matrix = np.zeros((MINI_BATCH_SIZE,NR_NEURONS_INPUT))
                output_matrix = np.zeros((MINI_BATCH_SIZE,NR_NEURONS_OUTPUT))
                startpos = mini_batch_nr * MINI_BATCH_SIZE
                row_counter = 0
                for next_sample_id in range(startpos,startpos + MINI_BATCH_SIZE):
                    # get next trainning sample from dataset class
                    # the dataset is a list of lists
                    #in each list entry there are two vectors:
                    # the input vector and the output vector
                    next_sample = data_samples[next_sample_id]
                    input_vec = next_sample[0]
                    output_vec = next_sample[1]
                    input_matrix[row_counter,:] = input_vec
                    output_matrix[row_counter,:]= output_vec
                    row_counter += 1
                # d) run the optimizer node --> training will happend now the 
                # actual feed-forward step and the computations will happen!
                _, curr_loss = my_session.run([optimizer,loss],feed_dict={x_in:input_matrix,y_out:output_matrix})
            end = timer()
            print("time needed to train on epoch: ",end - start," sec")
            print("Now Testing the MLP ...")
            visualize_decision_boundaries(my_session,epoch_nr,x_in,mlp_output_vec)

            def main():
                data_samples = generate_and_show_training_data()
                optimizer,mlp_output_vec,loss,x_in,y_out= build_TF_graph()
                MLP_training(data_samples,optimizer,mlp_output_vec,loss,x_in,y_out)
                print("end of MLP TensorFlow test")
