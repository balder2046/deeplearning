from random import randint
import cv2
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
'''
The ramp transfer function
'''
def f(x):
    if x <= 0:
        return 0
    else:
        return 1

f = np.vectorize(f)

'''
Download & unpack the MNIST data
Also prepare direct access to data matrices:
x_train,y_train,x_test,y_test
'''
def read_mnist_data():
    # 1. download and read data
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    # 2. show data type of the mnist object
    print("type of mnist is ",type(mnist))
    # 3. show number of trainning and test examples
    print("There are ", mnist.train.num_examples," trainning examples available.")
    print("There are ", mnist.test.num_examples," test examples available!")
    
    # 4. prepare matrices numpy.ndarrays to
    # access the trainning / test images and labels
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    print("type of x_train is ", type(x_train))
    print("x_train: ",x_train.shape)
    print("y_train: ",y_train.shape)
    print("x_test: ",x_test.shape)
    print("y_test: ",y_test.shape)
    return x_train,y_train,x_test,y_test

'''
This function will show n random example images from the training set,
visualized using OpenCVs imshow() function
'''
def show_some_mnist_images(n,x_train,y_train):
    nr_train_examples  = x_train.shape[0]
    for i in range(0,n):
        # 1. guess a random number between 0 and 55.000 - 1
        rnd_number = randint(0,nr_train_examples)
        # 2. get corresponding output vector
        correc_out_vec = y_train[rnd_number,:]
        # 3. get first row of 28x28 pixels = 784 values
        row_vec = x_train[rnd_number,:]
        print("type of row_vec is ",type(row_vec))
        print("shape of row_vec is ",row_vec.shape)
        M = row_vec.reshape(28,28)
        M = cv2.resize(M,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image',M)
        c = cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_weights(weights,x_test,y_test):
    correct_rate,nr_correct,nr_wrong = test_perceptron(weights,x_test,y_test)
    print(correct_rate * 100.0," % of the test patterns", " were correcty classified.")
    print("correctly classified: ",nr_correct)
    print("wrongly classified:" ,nr_wrong)
    print("*************************************************")



'''
Generate a weight matrix of dimension
(nr-of-inputs, nr-of-outputs)
and train the weights according to the Perception learning rule use random sample 
patterns <input,desired output> from the MNIST training dataset
'''
def generate_and_train_perception_classifier(nr_train_steps,x_train,y_train,x_test,y_test):
    nr_train_examples = x_train.shape[0]
    # 1. generate Perceptron with random weights
    weights = np.random.rand(785,10)
    # 2. do the desired number of trainning steps
    for train_step in range(0,nr_train_steps):
        if train_step % 1000 == 0 and train_step > 0:
            print("train step ", train_step)
            test_weights(weights,x_test,y_test)
        # 2.2 choose a random image
        rnd_number = randint(0,nr_train_examples - 1)
        
        input_vec = x_train[rnd_number,:]
        # add bias input "1"
        input_vec = np.append(input_vec,[1])
        input_vec = input_vec.reshape(1,785)
        # 2.3 compute Percetron output
        act = np.matmul(input_vec,weights)
        out_mat = f(act)
        # 2.4 compute difference vector
        teacher_out_mat = y_train[rnd_number,:]
        teacher_out_mat = teacher_out_mat.reshape(1,10)
        diff_mat = teacher_out_mat - out_mat
        #2.5 correct weights
        learn_rate = 0.01
        for neuron_nr in range(0,10):
            # 2.5.1 get neuron error
            neuron_error = diff_mat[0,neuron_nr]
            # 2.5.2 for all weights to the current
            # neuron
            for weight_nr in range(0,785):
                x_i = input_vec[0,weight_nr]
                delta_w_i = learn_rate * neuron_error * x_i
                weights[weight_nr,neuron_nr] += delta_w_i
    return weights

'''
New test how good the Perceptron can classify on data never seen before
i.e., the test  data
'''
def test_perceptron(weights,x_test,y_test):
    nr_test_examples = x_test.shape[0]

    # 1. initialize counters
    nr_correct = 0
    nr_wrong = 0
    # 2. forward all test patterns
    # then compare predicted label with ground truth label and check wheter
    # the prediction is right or not
    for test_vec_nr in range(0,nr_test_examples):
        input_vec = x_test[test_vec_nr,:]
        input_vec = np.append(input_vec,[1])
        input_vec = input_vec.reshape(1,785)
        # 2.2 get the desired output vector
        teacher_out_mat = y_test[test_vec_nr,:]
        teacher_out_mat = teacher_out_mat.reshape(1,10)
        teacher_class = np.argmax(teacher_out_mat)
        act = np.matmul(input_vec,weights)
        out_mat = f(act)
        actual_class = np.argmax(out_mat)
        if teacher_class == actual_class:
            nr_correct += 1
        else:
            nr_wrong += 1
    correct_rate = float(nr_correct) / float(nr_correct + nr_wrong)
    return correct_rate,nr_correct,nr_wrong




def main():
    x_train,y_train,x_test,y_test = read_mnist_data()
    #show_some_mnist_images(10,x_train,y_train)
    weights = generate_and_train_perception_classifier(100000,x_train,y_train,x_test,y_test)
    correct_rate,nr_correct,nr_wrong = test_perceptron(weights,x_test,y_test)
    print(correct_rate * 100.0," % of the test patterns", " were correcty classified.")
    print("correctly classified: ",nr_correct)
    print("wrongly classified:" ,nr_wrong)
    print("Program end.")
if __name__ == "__main__":
    main()