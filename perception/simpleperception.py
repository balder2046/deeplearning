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
def main():
    x_train,y_train,x_test,y_test = read_mnist_data()
    show_some_mnist_images(10,x_train,y_train)
if __name__ == "__main__":
    main()