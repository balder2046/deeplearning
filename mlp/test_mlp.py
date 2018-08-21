import numpy as np
from data_generator import data_generator
from mlp import mlp
from mlp import TF
import cv2

# test data parameters
WINSIZE = 600
NR_CLUSTERS = 5
NR_SAMPLES_TO_GENERATE = 20000

# MLP parameters
NR_EPOCHS = 31
LEARN_RATE = 0.1

# visualization parameters
RADIUS_SAMPLE = 3
COLOR_CLASS0 = (255,0,0)
COLOR_CLASS1 = (0,0,255)
NR_TEST_SAMPLES = 5000

# for saving images
image_counter = 0

def visualize_decision_boundaries(the_mlp,epoch_nr):
    global image_counter

    # 1. generate empty white color image
    img = np.ones((WINSIZE,WINSIZE,3),np.uint8) * 255
    for i in range(NR_TEST_SAMPLES):

        # 2. generate random coordinate in [0,1) x [0,1)
        rnd_x = np.random.rand()
        rnd_y = np.random.rand()

        # 3. prepare an input vector
        input_vec = np.array([rnd_x,rnd_y])

        # 4. now do a feedforward step with that input vector. i.e.,
        # compute the output values of the MLP for the input vector
        the_mlp.feedforward(input_vec)

        # 5. now get the predicted class from output vector
        output_vec = the_mlp.get_output_vector()
        class_label = 0 if output_vec[0] > output_vec[1] else 1

        # 6. Map class label to a color
        color = COLOR_CLASS0 if class_label == 0 else COLOR_CLASS1

        #7. Draw circle
        sample_coord = (int(input_vec[0] * WINSIZE),int(input_vec[1] * WINSIZE))
        cv2.circle(img,sample_coord,RADIUS_SAMPLE,color)
    # 8. show image with decision boundaries
    cv2.rectangle(img,(WINSIZE - 120,0),(WINSIZE - 1, 20),(255,255,255),-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,"epoch #" + str(epoch_nr).zfill(3),(WINSIZE - 110,15),font,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow('Decision boundaries of trained MLP',img)
    c = cv2.waitKey(1)
    if True:
        filename = "e:/tmp/img_{0:0>4}".format(image_counter)
        image_counter += 1
        cv2.imwrite(filename + ".png",img)
    
my_mlp = mlp()

my_mlp.add_layer(2,TF.identity)
my_mlp.add_layer(10,TF.sigmoid)
my_mlp.add_layer(6,TF.sigmoid)
my_mlp.add_layer(2,TF.identity)

my_dg = data_generator()
data_samples = my_dg.generate_samples_two_class_problem(NR_CLUSTERS,NR_SAMPLES_TO_GENERATE)
nr_samples = len(data_samples)

# 4. generate empty image for visiualization
# initialize image with the pixels(255,255,255)
img = np.ones((WINSIZE,WINSIZE,3),np.uint8) * 255
for i in range(nr_samples):
    # 5.1 get the next data_samples
    next_sample = data_samples[i]
    # 5.2 get the input and output vector
    # which are both numpy arrays
    input_vec = next_sample[0]
    output_vec = next_sample[1]

    # 5.3 prepare a tupel from th enumpy input vector
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
    # 5.5
    cv2.circle(img,sample_coord,RADIUS_SAMPLE,color)

# 6. show visualization of samples

cv2.imshow("Training data", img)
c = cv2
cv2.imwrite("e:/tmp/training_data.png",img)

# 7. now train the MLP
my_mlp.set_learn_rate(LEARN_RATE)
print("sample count",nr_samples)
for epoch_nr in range(NR_EPOCHS):
    print("Training epoch#",epoch_nr)
    sample_indices = np.arange(nr_samples)
    np.random.shuffle(sample_indices)
    count = 0
    for train_sample_nr in range(nr_samples):
        # get index of next training sample
        index = sample_indices[train_sample_nr]
        # get next sample
        next_sample = data_samples[index]
        input_vec = next_sample[0]
        output_vec = next_sample[1]
        # train the MLP with that vector paire
        my_mlp.train(input_vec,output_vec)
        count = count + 1
        if count % 1000 == 0:
            print("process count ",count)
    print("\nMLP state after training epoch #",epoch_nr,":")
    my_mlp.show_weight_statistics()
    my_mlp.show_neuron_states()
    visualize_decision_boundaries(my_mlp,epoch_nr)
print("MLP test finished")
    
