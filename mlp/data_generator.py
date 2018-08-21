import numpy as np

class data_generator:
    def __init__(self):
        print("A new data generator object has been created.")
    def generate_samples_two_class_problem(self,nr_clusters,nr_samples_to_generate):
        np.random.seed(4)
        CLUSTER_RADIUS = 0.2

        # 1. generate random cluster coordinates
        clusters = []
        for i in range(nr_clusters):
            # 1.1 generate random cluster center
            center_x = np.random.rand()
            center_y = np.random.rand()

            # 1.2 store the center
            clusters.append(np.array([center_x,center_y]))

        # 2. generate random samples
        data_samples = []
        for i in range(nr_samples_to_generate):
            # 2.1 generate random coordinates
            rnd_x = np.random.rand()
            rnd_y = np.random.rand()
            rnd_coord = np.array([rnd_x,rnd_y])

            # 2.2 check whether that coordinate is near to the cluster
            # if yes , we say it belongs to class 1
            # if no, we say it belongs to class 0
            class_label = 0
            for j in range(nr_clusters):
                cluster_coords = clusters[j]
                # computer the distance of sample(rnd_x,rnd_y) to cluster coordniates(center_x,center_y)
                dist = np.linalg.norm(cluster_coords - rnd_coord)
                # is the sample near to that cluster
                if dist < CLUSTER_RADIUS:
                    class_label = 1
                    break
            # store the sample
            input_vec = np.array([rnd_x,rnd_y])
            output_vec = np.array([1 - class_label,class_label])
            data_samples.append([input_vec,output_vec])
        # 3. return the generated samples
        return data_samples