"""
K-Means Segmentation Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to segment image using k-means clustering.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are allowed to add your own functions if needed.
You should design you algorithm as fast as possible. To avoid repetitve calculation, you are suggested to depict clustering based on statistic histogram [0,255]. 
You will be graded based on the total distortion, e.g., sum of distances, the less the better your clustering is.
"""
from typing import Any, Union, List

import utils
import numpy as np
import json
import time


def kmeans(img, k):
    """
    Implement kmeans clustering on the given image.
    Steps:
    (1) Random initialize the centers.
    (2) Calculate distances and update centers, stop when centers do not change.
    (3) Iterate all initializations and return the best result.
    Arg: Input image;
         Number of K. 
    Return: Clustering center values;
            Clustering labels of all pixels;
            Minimum summation of distance between each pixel and its center.  
    """
    # Implementing function for K-Means Clustering.
    np.seterr(over='ignore')
    np_img = np.array(img)
    np_img = np_img.flatten()

    # Number of initializations
    initializations_counts = 50
    # List to store the centroids, labels matrix and sum_distance for all initializations
    initializations = []

    for n in range(0, initializations_counts):
        centroids = []
        centroids_same: List[bool] = []
        img_cluster_labels = np.copy(img)
        img_pixel_distance = np.copy(img)

        # Randomly initializing the cluster centers
        for i in range(0, k):
            centroids.append(np.random.choice(np_img))
            centroids_same.append(False)

        # Updating the centers until the centers do not change.
        while False in centroids_same:
            # Assigning pixels to clusters
            for i in range(0, len(img)):
                for j in range(0, len(img[0])):
                    prev_min_dist = 9999
                    cluster = k + 1

                    for p in range(0, len(centroids)):
                        curr_dist = np.sqrt((centroids[p] - img[i][j]) ** 2)
                        if curr_dist <= prev_min_dist:
                            prev_min_dist = curr_dist
                            cluster = p

                    img_cluster_labels[i][j] = cluster
                    img_pixel_distance[i][j] = prev_min_dist

            # Updating the cluster centroids
            clustered_pixel_values = {}
            for i in range(0, len(img)):
                for j in range(0, len(img[0])):
                    for p in range(k):
                        if img_cluster_labels[i][j] == p:
                            if p not in clustered_pixel_values.keys():
                                clustered_pixel_values[p] = [img[i][j]]
                            else:
                                clustered_pixel_values[p].append(img[i][j])

            # Calculating the mean
            for p in range(k):
                if len(clustered_pixel_values[p]) != 0:
                    list_sum = sum(clustered_pixel_values[p])
                    mean = float(list_sum / len(clustered_pixel_values[p]))
                    if centroids[p] == mean:
                        centroids_same[p] = True
                    else:
                        centroids[p] = mean
                else:
                    centroids[p] = 0.0
        # print(centroids)

        # Storing the centroids, labels matrix and sum_distance for all initializations
        empty_cluster_flag = 0.0
        if empty_cluster_flag not in centroids:
            # Calculating the sum of the distances
            sum_distance = np.sum(img_pixel_distance, dtype=np.float)
            # Saving data for each initialization
            initialization = {'centroids': centroids, 'labels': img_cluster_labels, 'sumdistance': sum_distance}
            initializations.append(initialization)

    # Sorting the initializations based on sum_distance to get the optimized clustering with the minimum sum_distance
    initializations = sorted(initializations, key=lambda k: k['sumdistance'])
    result = initializations[0]
    return result['centroids'], result['labels'], result['sumdistance']


def visualize(centers, labels):
    """
    Convert the image to segmentation map replacing each pixel value with its center.
    Arg: Clustering center values;
         Clustering labels of all pixels. 
    Return: Segmentation map.
    """
    # Implementing the visualization
    for i in range(0, len(labels)):
        for j in range(0, len(labels[0])):
            label = int(labels[i][j])
            center = centers[label]
            labels[i][j] = center

    # Converting the labels type to Uint8
    labels = labels.astype(np.uint8)
    return labels


if __name__ == "__main__":
    img = utils.read_image('data/lenna.png')
    k = 2

    start_time = time.time()
    centers, labels, sumdistance = kmeans(img, k)

    result = visualize(centers, labels)
    end_time = time.time()

    running_time = end_time - start_time
    print(running_time)

    centers = list(centers)
    with open('results/task1.json', "w") as jsonFile:
        jsonFile.write(json.dumps({"centers": centers, "distance": sumdistance, "time": running_time}))
    utils.write_image(result, 'results/task1_result.jpg')
