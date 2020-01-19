"""
Denoise Problem
(Due date: Nov. 25, 11:59 P.M., 2019)
The goal of this task is to denoise image using median filter.

Do NOT modify the code provided to you.
Do NOT import ANY library or API besides what has been listed.
Hint: 
Please complete all the functions that are labeled with '#to do'. 
You are suggested to use utils.zero_pad.
"""

import utils
import numpy as np
import json


def median_filter(img):
    """
    Implement median filter on the given image.
    Steps:
    (1) Pad the image with zero to ensure that the output is of the same size as the input image.
    (2) Calculate the filtered image.
    Arg: Input image. 
    Return: Filtered image.
    """
    # Padding the image
    padded_img = utils.zero_pad(img, 1, 1)

    # Implementing the Median Filter
    for row in range(0, len(img)):
        for col in range(0, len(img[0])):
            # Finding the neighbouring pixels for the pixel at location row,col in the image
            filter_patch = padded_img[row: row + 3]
            filter_patch = [row[col: col + 3] for row in filter_patch]
            filter_matrix = []
            filter_matrix.extend(filter_patch[0])
            filter_matrix.extend(filter_patch[1])
            filter_matrix.extend(filter_patch[2])

            # Calculating the Median
            filter_matrix.sort()
            median_index = int((len(filter_matrix) + 1) / 2)
            median = filter_matrix[median_index - 1]

            # Replacing the selected pixel at location row,col with the Median
            img[row][col] = median

    return img


def mse(img1, img2):
    """
    Calculate mean square error of two images.
    Arg: Two images to be compared.
    Return: Mean square error.
    """
    # Mean square error implementation
    mse_val = 0
    if len(img1) == len(img2) and len(img1[0]) == len(img2[0]):
        total = 0
        for row in range(0, len(img1)):
            for col in range(0, len(img1[0])):
                total = total + ((img1[row][col] - img2[row][col]) ** 2)
        mse_val = total / (len(img1) * len(img1[0]))
    else:
        print("Images not of same size")
    return mse_val


if __name__ == "__main__":
    img = utils.read_image('data/lenna-noise.png')
    gt = utils.read_image('data/lenna-denoise.png')

    result = median_filter(img)
    error = mse(gt, result)

    with open('results/task2.json', "w") as file:
        json.dump(error, file)
    utils.write_image(result, 'results/task2_result.jpg')
