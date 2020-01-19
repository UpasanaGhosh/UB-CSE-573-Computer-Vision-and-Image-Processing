"""
Image Filtering
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with image filtering and familiarize you with 'tricks', e.g., padding, commonly used by computer vision 'researchers'.

Please complete all the functions that are labelled with '# TODO'. Steps to complete those functions are provided to make your lives easier. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
are building blocks you could use when implementing the functions labelled with 'TODO'.

I strongly suggest you read the function 'zero_pad' and 'crop' that are defined in 'utils.py'. You will need them!

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""

import argparse
import copy
import os

import cv2
import numpy as np

import utils

# low_pass filter and high-pass filter
low_pass = [[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]
high_pass = [[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 2, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]]


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task1.jpg",
        help="path to the image"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="high-pass",
        choices=["low-pass", "high-pass"],
        help="type of filter"
    )
    parser.add_argument(
        "--result-saving-dir",
        dest="rs_dir",
        type=str,
        default="./results/",
        help="directory to which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)

    img = [list(row) for row in img]
    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def write_image(img, img_saving_path):
    """Writes an image to a given path.
    """
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")

    cv2.imwrite(img_saving_path, img)


def convolve2d(img, kernel):
    """Convolves a given image and a given kernel.

    Steps:
        (1) flips the kernel
        (2) pads the image # IMPORTANT
            this step handles pixels along the border of the image, and ensures that the output image is of the same size as the input image
        (3) calculates the convolved image using nested for loop

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel.

    Returns:
        img_conv: nested list (int), convolved image.
    """
    # flips the kernel
    flipped_kernel = utils.flip2d(kernel)
    # using 2k+1 form to calc the value of k
    k = int((len(kernel) - 1)/2)

    # padding the image
    padded_img = utils.zero_pad(img, k, k)

    # TODO:implement this function.
    # implementing convolution
    # master loop for img array
    for row in range(0, len(img)):
        for col in range(0, len(img[0])):
            # creating the convolution matrix for applying the filter from img
            convolve_img_matrix = utils.crop(padded_img, row, row+len(kernel), col, col+len(kernel[0]))
            convolve_img_matrix_mul_kernel = utils.elementwise_mul(convolve_img_matrix, flipped_kernel)
            # calculating the convoluted value for the pixel
            convoluted_value = 0
            for val_list in convolve_img_matrix_mul_kernel:
                for val in val_list:
                    convoluted_value = convoluted_value + val
            # replacing the value in padded_img
            img[row][col] = convoluted_value

    return img

    # raise NotImplementedError


def main():
    args = parse_args()

    img = read_image(args.img_path)

    if args.filter == "low-pass":
        kernel = low_pass
    elif args.filter == "high-pass":
        kernel = high_pass
    else:
        raise ValueError("Filter type not recognized.")

    if not os.path.exists(args.rs_dir):
        os.makedirs(args.rs_dir)

    filtered_img = convolve2d(img, kernel)
    write_image(filtered_img, os.path.join(args.rs_dir, "{}.jpg".format(args.filter)))


if __name__ == "__main__":
    main()
