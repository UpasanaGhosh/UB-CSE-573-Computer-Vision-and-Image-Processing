"""
Image Stitching Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    good_matches_array = []
    MIN_MATCH_COUNT = 10
    left_img = cv2.resize(left_img, (0, 0), fx=1, fy=1)
    right_img = cv2.resize(right_img, (0, 0), fx=1, fy=1)
    left_image_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()

    # finding the key points and descriptors with SIFT
    kp_right, des_right = sift.detectAndCompute(right_image_gray, None)
    kp_left, des_left = sift.detectAndCompute(left_image_gray, None)
    right_keypoints = cv2.drawKeypoints(right_img, kp_right, None)
    # cv2.imwrite('results/right_keypoints.jpg', right_img)
    left_keypoints = cv2.drawKeypoints(left_img, kp_left, None)
    # cv2.imwrite('results/left_keypoints.jpg', left_img)

    match = cv2.BFMatcher()
    matches = match.knnMatch(des_right, des_left, k=2)

    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches_array.append(m)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       flags=2)
    img3 = cv2.drawMatches(right_img, kp_right, left_img, kp_left, good_matches_array, None, **draw_params)
    # cv2.imwrite("results/image_drawMatches.jpg", img3)

    # find homography
    if len(good_matches_array) > 10:
        src_pts = np.float32([kp_right[m.queryIdx].pt for m in good_matches_array]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_left[m.trainIdx].pt for m in good_matches_array]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = right_image_gray.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        left_image_gray = cv2.polylines(left_image_gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches_array), MIN_MATCH_COUNT))
        matchesMask = None

    # stitching the image
    stitched_img = cv2.warpPerspective(right_img, M, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    stitched_img[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    return stitched_img

    # raise NotImplementedError


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task2_result.jpg', result_image)
