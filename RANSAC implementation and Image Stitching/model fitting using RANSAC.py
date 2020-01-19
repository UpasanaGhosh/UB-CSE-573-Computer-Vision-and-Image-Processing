"""
RANSAC Algorithm Problem
(Due date: Oct. 23, 3 P.M., 2019)
The goal of this task is to fit a line to the given points using RANSAC algorithm, and output
the names of inlier points and outlier points for the line.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
You can use the library random
Hint: It is recommended to record the two initial points each time, such that you will Not 
start from this two points in next iteration.
"""
import random


def solution(input_points, t, d, k):
    """
    :param input_points:
           t: t is the perpendicular distance threshold from a point to a line
           d: d is the number of nearby points required to assert a model fits well, you may not need this parameter
           k: k is the number of iteration times
           Note that, n for line should be 2
           (more information can be found on the page 90 of slides "Image Features and Matching")
    :return: inlier_points_name, outlier_points_name
    inlier_points_name and outlier_points_name is two list, each element of them is str type.
    For example: If 'a','b' is inlier_points and 'c' is outlier_point.
    the output should be two lists of ['a', 'b'], ['c'].
    Note that, these two lists should be non-empty.
    """
    # TODO: implement this function.
    inlier_points_name = []
    outlier_points_name = []
    considered_pt_pairs = []
    err_avg = ''

    for i in range(k):
        p1 = random.choice(input_points)
        p2 = random.choice(input_points)

        if p1['name'] != p2['name'] and (p1['name'], p2['name']) not in considered_pt_pairs \
                and (p2['name'], p1['name']) not in considered_pt_pairs:
            considered_pt_pairs.append((p1['name'], p2['name']))
            p1_value = p1['value']
            p2_value = p2['value']

            x_coeff = p1_value[1] - p2_value[1]
            y_coeff = p2_value[0] - p1_value[0]
            c = (p1_value[0] * p2_value[1]) - (p2_value[0] * p1_value[1])
            inliers = []
            outliers = []
            err_sum = 0

            for point in input_points:
                if point['name'] != p1['name'] and point['name'] != p2['name']:
                    p_value = point['value']
                    numerator = abs((x_coeff * p_value[0]) + (y_coeff * p_value[1]) + c)
                    denominator = (x_coeff ** 2 + y_coeff ** 2) ** 0.5
                    dist = float(numerator / denominator)

                    if dist <= t:
                        inliers.append(point['name'])
                        err_sum = err_sum + dist
                    else:
                        outliers.append(point['name'])

            if len(inliers) >= d:
                current_err_avg = err_sum/len(inliers)
                if err_avg == '':
                    inlier_points_name = inliers
                    inlier_points_name.append(p1['name'])
                    inlier_points_name.append(p2['name'])
                    outlier_points_name = outliers
                    err_avg = str(current_err_avg)

                elif current_err_avg < float(err_avg):
                    inlier_points_name = inliers
                    inlier_points_name.append(p1['name'])
                    inlier_points_name.append(p2['name'])
                    outlier_points_name = outliers
                    err_avg = str(current_err_avg)

                elif current_err_avg == float(err_avg):
                    if len(inliers) >= (len(inlier_points_name) - 2):
                        inlier_points_name = inliers
                        outlier_points_name = outliers
                        inlier_points_name.append(p1['name'])
                        inlier_points_name.append(p2['name'])
                        err_avg = str(current_err_avg)

    # raise NotImplementedError
    return inlier_points_name, outlier_points_name


if __name__ == "__main__":
    input_points = [{'name': 'a', 'value': (0.0, 1.0)}, {'name': 'b', 'value': (2.0, 1.0)},
                    {'name': 'c', 'value': (3.0, 1.0)}, {'name': 'd', 'value': (0.0, 3.0)},
                    {'name': 'e', 'value': (1.0, 2.0)}, {'name': 'f', 'value': (1.5, 1.5)},
                    {'name': 'g', 'value': (1.0, 1.0)}, {'name': 'h', 'value': (1.5, 2.0)}]
    t = 0.5
    d = 3
    k = 100
    inlier_points_name, outlier_points_name = solution(input_points, t, d, k)  # TODO
    assert len(inlier_points_name) + len(outlier_points_name) == 8
    f = open('./results/task1_result.txt', 'w')
    f.write('inlier points: ')
    for inliers in inlier_points_name:
        f.write(inliers + ',')
    f.write('\n')
    f.write('outlier points: ')
    for outliers in outlier_points_name:
        f.write(outliers + ',')
    f.close()
