
import cv2
import numpy as np
import time
import math


class HorizonDetector(object):
    """
    Abstract definition of a horizon detector
    """
    def __init__(self, params):
        self._params = params

    def get_horizon(self):
        """
        :returns: line_slope_x, line_slope_y, x, y, confidence
        """
        raise NotImplementedError

class KMeanHorizon(HorizonDetector):
    def __init__(self, params):
        super().__init__(params)

    def get_horizon(self, image):
        # Load params
        k_mean_stepsize = self._params['k_mean_stepsize']
        k_mean_width = self._params['k_mean_width']

        # Make gray image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binary image in which the horizon points are placed
        points = np.zeros_like(gray_image)


        # Iterate over vertical image slices
        for i in range(0, int(image.shape[1] - k_mean_width), k_mean_stepsize):
            # Get vertical image slice as float array
            Z = np.float32(image[:, i:i + k_mean_width])

            # K-Means termination settings
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # Number of classes
            K = 2
            # K-Means calculation
            ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

            # Determine which class is the sky
            if label[0] != 1:
                # Invert if the sky is not class 1
                label = np.invert(label)

            # Weired bug fix
            if (int(np.count_nonzero((label))) == 400):
                continue

            # Determine how many sky pixels are in the slice and approximate them as the y coordinate
            point = (i, int(np.count_nonzero((label))))

            # Draw horizon point in map
            cv2.circle(points, point, 1, 255, -1)    # TODO  use list of points instead

        # Fit a RANSEC like line in the horizon point map  (default params)
        line_slope_x, line_slope_y, line_base_x, line_base_y = cv2.fitLine(np.argwhere(points == 255), cv2.DIST_L1, 0, 0.005, 0.01)

        confidence = 1 # TODO find better confidence metric

        return line_slope_x, line_slope_y, line_base_x, line_base_y, confidence
