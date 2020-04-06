
import cv2
import numpy as np
import time
import math


class ROIBoatFinder(object):
    """
    Abstract definition of a roi boat finder
    """
    def __init__(self, params):
        self._params = params

    def find_boats_in_roi(self, roi):
        raise NotImplementedError


class DOGBoatFinder(ROIBoatFinder):
    def __init__(self, params):
        super().__init__(params)

    def find_boats_in_roi(self, roi):
        # Get params
        big_kernel = self._params['boat_finder_dog_big_kernel']
        small_kernel = self._params['boat_finder_dog_small_kernel']

        # Get gray roi
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Calculate a mean in the vertical direction
        roi_mean = np.mean(gray_roi, axis=0).astype(np.uint8).reshape(1,roi.shape[1])

        # Make fft  (not used currently)
        f = np.fft.fft2(roi_mean)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = (20*np.log(np.abs(fshift))).astype(np.uint8)


        # Calculate difference of gaussians
        blur_large = cv2.blur(roi_mean, (big_kernel,1)).astype(np.float)
        blur_small = cv2.blur(roi_mean, (small_kernel,1)).astype(np.float)
        dog = blur_small - blur_large

        # Ensure all values are above 0
        dog[dog < 0] = 0

        # Scale image to uint8 scale
        dog = dog * 255

        # Convert image
        dog = dog.astype(np.uint8)

        # Show debug images
        if self._params['debug']:
            # Repeat for viz
            roi_mean = np.repeat(roi_mean, 60, axis=0)
            cv2.imshow('ROI MEAN', roi_mean)
            cv2.imshow('SPECTRUM', magnitude_spectrum)

        return dog


class GradientBoatFinder(ROIBoatFinder):
    def __init__(self, params):
        super().__init__(params)

    def find_boats_in_roi(self, roi):
        gain = self._params['gradient_gain']

        # Get gray roi
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Exec horizontal sobel filter
        sobelx64f = cv2.Sobel(gray_roi,cv2.CV_64F,1,0,ksize=3)

        # Normalize values back to uint8 ranges
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_8u = np.uint8(abs_sobel64f) * gain

        # Pull the maximum gradient vertically
        sobel = np.mean(sobel_8u, axis=0).reshape(1, -1).astype(np.uint8)

        return sobel

