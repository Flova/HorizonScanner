
import cv2
import numpy as np
import time
import imutils
import math


class BoatDetector(object):
    def __init__(self):
        # Placeholders
        self.cap = None
        self._video_input = ""
        self._last_frame_dog = None

    def set_video_input(self, video_input):
        self._video_input = video_input
        self._cap = cv2.VideoCapture(self._video_input)  # "/home/florian/Projekt/BehindTheHorizon/data/.mp4"

    def _draw_mask(self, image, mask, color, opacity=0.5):
            # Make a colored image
            colored_image = np.zeros_like(image)
            colored_image[:, :] = tuple(np.multiply(color, opacity).astype(np.uint8))

            # Compose debug image with lines
            return cv2.add(cv2.bitwise_and(
                image,  image, mask=255-mask),
                cv2.add(colored_image*opacity, image*(1-opacity), mask=mask).astype(np.uint8))

    def run_detector_on_video_input(self, video_input=None):
        if video_input is not None:
            self.set_video_input(video_input)

        while(True):
            # Capture frame-by-frame
            ret, frame = self._cap.read()

            # Resize frame
            frame = cv2.resize(frame, (1200,800)) # TODO keep aspect ratio

            # Run detection on frame
            self.analyse_image(frame, history=True)

             # Show images
            cv2.imshow('ROI', self.roi_view)
            cv2.imshow('ROT', self.rotated)
            cv2.imshow('ROI MEAN', self.roi_mean)
            cv2.imshow('SPECTRUM', self.magnitude_spectrum)
            cv2.imshow('DOG', self.dog)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self._cap.release()
        cv2.destroyAllWindows()

    def analyse_image(self, image, roi_height=10, k_mean_width=5, history=True):
        # Make grayscale version
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binary image in which the horizon points are placed
        points = np.zeros_like(gray)

        k_mean_stepsize = 10

        for i in range(0, int(gray.shape[1] - k_mean_width), k_mean_stepsize):
            # Get vertical image slice as float array
            Z = np.float32(gray[:, i:i + k_mean_width])

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
        vx, vy, x0, y0 = cv2.fitLine(np.argwhere(points == 255), cv2.DIST_L1, 0, 0.005, 0.01)

        """ Debug line printout
        cv2.line(
            gray,
            (int(y0 + -500 * vy) , int(x0 + -500 * vx)),
            (int(y0 + 500 * vy) , int(x0 + 500 * vx)),
            (0,0,255),
            2)
        """

        # Rotate image using imutils TODO do this in opencv
        rotated = imutils.rotate(image, math.degrees(math.atan(vx/ vy)))

        # Crop roi out of rotated image
        roi = rotated[
            max(
                int(x0 - roi_height // 2),
                0)
            :
            min(
                int(x0 + roi_height // 2),
                rotated.shape[0]),
            :]

        # Get gray roi
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Calculate a mean in the vertical direction
        roi_mean = np.mean(gray_roi, axis=0).astype(np.uint8).reshape(1,1200)
        # Repeat for viz
        roi_mean = np.repeat(roi_mean, 60, axis=0)

        # Make fft
        f = np.fft.fft2(roi_mean)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = (20*np.log(np.abs(fshift))).astype(np.uint8)

        # Calculate difference of gaussians
        blur_large = cv2.blur(roi_mean, (51,1)).astype(np.float)
        blur_small = cv2.blur(roi_mean, (31,1)).astype(np.float)
        dog = blur_small - blur_large

        # Enshure all values are above 0
        dog[dog < 0] = 0

        # Scale image to uint8 scale
        dog = dog*255

        # Convert image
        dog = dog.astype(np.uint8)

        # Get K for the complementary filter
        K = 0.8

        # Calculate time based low pass using the complementary filter
        if history and self._last_frame_dog is not None:
            dog = (self._last_frame_dog * K + dog * (1 - K)).astype(np.uint8)

        # Scale roi view
        self.roi_view = cv2.resize(roi, (1200,40))

        self.roi_mean = roi_mean
        self.rotated = rotated
        self.magnitude_spectrum = magnitude_spectrum
        self.dog = dog

        if history:
            self._last_frame_dog = self.dog.copy()



if __name__ == "__main__":
    bt = BoatDetector()
    bt.run_detector_on_video_input("/home/florian/Projekt/BehindTheHorizon/data/VID_20180818_063412.mp4")

