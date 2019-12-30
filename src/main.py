
import cv2
import numpy as np
import time
import imutils
import math
import os
import yaml
from horizon import KMeanHorizon
from roi_boat_finder import DOGBoatFinder


class BoatDetector(object):
    def __init__(self, params):
        # Placeholders
        self.cap = None
        self._video_input = ""
        self._last_frame_dog = None

        self._params = params

        self._horizon_detector = KMeanHorizon(self._params['horizon_detector'])
        self._roi_boat_finder = DOGBoatFinder(self._params['boat_finder'])

    def set_video_input(self, video_input):
        self._video_input = video_input
        self._cap = cv2.VideoCapture(self._video_input)

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
            roi, rotated, dog  = self.analyse_image(frame, roi_height=self._params['default_roi_height'], history=True)

            roi_view = cv2.resize(roi, (1200, 60))

            # Show images
            cv2.imshow('ROI', roi_view)
            cv2.imshow('ROT', rotated)
            # Repeat for viz
            dog_large = np.repeat(dog, 60, axis=0)
            cv2.imshow('DOG', dog_large)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self._cap.release()
        cv2.destroyAllWindows()

    def analyse_image(self, image, roi_height=10, horizon=None, history=True):
        # Make grayscale version
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the horizon in the image
        if horizon is None:
            line_slope_x, line_slope_y, line_base_x, line_base_y, _ = self._horizon_detector.get_horizon(image)
        else:
            line_slope_x, line_slope_y, line_base_x, line_base_y, _ = horizon

        # Rotate image using imutils TODO do this in opencv
        rotated = imutils.rotate(image, math.degrees(math.atan(line_slope_x/ line_slope_y)))

        # Crop roi out of rotated image
        roi = rotated[
            max(
                int(line_base_x - roi_height // 2),
                0)
            :
            min(
                int(line_base_x + roi_height // 2),
                rotated.shape[0]),
            :]

        dog = self._roi_boat_finder.find_boats_in_roi(roi)

        # Get K for the complementary filter
        K = self._params['complementary_filter_k']

        # Calculate time based low pass using the complementary filter
        if history and self._last_frame_dog is not None:
            dog = (self._last_frame_dog * K + dog * (1 - K)).astype(np.uint8)

        # Set last image to current image
        if history:
            self._last_frame_dog = dog.copy()

        return (
            roi,
            rotated,
            dog
        )

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")

    config_path = os.path.realpath(config_path)

    if not os.path.exists(config_path):
        print("No config file specified, see the 'example.config.yaml' in 'config' and save your version as 'config.yaml'!")

    with open(config_path, "r") as f:q
        params = yaml.safe_load(f)

    bt = BoatDetector(params)
    bt.run_detector_on_video_input(params['video_source'])
