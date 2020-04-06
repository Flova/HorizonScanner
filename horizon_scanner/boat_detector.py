#! /usr/bin/env python3

import cv2
import numpy as np
import time
import imutils
import math
import os
import yaml
from modules.horizon import KMeanHorizon, RoiCombinedHorizon
from modules.roi_boat_finder import DOGBoatFinder, GradientBoatFinder
from modules.candidate_finder import CandidateFinder
from modules import candidate_finder
from modules.candidate_matcher import HistogramMatcher


class BoatDetector(object):
    def __init__(self, params):
        # Placeholders
        self.cap = None
        self._video_input = ""
        self._last_frame_feature_map = None

        self._params = params

        self._horizon_detector = RoiCombinedHorizon(self._params['horizon_detector'])
        self._roi_boat_finder = GradientBoatFinder(self._params['boat_finder'])
        self._candidate_finder = CandidateFinder()
        self._candidate_matcher = HistogramMatcher()

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

        previous_rendered_candidates = []

        while(True):
            # Capture frame-by-frame
            ret, frame_buffer = self._cap.read()

            if frame_buffer is None:
                print("Video finished / source closed")
            else:
                frame = frame_buffer

            # Resize frame
            frame = cv2.resize(frame, (1200,800)) # TODO keep aspect ratio

            # Run detection on frame
            valid, roi, rotated, feature_map, binary_detections, candidates  = self.analyse_image(frame, roi_height=self._params['default_roi_height'], history=True)

            rendered_candidates = candidate_finder.render_candiates(roi, candidates)

            print(self._candidate_matcher.relocate_candidates(rendered_candidates, previous_rendered_candidates))

            if valid == True:
                roi_view = cv2.resize(roi, (1200, 60))

                # Draw candidates
                candidate_canvas = np.zeros_like(roi)
                if candidates:
                    candidate_canvas_part = np.zeros((20,0,3), dtype=np.uint8)
                    for index, candidate in enumerate(candidates):
                        candidate_canvas_part = np.concatenate((candidate_canvas_part, roi[:, candidate[0]: candidate[1], :]), axis=1)
                        candidate_canvas_part = np.concatenate((candidate_canvas_part, np.zeros((20,5,3), dtype=np.uint8)), axis=1)
                    candidate_canvas[:,:candidate_canvas_part.shape[1]-5,:] += candidate_canvas_part[:, :-5,:]

                # Repeat for viz
                feature_map_large = np.repeat(feature_map, 60, axis=0)
                binary_detections_large = np.repeat(binary_detections, 60, axis=0)
                combines_detections = np.concatenate((feature_map_large, binary_detections_large), axis=0)
                detections_with_roi = np.concatenate((roi_view, cv2.cvtColor(combines_detections, cv2.COLOR_GRAY2BGR)), axis=0)
                detections_with_roi = np.concatenate((detections_with_roi, cv2.resize(candidate_canvas, (1200,60))), axis=0)
                # Show images
                cv2.imshow('Detections', detections_with_roi)
                cv2.imshow('ROT', rotated)

                previous_rendered_candidates = rendered_candidates.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self._cap.release()
        cv2.destroyAllWindows()

    def analyse_image(self, image, roi_height=20, horizon=None, history=True):
        # Bring image to constant size
        image = cv2.resize(image, (1200,800))

        # Make grayscale version
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the horizon in the image
        start=time.time()
        if horizon is None:
            line_slope_x, line_slope_y, line_base_x, line_base_y, _ = self._horizon_detector.get_horizon(image)
        else:
            line_slope_x, line_slope_y, line_base_x, line_base_y, _ = horizon

        if line_base_x == -1: # invalid horizon
            return (False,0,0,0,0,[])

        """
        # Show debug image with the detected line
        horizonImg = image.copy()
        lefty = int((-line_base_x*line_slope_y/line_slope_x) + line_base_y)
        righty = int(((horizonImg.shape[1]-line_base_x)*line_slope_y/line_slope_x)+line_base_y)
        cv2.line(horizonImg,(horizonImg.shape[1]-1,righty),(0,lefty),(0,0,255),1)
        cv2.imshow('horizont', horizonImg)"""

        # Rotate image
        rotated =  self.rotate_and_center_horizon(image,line_slope_x, line_slope_y, line_base_x, line_base_y)#imutils.rotate(image, math.degrees(math.atan(line_slope_x/ line_slope_y)))

        # Crop roi out of rotated image
        horizon_y_pos=image.shape[0] / 2  # line_base_x
        roi = rotated[
            max(
                int(horizon_y_pos - roi_height // 2),
                0)
            :
            min(
                int(horizon_y_pos + roi_height // 2),
                rotated.shape[0]),
            :]

        start=time.time()

        # Get feature map
        feature_map = self._roi_boat_finder.find_boats_in_roi(roi)

        # Get K for the complementary filter
        K = self._params['complementary_filter_k']

        # Calculate time based low pass using the complementary filter
        if history and self._last_frame_feature_map is not None:
            feature_map = (self._last_frame_feature_map * K + feature_map * (1 - K)).astype(np.uint8)
        elif history:
            history_map = (feature_map * (1 - K)).astype(np.uint8)

        # Add median filter for noise suppression
        median_features = cv2.medianBlur(feature_map, 3)

         # Calculate mean of detections
        mean = np.mean(median_features, axis=1)

        # Add threshold
        _, median_features_thresh = cv2.threshold(median_features, int(mean + self._params['threshold']), 255, cv2.THRESH_BINARY)

        # Set last image to current image
        if history:
            self._last_frame_feature_map = feature_map.copy()

        candidates = self._candidate_finder.get_candidates(median_features_thresh)

        return (
            True,
            roi,
            rotated,
            median_features,
            median_features_thresh,
            candidates
        )

    def rotate_and_center_horizon(self,img,vx,vy,x,y):
        sx=img.shape[1]/2
        l=(sx-x)/vx
        sy=y+l*vy
        horizonCenterPt=(sx,sy)
        angle=np.rad2deg(np.tan(vy,vx))

        M_rotate = cv2.getRotationMatrix2D(horizonCenterPt,angle,1)
        M_translate2Center=np.zeros((2,3),np.float)
        M_translate2Center[0,2]=(img.shape[1]/2)-horizonCenterPt[0]
        M_translate2Center[1,2]=(img.shape[0]/2)-horizonCenterPt[1][0]
        M=M_rotate+M_translate2Center

        cols=img.shape[1]
        rows=img.shape[0]
        img_corr = cv2.warpAffine(img,M,(cols,rows))

        return img_corr

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")

    config_path = os.path.realpath(config_path)

    if not os.path.exists(config_path):
        print("No config file specified, see the 'example.config.yaml' in 'config' and save your version as 'config.yaml'!")

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    bt = BoatDetector(params)
    bt.run_detector_on_video_input(params['video_source'])

