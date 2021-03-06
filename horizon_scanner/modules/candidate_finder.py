
import cv2
import numpy as np
import time
import math


def render_candiates(roi, candidates):
    rendered_candidates = []
    for index, candidate in enumerate(candidates):
        rendered_candidates.append(roi[:, candidate[0]: candidate[1], :])
    return rendered_candidates


class CandidateFinder(object):
    def __init__(self):
        pass

    def get_candidates(self, threshold_feature_map):

        border_size = 10 # TODO Param
        kernel_size = (41,1)
        gap_less_feature_map = cv2.blur(threshold_feature_map, kernel_size)

        candidates = []

        # Init state
        previously_on_candidate = False

        for i in range(gap_less_feature_map.shape[1]):
            if gap_less_feature_map[0, i] != 0:
                if not previously_on_candidate:
                    previously_on_candidate = True
                    start_index = i
            else:
                if previously_on_candidate:
                    previously_on_candidate = False
                    stop_index = min(i - (kernel_size[0] // 2) + border_size, gap_less_feature_map.shape[1] - 1)
                    start_index = max(start_index + kernel_size[0] // 2 - border_size, 0)
                    candidates.append((start_index, stop_index))

        return candidates
