import cv2

class HistogramMatcher(object):
    def __init__(self):
        pass

    def relocate_candidates(self, candidates, previous_candidates):
        if not previous_candidates or not candidates:
            return {}

        lst=[]
        for candi in candidates:
            hist = cv2.calcHist([candi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            lst.append(hist)

        olst=[]
        for candi in previous_candidates:
            hist = cv2.calcHist([candi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            olst.append(hist)


        convs = []
        for ni, a in enumerate(lst):
            hists = []
            for i, b in enumerate(olst):
                hists.append((ni, i, abs(cv2.compareHist(a, b, cv2.HISTCMP_CHISQR))))
            convs.extend(hists)

        convs = sorted(convs, key = lambda x: x[2], reverse=False)

        matched_cands = []
        matched_o_cands = []
        correlations = {}
        for x in convs:
            if not x[0] in matched_cands and not x[1] in matched_o_cands:
                matched_cands.append(x[0])
                matched_o_cands.append(x[1])
                correlations[x[1]] = x[0]

        return correlations
