
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


class RoiCombinedHorizon(HorizonDetector):
    def __init__(self, params):
        super().__init__(params)
        self.invalid_horizon = (-1,-1,-1,-1,-1)

    def get_horizon(self, image):
        start = time.time()
        if self._params['horiz_det_use_roi_det']:
            start =time.time()
            ROI=self.detect_roi(image)
            image = image[ROI[0]:ROI[1],ROI[2]:ROI[3]]
            end=time.time()
            if self._params['horiz_det_profile']: print("ROI det:",end-start)

        vx,vy,x,y, confidence = self.detect_horizon(image)

        if self._params['horiz_det_use_roi_det']: # in case ROI is used, correct y coord
            y = y + ROI[0]

        confidence = 1 #TODO find metric
        end=time.time()
        if self._params['horiz_det_profile']: print("horizont:",end-start)

        return vx,vy,x,y, confidence


    def detect_roi(self, image):
        #create horizontal regions of the image with 50% overlap
        regionSplitsCount = self._params['horiz_det_roi_det_horizontal_regions']
        regionSubSplitsCount = regionSplitsCount+2 # one at the top , one at the bottom of the image
        imgHeight = image.shape[0]
        regionStepSize = imgHeight/(regionSubSplitsCount-1)

        curYval=0
        regionsMeanAndCovar=[]
        regionsImgs=[]
        regions=[]
        for i in range(regionSubSplitsCount):
            regionSplitYStart=int(curYval-regionStepSize)
            regionSplitYEnd=int(curYval+regionStepSize)
            regionSplitYStart = max(regionSplitYStart,0)
            regionSplitYEnd = min(regionSplitYEnd,imgHeight)
            region = [regionSplitYStart,regionSplitYEnd,0,image.shape[1]]
            regions.append(region)

            curYval = curYval+regionStepSize
            regionImg = image[region[0]:region[1],region[2]:region[3]]
            if self._params['debug']:
                regionsImgs.append(regionImg)

            regionColorMean = self.getColorMeans(regionImg)
            regionColorCovar = self.getColorCovar(regionImg)
            regionsMeanAndCovar.append([regionColorMean,regionColorCovar])


        # Calc distance of each subRegionDists[i] to subRegionDists[i+1]
        subRegionDists = []
        for i in range(regionSubSplitsCount-1):
            subRegionDist = self.calc_bhattacharyya_dist(
                regionsMeanAndCovar[i][0],regionsMeanAndCovar[i+1][0],
                regionsMeanAndCovar[i][1],regionsMeanAndCovar[i+1][1])
            subRegionDists.append(subRegionDist)

        # Calc distance of each region[i] to subRegionDists[i+1] and subRegionDists[i-1]
        regionDists = []
        for i in range(regionSplitsCount):
            regionDist = subRegionDists[i] + subRegionDists[i+1]
            regionDists.append(regionDist)

        #Most propable horizon area is at max distance to previous and next region
        regionMaxDistIdx = regionDists.index(max(regionDists))+1
        maxDistRegion = regions[regionMaxDistIdx]

        if self._params['horiz_det_roi_det_use_extended_region']:
            regionMaxDistPrev =  regionDists[regionMaxDistIdx-1-1]
            regionMaxDistNext =  regionDists[regionMaxDistIdx+1-1]
            if regionMaxDistPrev > regionMaxDistNext:
                extendedRegionIdx = regionMaxDistIdx-1
            else:
                extendedRegionIdx = regionMaxDistIdx+1

            extendedRegion = regions[extendedRegionIdx]

            ROI = [min(maxDistRegion[0],extendedRegion[0]),max(maxDistRegion[1],extendedRegion[1]),0,image.shape[1]]
        else:
            ROI = [maxDistRegion[0],maxDistRegion[1],0,image.shape[1]]

        if self._params['debug']:
            boarder = 30
            regionsSpacer = 20
            regionDebugImg = np.zeros((regionSubSplitsCount*(int(regionStepSize*2)+regionsSpacer)+2*boarder,image.shape[1]+2*boarder,3),dtype=np.uint8)
            ypos = boarder
            xpos = boarder
            for i in range(regionSubSplitsCount):
                regionDebugImg[ypos:ypos+regionsImgs[i].shape[0], xpos:xpos+regionsImgs[i].shape[1]] = regionsImgs[i]
                if i >= 1 and i <= len(regionDists):
                    pos =(regionDebugImg.shape[1]//2,ypos+int(regionStepSize))
                    regionDebugImg = cv2.putText(
                        img=regionDebugImg, text=str(round(float(regionDists[i-1]), 2)),org=pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(0, 255, 0) ,thickness=2)
                    if i == regionMaxDistIdx:
                        cv2.rectangle(regionDebugImg, (xpos,ypos), (xpos+regionsImgs[i].shape[1],ypos+regionsImgs[i].shape[0]), (0,255,0), thickness=5)
                    if self._params['horiz_det_roi_det_use_extended_region'] and i == extendedRegionIdx:
                        cv2.rectangle(regionDebugImg, (xpos,ypos), (xpos+regionsImgs[i].shape[1],ypos+regionsImgs[i].shape[0]), (0,255,255), thickness=5)
                ypos=ypos+regionsSpacer+int(regionStepSize*2)

            scale_percent = 50 # percent of original size
            width = int(regionDebugImg.shape[1] * scale_percent / 100)
            height = int(regionDebugImg.shape[0] * scale_percent / 100)
            dim = (width, height)
            regionDebugImg = cv2.resize(regionDebugImg, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('region dists', regionDebugImg)

        return ROI

    def getColorMeans(self, inpic):
        r,g,b=cv2.split(inpic)
        X = np.array([
            [np.mean(r)],
            [np.mean(g)],
            [np.mean(b)]])
        return X

    def getColorCovar(self, inpic):
        r,g,b=cv2.split(inpic)
        r=r.flat
        g=g.flat
        b=b.flat
        X = np.stack((r,g,b), axis=0)
        a = np.cov(X)
        return a

    def calc_bhattacharyya_dist(self, m1, m2, c1, c2):
        #not the correct formula but still works according to the paper and some experiments
        #Dist(R1,R2)=(mean1-mean2)^T*((Cov1+Cov2)/2.0)^-1 *(mean1-mean2)
        meandiff = m1-m2
        cov = (c1+c2)/2
        dist = np.matmul(np.linalg.inv(cov),meandiff)
        dist = np.matmul(meandiff.T,dist)
        #dist = dist*1/8
        #dist = dist + 0.5*np.log((np.linalg.det(covdiff))/np.sqrt(np.linalg.det(c1)*np.linalg.det(c2)) )
        return dist

    def detect_horizon(self, image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        """
        #sharpe image
        start=time.time()
        blured = cv2.GaussianBlur(gray, (0,0), 15)
        gray = cv2.addWeighted(gray, 1.5, blured, -0.5, 0)
        #cv2.imshow('gray', gray)
        #cv2.imshow('sharpened', sharpened)
        end=time.time()
        if self._params['horiz_det_profile']: print( "sharpe:",end-start)
        """

        #multiscale edge processing
        start=time.time()
        edgeimage = np.zeros(gray.shape,dtype=np.uint8)
        medianscales = [5,11,15,21]

        weight = np.float32(1.0/(len(medianscales)))
        cannyMin = self._params['horiz_det_canny_min']  # below will not be considered as edge
                                                        # between will be considered as edge, if connected to strong edge
        cannyMax = self._params['horiz_det_canny_max']  # above will be considered as strong edge
        for oneMedianScale in medianscales:
            if oneMedianScale < 1:
                median = gray
            else:
                median = cv2.medianBlur(gray,oneMedianScale)
            canny = cv2.Canny(median,cannyMin,cannyMax) * weight
            #cannyweighted = astype(canny * weight)
            edgeimage = cv2.add(edgeimage, canny.astype(np.uint8))

        end=time.time()
        if self._params['horiz_det_profile']: print( "canny:",end-start)

        """ additional sobel for weak / foggy horizont => doesnt work yet
        ksize = 5
        kw = dict(ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, **kw)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        #abs_grad_y_float = np.float32(abs_grad_y) * weight
        threshold=150
        ret, abs_grad_y_threshed = cv2.threshold(abs_grad_y, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('abs_grad_y_threshed', abs_grad_y)
        edgeimage = cv2.add(edgeimage, abs_grad_y_threshed * np.float32(0.3))
        """

        if self._params['debug']:
            cv2.imshow('edgeimage', edgeimage)

        if np.max(edgeimage) == 0:
            print("edgeimage max == 0")
            return self.invalid_horizon

        #Only keep strongest edges
        edges_threshold=self._params['horiz_det_edge_threshold']
        ret, threshed = cv2.threshold(edgeimage, edges_threshold, 255, cv2.THRESH_BINARY)
        if self._params['debug']:
            cv2.imshow('threshed', threshed)
            #cv2.waitKey(0)

        if np.max(threshed) == 0:
            print("threshed max == 0")
            return self.invalid_horizon

        #Inital horizont guess via hough transform
        houghThresh = threshed.shape[0]//8
        houghAngularRes = np.deg2rad(0.5)
        houghMaxAngleHorizon=30
        #propabilistic approach is not as accurate and fast as normal hough (due to angle min max @ normal)
        """
        print("----")
        start=time.time()
        minLineLength = threshed.shape[0]//1.5
        maxLineGap = threshed.shape[0]//4

        lines = cv2.HoughLinesP(image=threshed,rho=1,theta=houghAngularRes,threshold=houghThresh,minLineLength=minLineLength,maxLineGap=maxLineGap)
        if lines is None:
            print("lines == 0")
            return self.invalid_horizon
        for oneline in lines:
            for x1,y1,x2,y2 in oneline:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
                break # only take first line
            break # only take first line

        end=time.time()
        if self._params['horiz_det_profile']: print( end-start)
        """
        start=time.time()
        lines = cv2.HoughLines(image=threshed,rho=1,theta=houghAngularRes,threshold=houghThresh,min_theta=np.deg2rad(90-houghMaxAngleHorizon),max_theta=np.deg2rad(90+houghMaxAngleHorizon))
        if lines is None:
            print("lines == 0")
            return self.invalid_horizon
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            if self._params['debug']:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
        end=time.time()
        if self._params['horiz_det_profile']: print( "hough:",end-start)

        #calc residual / error of every pixel in edge image to hough line
        start=time.time()
        x1,y1,x2,y2 in lines[0]#get line description by using the two points of the hough line
        denom=np.sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1))
        thresholdedNonZero = cv2.findNonZero(threshed)
        residuals=np.zeros((len(thresholdedNonZero)))
        for i,onePix in enumerate(thresholdedNonZero):
            x3 = int(onePix[0][0])
            y3 = int(onePix[0][1])
            d1 = abs((y2-y1)*x3-(x2-x1)*y3+x2*y1-y2*x1)/denom
            residuals[i]=d1
        end=time.time()
        if self._params['horiz_det_profile']: print( "res:",end-start)

        #median filter thresholded pixels by residual / error => only keep pts with low error
        start=time.time()
        q1=np.quantile(residuals, self._params['horiz_det_thresholded_q1']) #calc q1 quantile for filtering
        q1FilteredHorizontPts = thresholdedNonZero[residuals < q1]
        end=time.time()
        if self._params['horiz_det_profile']: print( "medianfilter:",end-start)

        #fit final horizontal line by calc line through median filted points
        start=time.time()
        [vx,vy,x,y] = cv2.fitLine(
            points=q1FilteredHorizontPts,
            distType=cv2.DIST_L1,
            param=0,
            reps=0.01,
            aeps=0.01)
        end=time.time()
        if self._params['horiz_det_profile']: print( "fitline:",end-start)

        # Now find two extreme points on the line to draw line
        #lefty = int((-x*vy/vx) + y)
        #righty = int(((gray.shape[1]-x)*vy/vx)+y)
        #cv2.line(image,(gray.shape[1]-1,righty),(0,lefty),(255,0,0),1)
        #cv2.imshow('line', image)

        confidence = 1

        print(vx,vy,x,y, confidence)

        return vx,vy,x,y, confidence


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
