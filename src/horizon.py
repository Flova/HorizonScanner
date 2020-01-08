
import cv2
import numpy as np
import time
import math


import matplotlib.pyplot as plt

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
        
        
    def get_horizon(self, image):

        X = np.array([
            [1],
            [2],
            [3]])
        A = np.eye(3)
        B = np.array([
            [1],
            [2],
            [3]])

        r = np.matmul(A,B)
        r = np.matmul(X.T,r)
        self.detect_roi(image)
    

    def getColorMeans(self, inpic):
        r,g,b=cv2.split(inpic)
        #X=np.array([np.mean(r),np.mean(g),np.mean(b)])
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
        a= np.cov(X)
        #b = cv2.calcCovarMatrix([r,g,b])
        return a

    def calc_bhattacharyya_dist(self, m1, m2, c1, c2):
        #D(R1,R2)=(mean1-mean2)^T*(S1-S2)^-1 *(m1-m2)
        meandiff = m1-m2
        covdiff = (c1+c2)/2
        dist = np.matmul(np.linalg.inv(covdiff),meandiff)
        dist = np.matmul(meandiff.T,dist)
        #dist=meandiff.T*(np.linalg.inv(covdiff)*meandiff)
        return dist
 
    def detect_roi(self, image):


        regionSplitsCount = 8
        overlappercentage = 0.5
        imgHeight= image.shape[0]
        regionSplitHeight = imgHeight / regionSplitsCount
        print(regionSplitHeight)

        curYval=0
        regionsMeanAndCovar=[]
        regionsImgs=[]
        for i in range(regionSplitsCount):
            regionSplitYStart=int(curYval-regionSplitHeight-regionSplitHeight*overlappercentage) 
            regionSplitYEnd=int(curYval+regionSplitHeight+regionSplitHeight*overlappercentage)
            regionSplitYStart = max(regionSplitYStart,0)
            regionSplitYEnd = min(regionSplitYEnd,imgHeight)
            print("-----")                        
            print(regionSplitYStart)
            print(regionSplitYEnd)
            #oneImgSplit = inpic[regionSplitYStart:regionSplitYEnd,0:inpic.shape[1]]
            #print(oneImgSplit.shape)
            curYval = curYval+regionSplitHeight
            #imgSplitted.append(oneImgSplit)
            #ySplits.append([regionSplitYStart,regionSplitYEnd])
            regionImg = image[regionSplitYStart:regionSplitYEnd,0:image.shape[1]]
            regionsImgs.append(regionImg)
            cv2.imshow('regionImg', regionImg)
            
            regionColorMean = self.getColorMeans(regionImg)
            regionColorCovar = self.getColorCovar(regionImg)
            regionsMeanAndCovar.append([regionColorMean,regionColorCovar])


        print("asf")
        i=1
        while True:
            d1 = self.calc_bhattacharyya_dist(
                regionsMeanAndCovar[i-1][0],regionsMeanAndCovar[i][0],
                regionsMeanAndCovar[i-1][1],regionsMeanAndCovar[i][1])
            d2 = self.calc_bhattacharyya_dist(
                regionsMeanAndCovar[i+1][0],regionsMeanAndCovar[i][0],
                regionsMeanAndCovar[i+1][1],regionsMeanAndCovar[i][1])
            d=d1+d2
            print("--------")
            print(d1)
            print(d2)
            print(d)
            i=i+1
            if i == regionSplitsCount-1:
                break

        fig, axes = plt.subplots(regionSplitsCount, 1) #, sharex='row', sharey='row'
        for i in range(regionSplitsCount):
            axes[i].imshow(regionsImgs[i])
        plt.show()
        cv2.waitKey(0)

    def detect_horizon(self, image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #sharpe image
        blured = cv2.GaussianBlur(gray, (0,0), 15)
        sharpened = cv2.addWeighted(gray, 1.5, blured, -0.5, 0)
        #cv2.imshow('gray', gray)
        #cv2.imshow('sharpened', sharpened)

        #multiscale edge processing
        edgeimage = np.zeros(gray.shape,dtype=np.float)
        medianscales = [5,11,15,21]
        
        weight = 1.0/len(medianscales)
        cannyMin = 20 # pixelvalues below will be thrown out
                      # pixelvalues between will be considered as edge, if connected to strong edge
        cannyMax = 60 # pixelvalues above will be considered as strong edge 
        for oneMedianScale in medianscales:
            if oneMedianScale < 1:
                median = sharpened
            else:
                median = cv2.medianBlur(sharpened,oneMedianScale)
            canny = cv2.Canny(median,cannyMin,cannyMax)
            edgeimage = cv2.add(edgeimage, canny * weight)
            #cv2.imshow('median', median)
            #cv2.imshow('canny', canny)
            #cv2.waitKey(0)
                
        edgeimage = edgeimage.astype(np.uint8)
        cv2.imshow('edgeimage', edgeimage) 

        #Only keep strongest edges
        threshold=90
        ret, threshed = cv2.threshold(edgeimage, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('threshed', threshed) 
        cv2.waitKey(0)

        #Inital horizont guess via hough transform
        minLineLength = 100
        maxLineGap = 50
        houghThresh=1
        lines = cv2.HoughLinesP(image=threshed,rho=1,theta=np.pi/(180*2),threshold=houghThresh,minLineLength=minLineLength,maxLineGap=maxLineGap)
        for oneline in lines:
            for x1,y1,x2,y2 in oneline:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
                break # only take first line
            break # only take first line

        #calc residual / error of every pixel to hough line
        threshedPixelList=[]
        distances=[]
        x1,y1,x2,y2 in lines[0]#get line description by using the two points of the hough line
        p1=np.array([x1,y1])
        p2=np.array([x2,y2])
        for onePix in cv2.findNonZero(threshed):
            threshedPixelList.append(onePix[0])
            p3=np.array([onePix[0][0],onePix[0][1]])
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1) 
            distances.append(d)
        
        #median filter thresholded pixels by residual / error => only keep pts with low error
        q1=np.quantile(distances, 0.2) #calc q1 quantile for filtering
        medianFilteredHorizPts = []
        for idx, oneDist in enumerate(distances):
            if oneDist <= q1:
                medianFilteredHorizPts.append(threshedPixelList[idx])
        medianFilteredHorizPts = np.array(medianFilteredHorizPts)

        #fit final horizontal line by calc least square 
        [vx,vy,x,y] = cv2.fitLine(
            points=medianFilteredHorizPts, 
            distType=cv2.DIST_L1,
            param=0, 
            reps=0.01, 
            aeps=0.01)

        # Now find two extreme points on the line to draw line
        lefty = int((-x*vy/vx) + y)
        righty = int(((gray.shape[1]-x)*vy/vx)+y)
        cv2.line(image,(gray.shape[1]-1,righty),(0,lefty),(255,0,0),1)
        cv2.imshow('line', image)
        cv2.waitKey(0) 
        return 1


a = RoiCombinedHorizon(1)
images=["horizontest2.jpg","horizontest4.jpg","horizontest5.png"]
img=cv2.imread(images[1], cv2.IMREAD_COLOR)
img = cv2.resize(img, (800,300))

a.get_horizon(img)


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
                # Invert
                # if the sky is not class 1
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
