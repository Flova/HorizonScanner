
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ROI=self.detect_roi(image)
        ROIImage = image[ROI[0]:ROI[1],ROI[2]:ROI[3]]

        vx,vy,x,y = self.detect_horizon(ROIImage)

        y = y + ROI[0]

        lefty = int((-x*vy/vx) + y)
        righty = int(((image.shape[1]-x)*vy/vx)+y)
        cv2.line(image,(image.shape[1]-1,righty),(0,lefty),(255,0,0),1)
        cv2.imshow('line', image)

        cv2.waitKey(0) 
        confidence = 1

        return vx,vy,x,y, confidence 

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
        covdiff = (c1+c2)/2
        dist = np.matmul(np.linalg.inv(covdiff),meandiff)
        dist = np.matmul(meandiff.T,dist)
        #dist = dist*1/8
        #dist = dist + 0.5*np.log((np.linalg.det(covdiff))/np.sqrt(np.linalg.det(c1)*np.linalg.det(c2)) )

        return dist
 
    def detect_roi(self, image):

        #create horizontal region slides of the image with 50% overlap
        regionSplitsCount = 4
        regionSubSplitsCount = regionSplitsCount+2 # one at the top , one at the bottom of the image
        imgHeight = image.shape[0] 
        regionStepSize = imgHeight/(regionSubSplitsCount-1)
 
        curYval=0
        regionsMeanAndCovar=[]
        #regionsImgs=[]
        regions=[]
        for i in range(regionSubSplitsCount):
            regionSplitYStart=int(curYval-regionStepSize) 
            regionSplitYEnd=int(curYval+regionStepSize)
            regionSplitYStart = max(regionSplitYStart,0)
            regionSplitYEnd = min(regionSplitYEnd,imgHeight)
            region = [regionSplitYStart,regionSplitYEnd,0,image.shape[1]]
            regions.append(region)
            #print("-----")                        
            #print(regionSplitYStart)
            #print(regionSplitYEnd)
            #oneImgSplit = inpic[regionSplitYStart:regionSplitYEnd,0:inpic.shape[1]]
            #print(oneImgSplit.shape)
            curYval = curYval+regionStepSize
            #imgSplitted.append(oneImgSplit)
            #ySplits.append([regionSplitYStart,regionSplitYEnd])
            regionImg = image[region[0]:region[1],region[2]:region[3]]
            #regionsImgs.append(regionImg)
            #regionsImgs.append(cv2.cvtColor(regionImg, cv2.COLOR_HSV2RGB))
            #cv2.imshow('regionImg', regionImg)
            
            regionColorMean = self.getColorMeans(regionImg)
            regionColorCovar = self.getColorCovar(regionImg)
            regionsMeanAndCovar.append([regionColorMean,regionColorCovar])

        #print("-------------")
        subRegionDists = []
        for i in range(regionSubSplitsCount-1):
            subRegionDist = self.calc_bhattacharyya_dist(
                regionsMeanAndCovar[i][0],regionsMeanAndCovar[i+1][0],
                regionsMeanAndCovar[i][1],regionsMeanAndCovar[i+1][1])
            subRegionDists.append(subRegionDist)
            #print(subRegionDist)
        
        #print("-------------")
        regionDists = []
        for i in range(regionSplitsCount):
            regionDist = subRegionDists[i] + subRegionDists[i+1]
            regionDists.append(regionDist)
            #print(regionDist)
        

        #fig, axes = plt.subplots(regionSubSplitsCount, 1) #, sharex='row', sharey='row'
        #for i in range(regionSubSplitsCount):
        #    axes[i].imshow(regionsImgs[i])
        #plt.show()
        #cv2.waitKey(0)

        regionMaxDistIdx = regionDists.index(max(regionDists))+1
        ROI = regions[regionMaxDistIdx]
        return ROI

    def detect_horizon(self, image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #sharpe image
        blured = cv2.GaussianBlur(gray, (0,0), 15)
        sharpened = cv2.addWeighted(gray, 1.5, blured, -0.5, 0)
        cv2.imshow('gray', gray)
        #cv2.imshow('sharpened', sharpened)

        #multiscale edge processing
        edgeimage = np.zeros(gray.shape,dtype=np.float)
        medianscales = [5,11,15,21]
        
        weight = 1.0/len(medianscales)
        cannyMin = 40 # pixelvalues below will be thrown out
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
        
        if np.max(edgeimage) == 0:
            print("edgeimage max == 0")
            return


        #Only keep strongest edges
        threshold=80
        ret, threshed = cv2.threshold(edgeimage, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('threshed', threshed) 
        #cv2.waitKey(0)

        if np.max(threshed) == 0:
            print("threshed max == 0")
            return

        #Inital horizont guess via hough transform
        minLineLength = 80
        maxLineGap = 50
        houghThresh=5
        lines = cv2.HoughLinesP(image=threshed,rho=1,theta=np.pi/(180*2),threshold=houghThresh,minLineLength=minLineLength,maxLineGap=maxLineGap)
        if lines is None:
            print("lines == 0")
            return
        for oneline in lines:
            for x1,y1,x2,y2 in oneline:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
                break # only take first line
            break # only take first line

        #calc residual / error of every pixel in edge image to hough line
        threshedPixelList=[]
        residuals=[]
        x1,y1,x2,y2 in lines[0]#get line description by using the two points of the hough line
        p1=np.array([x1,y1])
        p2=np.array([x2,y2])
        for onePix in cv2.findNonZero(threshed):
            threshedPixelList.append(onePix[0])
            p3=np.array([onePix[0][0],onePix[0][1]])
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1) 
            residuals.append(d)
        
        #median filter thresholded pixels by residual / error => only keep pts with low error
        q1=np.quantile(residuals, 0.1) #calc q1 quantile for filtering
        medianFilteredHorizPts = []
        for idx, oneDist in enumerate(residuals):
            if oneDist <= q1:
                medianFilteredHorizPts.append(threshedPixelList[idx])
        medianFilteredHorizPts = np.array(medianFilteredHorizPts)

        #fit final horizontal line by calc line through median filted points
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

        return vx,vy,x,y


a = RoiCombinedHorizon(0)
images=["horizontest4.jpg","horizontest5.png","test6.jpg","test7.png","test8.png","test9.png"] # "horizontest.jpg",
for oneimg in images:
    img = cv2.imread(oneimg, cv2.IMREAD_COLOR)
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
