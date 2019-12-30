
import cv2
import numpy as np
import time
import imutils
import math


def draw_mask(image, mask, color, opacity=0.5):
        # Make a colored image
        colored_image = np.zeros_like(image)
        colored_image[:, :] = tuple(np.multiply(color, opacity).astype(np.uint8))

        # Compose debug image with lines
        return cv2.add(cv2.bitwise_and(
            image,  image, mask=255-mask),
            cv2.add(colored_image*opacity, image*(1-opacity), mask=mask).astype(np.uint8))

cap = cv2.VideoCapture("/home/florian/Projekt/BehindTheHorizon/data/.mp4")

last_frame = None

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.resize(frame, (1200,800))

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros_like(gray)

    points = np.zeros_like(gray).astype(np.uint8)

    width = 5
    for i in range(0, int(1*gray.shape[1] - width + 1), 10):
        Z = np.float32(gray[:,i:i + width])

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        if label[0] != 1:
            label = np.invert(label)

        if (int(np.count_nonzero((label))) == 400):
            continue

        point = (i, int(np.count_nonzero((label))))
        cv2.circle(gray, point, 3, 0, -1)
        cv2.circle(points, point, 1, 255, -1)
        canvas[:, i] = label.reshape(gray.shape[0])



    canvas = canvas.astype(np.uint8) * 255
q
    vx, vy, x0, y0 = cv2.fitLine(np.argwhere(points == 255), cv2.DIST_L1, 0, 0.005, 0.01)  # 2 = CV_DIST_L2

    cv2.line(gray,(int(y0 + -500 * vy) , int(x0 + -500 * vx)), (int(y0 + 500 * vy) , int(x0 + 500 * vx)),(0,0,255),2)

    rotated = imutils.rotate(frame, math.degrees(math.atan(vx/ vy)))

    roi_height = 10

    roi = rotated[max(int(x0 - roi_height // 2), 0) : min(int(x0 + roi_height // 2), rotated.shape[0]), :]

    roi_mean = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), axis=0).astype(np.uint8).reshape(1,1200)

    roi_mean = np.repeat(roi_mean, 60, axis=0)

    f = np.fft.fft2(roi_mean)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = (20*np.log(np.abs(fshift))).astype(np.uint8)

    blur1 = cv2.blur(roi_mean, (51,1)).astype(np.float)

    blur2 = cv2.blur(roi_mean, (31,1)).astype(np.float)

    dog = blur2 - blur1

    dog[dog < 0] = 0

    dog = dog*255

    dog = dog.astype(np.uint8)

    K = 0.8

    if last_frame is not None:
        dog = (last_frame * K + dog * (1 - K)).astype(np.uint8)

    roi_view = cv2.resize(roi, (1200,40))

    cv2.imshow('ROI', roi_view)
    cv2.imshow('ROT', rotated)
    cv2.imshow('ROI MEAN', roi_mean)
    cv2.imshow('SPECTRUM', magnitude_spectrum)
    cv2.imshow('DOG', dog)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    last_frame = dog.copy()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

