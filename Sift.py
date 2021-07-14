import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('TargetImage1.jpg')
myVid = cv2.VideoCapture('game.jfif')

st = None

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

def SIFT():
    # Initiate SIFT detector
    SIFT = cv2.xfeatures2d.SIFT_create()
    return SIFT

def BRIEF():
    # Initiate BRIEF descriptor
    BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    return BRIEF

def ORB():
    # Initiate ORB detector
    ORB = cv2.ORB_create()
    return ORB

_detect=SIFT()
_compute =SIFT()
kp1=_detect.detect(imgTarget,None)
kp1, des1=_compute.compute(imgTarget, kp1)
print(type(des1))
#kp1, des1 = sift.detectAndCompute(imgTarget, None)

# imgTarget1=cv2.drawKeypoints(imgTarget,kp1,None)
# cv2.imshow("tt",imgTarget1)
# cv2.waitKey(0)

while (True):
    sucess, imgWebcam = cap.read()
    time_a=time.time()
    imgAug=imgWebcam.copy()
    kp2 = _detect.detect(imgWebcam, None)
    kp2, des2 = _compute.compute(imgWebcam, kp2)
    # kp2, des2 = sift.detectAndCompute(imgWebcam, None)
    # imgWebcam=cv2.drawKeypoints(imgWebcam,kp1,None)
    bf = cv2.BFMatcher(normType = cv2.NORM_L2,crossCheck = True)
    if (type(des1)!=type(st) and type(des2)!=type(st)):
        matches = bf.match(des1, des2)
        good = []
        # try:
        #     for m, n in matches:
        #         if (m.distance < 0.65*n.distance):
        #             good.append(m)
        # except(ValueError):
        #     tmp=0
        matches = sorted(matches, key=lambda x: x.distance)
        good=matches
        # Draw first 30 matches
        imgFeature= cv2.drawMatches(img1=imgTarget,
                                        keypoints1=kp1,
                                        img2=imgWebcam,
                                        keypoints2=kp2,
                                        matches1to2=matches[:30],
                                        outImg=None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # imgFeature = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
        cv2.imshow("feature",imgFeature)
        #cv2.waitKey(50)
        imgAug=imgWebcam
        if (len(good) > 130):
            print(len(good))
            try:
                srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                #print(srcPts)
                dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix,mask=cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
                if (type(matrix)!=type(st)):
                    pts=np.float32([[0,0],[0,hT-1],[wT-1,hT-1],[wT-1,0]]).reshape(-1,1,2)
                    #print(pts)
                    dst=cv2.perspectiveTransform(pts,matrix)
                    # img2=cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)
                    #cv2.imshow("img2",img2)
                    imgWarp=cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))
                    # cv2.imshow("imgWarp",imgWarp)
                    maskNew=np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
                    #cv2.imshow("maskNew",maskNew)
                    cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
                    maskInv=cv2.bitwise_not(maskNew)
                    imgAug=cv2.bitwise_and(imgAug,imgAug,mask=maskInv)
                    imgAug=cv2.bitwise_or(imgWarp,imgAug)
                    cv2.imshow("imgAug",imgAug)

                    #add some news
                    matchesMask = mask.ravel().tolist()
                    draw_params = dict(matchColor = (0,255,0),
                                       singlePointColor = None,
                                       matchesMask = matchesMask,
                                       flags = 2)
                    imgFeature = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good,None, **draw_params)
                    #cv2.imshow("inlier only", img3)

                    cv2.imshow("feature", imgFeature)
                    # cv2.waitKey(0)
            except (IndexError):
                tmp=0
    # cv2.imshow("imgAug", imgAug)
    #cv2.imshow("Webcam", imgWebcam)
    time_b=time.time()
    print(time_b-time_a)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
