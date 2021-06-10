import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('TargetImage.jpg')
myVid = cv2.VideoCapture('video.mp4')

st = None

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(imgTarget, None)
imgTarget1=cv2.drawKeypoints(imgTarget,kp1,None)
# cv2.imshow("tt",imgTarget1)
# cv2.waitKey(0)
while (True):
    sucess, imgWebcam = cap.read()
    imgAug=imgWebcam.copy()

    kp2, des2 = sift.detectAndCompute(imgWebcam, None)
    # imgWebcam=cv2.drawKeypoints(imgWebcam,kp1,None)
    bf = cv2.BFMatcher()

    if (type(des1)!=type(st) and type(des2)!=type(st)):
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        try:
            for m, n in matches:
                if (m.distance < 0.65*n.distance):
                    good.append(m)
        except(ValueError):
            tmp=0
        imgFeature = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

        if (len(good) > 15):
            print(len(good))
            try:
                srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                print(srcPts)
                dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix,mask=cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
                if (type(matrix)!=type(st)):
                    pts=np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
                    print(pts)
                    dst=cv2.perspectiveTransform(pts,matrix)
                    img2=cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)
                    cv2.imshow("img2",img2)
                    imgWarp=cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))
                    # cv2.imshow("imgWarp",imgWarp)
                    maskNew=np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
                    cv2.imshow("maskNew",maskNew)
                    cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
                    maskInv=cv2.bitwise_not(maskNew)
                    imgAug=cv2.bitwise_and(imgAug,imgAug,mask=maskInv)
                    imgAug=cv2.bitwise_or(imgWarp,imgAug)
                    cv2.imshow("imgAug",imgAug)
                    cv2.imshow("feature", imgFeature)
                    # cv2.waitKey(0)
            except (IndexError):
                tmp=0

    cv2.imshow("Webcam", imgWebcam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
