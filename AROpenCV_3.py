import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('TargetImage.jpg')
myVid = cv2.VideoCapture('video.mp4')

st = None

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

def SIFT():
    # Initiate SIFT detector
    SIFT = cv2.xfeatures2d.SIFT_create()
    return SIFT

def SURF():
    SURF = cv2.xfeatures2d.SURF_create()
    return

def KAZE():
    # Initiate KAZE descriptor
    KAZE = cv2.KAZE_create()
    return KAZE

def BRIEF():
    # Initiate BRIEF descriptor
    BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    return BRIEF
def ORB():
    # Initiate ORB detector
    ORB = cv2.ORB_create()

    return ORB

# Call function BRISK
def BRISK():
    # Initiate BRISK descriptor
    BRISK = cv2.BRISK_create()

    return BRISK

# Call function AKAZE
def AKAZE():
    # Initiate AKAZE descriptor
    AKAZE = cv2.AKAZE_create()

    return AKAZE
    #khong tot

# Call function FREAK
def FREAK():
    # Initiate FREAK descriptor
    FREAK = cv2.xfeatures2d.FREAK_create()

    return FREAK

feature_detect =ORB()
kp1, des1 = feature_detect.detectAndCompute(imgTarget, None)
imgTarget1=cv2.drawKeypoints(imgTarget,kp1,None)
cv2.imshow("tt",imgTarget1)
cv2.waitKey(0)
while (True):
    sucess, imgWebcam = cap.read()
    imgAug=imgWebcam.copy()

    kp2, des2 = feature_detect.detectAndCompute(imgWebcam, None)
    # imgWebcam=cv2.drawKeypoints(imgWebcam,kp1,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    bf = cv2.FlannBasedMatcher(index_params, search_params)
    if (type(des1)!=type(st) and type(des2)!=type(st)):
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        try:
            for m, n in matches:
                if (m.distance < 0.75*n.distance):
                    good.append(m)
        except(ValueError):
            tmp=0
        imgFeature = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
        cv2.imshow("feature", imgFeature)
        if (len(good) > 5):
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
                    # img2=cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)
                    # cv2.imshow("img2",img2)
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
