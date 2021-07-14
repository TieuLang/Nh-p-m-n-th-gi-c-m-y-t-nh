import cv2
import numpy as np
import time
import argparse
import math
import os

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            #elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            #elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))

DEFAULT_COLOR = (50, 50, 50)

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    print(model.shape)
    h=model.shape[0]
    w=model.shape[1]
    # h, w = model.shape
    print("faces",len(obj.faces))
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            # color = hex_to_rgb(face[-1])
            print("len",len(face))
            color=face[-1]
            color1=np.zeros((1,1,3),dtype=np.uint8)
            for i in range(3):
                color1[0][0][i]=np.uint8((color[i]))
            print(type(color[0]))
            color1=cv2.cvtColor(color1,cv2.COLOR_HSV2BGR)
            for i in range(3):
                color[i]=int(color1[0][0][i])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    print("hex_to_rgb: ",hex_color)
    print(type(hex_color))
    # hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color))
    # return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
obj = OBJ('fox.obj', swapyz=True)

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
                    img2=cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)
                    projection = projection_matrix(camera_parameters, matrix)
                    imgAug = render(imgAug, obj, projection, imgTarget, False)

                    # frame = render(frame, imgTarget, projection,imgTarget)
                    #cv2.imshow("img2",img2)
                    imgWarp=cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))
                    # cv2.imshow("imgWarp",imgWarp)
                    maskNew=np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
                    #cv2.imshow("maskNew",maskNew)
                    cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
                    # maskInv=cv2.bitwise_not(maskNew)
                    # imgAug=cv2.bitwise_and(imgAug,imgAug,mask=maskInv)
                    # imgAug=cv2.bitwise_or(imgWarp,imgAug)
                    # cv2.imshow("imgAug",imgAug)

                    #add some news
                    matchesMask = mask.ravel().tolist()
                    draw_params = dict(matchColor = (0,255,0),
                                       singlePointColor = None,
                                       matchesMask = matchesMask,
                                       flags = 2)
                    imgFeature = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good,None, **draw_params)
                    #cv2.imshow("inlier only", img3)
                    cv2.imshow("imgAug",imgAug)
                    cv2.imshow("feature", imgFeature)
                    # cv2.waitKey(0)
            except (IndexError):
                tmp=0
    cv2.imshow("imgAug", imgAug)
    #cv2.imshow("Webcam", imgWebcam)
    time_b=time.time()
    print(time_b-time_a)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
