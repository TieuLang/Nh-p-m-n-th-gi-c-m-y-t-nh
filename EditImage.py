import numpy as np
import cv2

img=cv2.imread("C:\\Users\\thang\\PycharmProjects\\AROpenCV\\girl.jpg")
print(type(img))
min_width=300
max_width=470
min_height=100
max_height=500
tmp=img[300:470,100:500,:]
d=np.zeros_like(tmp)
for i in range(tmp[0]):
    d[i][0]=tmp[i][0]
for i in range(1,tmp):
    for j in range(1,tmp[i]-1):
        vtmin = int(-1)
        for z in range(-1,2):
            s=int(0)
            for k in range(tmp[i][j]):
                s=s+tmp[i][j][k]-tmp[i-1][j+range][k]
        if (vtmin==-1 or d[vtmin])
cv2.imshow("t",tmp)
cv2.waitKey(0)