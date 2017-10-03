import numpy as np
import cv2

def NSS(saliencyMap, fixationMap):
    map=cv2.resize(saliencyMap,(fixationMap.shape[1],fixationMap.shape[0]))
    map=(map-np.mean(map))/np.std(map)
    sum=0
    count=0
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if fixationMap[i][j]!=0:
               sum+=map[i][j]
               count+=1
    score=(float)(sum)/(count)
    return score


img1 = cv2.imread('2.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)
print(NSS(img1,img2))