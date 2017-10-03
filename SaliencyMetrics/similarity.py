import numpy as np
import cv2

def similarity(saliencyMap1, saliencyMap2, toPlot=0):
    map1=cv2.resize(saliencyMap1,(saliencyMap2.shape[1],saliencyMap2.shape[0]))
    map1=map1.astype(float)
    map1=map1/255.0
    map2=saliencyMap2.astype(float)
    map2=map2/255.0	
    map1= (map1-np.min(map1))/(np.max(map1)-np.min(map1))
    map1 = map1/np.sum(map1)

    map2= (map2-np.min(map2))/(np.max(map2)-np.min(map2))
    map2 = map2/np.sum(map2)
	
    score=10000000
    diff=np.minimum(map1,map2)
    score=np.sum(diff)
	
    return score
	
img1 = cv2.imread('2.png',cv2.IMREAD_GRAYSCALE)

img2 = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)

print(similarity(img1,img2))