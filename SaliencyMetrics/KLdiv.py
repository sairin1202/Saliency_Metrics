import numpy as np
import cv2
import math


def KLdiv(saliencyMap1, saliencyMap2):
    map1=cv2.resize(saliencyMap1,(saliencyMap2.shape[1],saliencyMap2.shape[0]))
    map1=map1.astype(float)
    map1=map1/255.0
    map2=saliencyMap2.astype(float)
    map2=map2/255.0	
	
    map1 = map1/np.sum(map1)
    map2 = map2/np.sum(map2)
	
    score= np.sum(np.sum(map2*np.log(2.2204e-16 + map2/(map1+2.2204e-16))))
	
    return score
	
img1 = cv2.imread('2.png',cv2.IMREAD_GRAYSCALE)

img2 = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)

print(KLdiv(img1,img2))
