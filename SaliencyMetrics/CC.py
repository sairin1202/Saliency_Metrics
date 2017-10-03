import numpy as np
import cv2
import math

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


def CC(saliencyMap1, saliencyMap2):
    map1=cv2.resize(saliencyMap1,(saliencyMap2.shape[1],saliencyMap2.shape[0]))
    map1=map1.astype(float)
    map1=map1/255.0
    map2=saliencyMap2.astype(float)
    map2=map2/255.0	
	
    map1=(map1-np.mean(map1))/np.std(map1)
    map2=(map2-np.mean(map2))/np.std(map2)
	
    score=corr2(map1,map2)
	
    return score
	
img1 = cv2.imread('3.png',cv2.IMREAD_GRAYSCALE)

img2 = cv2.imread('2.png',cv2.IMREAD_GRAYSCALE)

print(CC(img2,img1))