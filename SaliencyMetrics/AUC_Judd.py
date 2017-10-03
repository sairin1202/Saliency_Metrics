import numpy as np
import cv2

def AUC_Judd(saliencyMap, fixationMap,jitter=1, toPlot=0):
    score=100000000
    saliencyMap=saliencyMap.astype(float)
    fixationMap=fixationMap.astype(float)
    if saliencyMap.shape != fixationMap.shape:
	saliencyMap = cv2.resize(saliencyMap,(fixationMap.shape[1],fixationMap.shape[0]))
    if jitter:
	saliencyMap = saliencyMap+np.random.rand(fixationMap.shape[0],fixationMap.shape[1])/10000000.0
    saliencyMap=(saliencyMap-np.min(saliencyMap))/(np.max(saliencyMap)-np.min(saliencyMap))
    S=saliencyMap        
    S=np.reshape(S,S.shape[0]*S.shape[1],order='F')
    F=fixationMap        
    F=np.reshape(F,F.shape[0]*F.shape[1],order='F')
    Sth=S[np.where(F>0)]
    Nfixations = len(Sth)
    Npixels = len(S)        
    allthreshes = np.sort(Sth, axis=None)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(Nfixations+2)
    fp = np.zeros(Nfixations+2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for i in range(0,Nfixations):
	thresh = allthreshes[i]
	aboveth = np.sum(len(np.where(S>= thresh)[0]))
	tp[i+1] = (float)(i) / Nfixations
	fp[i+1] = (float)(aboveth-i) / (Npixels - Nfixations)

    score = np.trapz(x=fp,y=tp);
    return score
	
img1 = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)

img2 = cv2.imread('2.png',cv2.IMREAD_GRAYSCALE)

print(AUC_Judd(img1,img2,jitter=1))
