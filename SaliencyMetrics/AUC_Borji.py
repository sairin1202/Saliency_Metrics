import numpy as np
import cv2

def AUC_Borji(saliencyMap, fixationMap,Nsplits=100, stepSize=0.1, toPlot=0):
    if np.sum(fixationMap)<=1:
	print('no fixationMap')
	return 
    score=100000000
    saliencyMap=saliencyMap.astype(float)
    fixationMap=fixationMap.astype(float)
    if saliencyMap.shape != fixationMap.shape:
	saliencyMap = cv2.resize(saliencyMap,(fixationMap.shape[1],fixationMap.shape[0]))
    saliencyMap = (saliencyMap-np.min(saliencyMap))/(np.max(saliencyMap)-np.min(saliencyMap))
    S=saliencyMap
    S=np.reshape(S,S.shape[0]*S.shape[1],order='F')
    F=fixationMap
    F=np.reshape(F,F.shape[0]*F.shape[1],order='F')	
    Sth=S[np.where(F>0)]
    Nfixations=len(Sth)
    Npixels=len(S)
    r=np.random.randint(Npixels,size=(Nfixations,Nsplits))
    randfix=S[r]
	
    auc=[0]*Nsplits
    for s in range(Nsplits):
	curfix=randfix[:,s];
        temp=list(Sth)
        temp.extend(list(curfix))
	allthreshes=x2 = np.arange(0,np.max(temp)+stepSize,stepSize)
        allthreshes=allthreshes[::-1]
	tp=np.zeros(len(allthreshes)+2)
	fp=np.zeros(len(allthreshes)+2)
	tp[0]=0
	tp[-1]=1
	fp[0]=0
	fp[-1]=1
	for i in range(len(allthreshes)):
	    thresh=allthreshes[i]
	    tp[i+1]=np.sum(len(np.where(Sth>= thresh)[0]))/(float)(Nfixations)
	    fp[i+1]=np.sum(len(np.where(curfix>= thresh)[0]))/(float)(Nfixations)
	auc[s]=np.trapz(x=fp,y=tp)
        #print(auc)
    score=np.mean(auc)
    return score

img1 = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)

img2 = cv2.imread('2.png',cv2.IMREAD_GRAYSCALE)

print(AUC_Borji(img1,img2))
