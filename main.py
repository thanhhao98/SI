import os
import cv2 
import numpy as np
import math
import scipy.stats 
import pickle
Theshold = 100

def getTableFromImage(img):
    if len(img.shape) == 2:
        gray_img = img
    elif len(img.shape) ==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    h_img = thresh_img.copy()
    v_img = thresh_img.copy()
    scale = 60 # play around with this

    h_size = int(h_img.shape[1]/scale)

    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size,1))
    h_erode_img = cv2.erode(h_img,h_structure, 1)

    h_dilate_img = cv2.dilate(h_erode_img,h_structure, 1)

    v_size = int(v_img.shape[0] / scale)

    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    v_erode_img = cv2.erode(v_img, v_structure, 1)
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

    mask_img = h_dilate_img + v_dilate_img
    joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)

    return mask_img, joints_img


def normalDistribution(x,mean,sd):
	# var = float(sd)**2
	# denom = (2*math.pi*var)**.5
	# num = math.exp(-(float(x)-float(mean))**2/(2*var))
	# return num/denom
	return scipy.stats.norm(m, sd).pdf(x)

def getMean(array):
	result = 0.0
	for i in array:
		result +=i
	return result/len(array)

def getVariance(array,mean):
	result = 0.0
	n = len(array)
	for i in array:
		result += (i-mean)*(i-mean)
	if n < 30:
		return math.sqrt(result/(n-1))
	else:
		return math.sqrt(result/n)

def getDistance(A, B):
	newA = np.copy(A)
	newB = np.copy(B)
	if len(newA.shape) != 1:
		newA = A.flatten()
	if len(newB.shape) != 1:
		newB = B.flatten()
	if newA.shape == newB.shape:
		return np.linalg.norm(newA-newB)/len(newA)
	else:
		return None

def reshapeImage(image, width=2500, hight=3300):
	return cv2.resize(image,(width,hight))

def checkMap(distance,m,v):
	p = normalDistribution(distance,m,v)
	if p > Theshold:
		return True
	else:
		return False

pathImage = '/Users/mpxt2/Downloads/si_3000_1/'
root, array, m, v, listImages = pickle.load(open('/Users/mpxt2/Downloads/labeled/2/data1.pickle', "rb"))
print(len(listImages))
files = os.listdir(pathImage)
print(listImages)
print(m)
print(v)
numOk = 0
numNotOk = 0
for f in files:
	tokens = f.split('.')
	if len(tokens) == 2 and tokens[-1] in ['png','jpg','JPG'] and (not f in listImages):
		img = cv2.imread(pathImage + f,0)
		img,_ = getTableFromImage(img)
		img = reshapeImage(img)
		result = getDistance(root,img)
		p = normalDistribution(result,m,v)
		if result < m + v  or  checkMap(result,m,v):
			numOk +=1
			array.append(result)
			print(f+'\t map successfully' + '\t' + str(result) + '\t'+str(p))
		else:
			numNotOk +=1

print(numOk)
print(numNotOk)





