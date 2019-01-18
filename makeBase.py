from main import normalDistribution, reshapeImage, getDistance, getMean, getVariance, getTableFromImage
import cv2
import os
import pickle

pathListFolder = '/Users/mpxt2/Downloads/labeled/'
folders = os.listdir(pathListFolder)
for folder in folders:
    if len(folder.split('.')) == 1 and folder == '2' :
        pathFolder = pathListFolder + folder + '/'
        images = os.listdir(pathFolder)
        array = list()
        listName = list()
        rootImage = images[-1]
        root = cv2.imread(pathFolder + rootImage,0)
        cv2.imwrite('table.png',root)
        root, _ = getTableFromImage(root)
        root = reshapeImage(root)
        cv2.imwrite('table.png',root)
        images = images[:-1]
        for f in images:
            tokens = f.split('.')
            if len(tokens) == 2 and tokens[-1] in ['png','jpg','JPG']:
                img = cv2.imread(pathFolder + f,0)
                img,_ = getTableFromImage(img)
                img = reshapeImage(img)
                cv2.imwrite(f+'.png',img)
                listName.append(f)                
                distance = getDistance(root,img)
                array.append(distance)
        m = getMean(array)
        v = getVariance(array,m)
        if v == 0.0:
            v = 0.001
        result = (root,array,m,v,listName)
        with open(pathFolder + 'data1.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(array)
        print(m)
        print(v)
        print(listName)
        print("Saving ok")
