import cv2
import re
import numpy as np
import itertools
import sys
import scipy.misc as misc
from scipy.ndimage.filters import gaussian_laplace as LoG
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def findKeyPoints(img, template, distance=200):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)

    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            skp_final.append(skp[i])

    flann = cv2.flann_Index(td, flann_params)
    idx, dist = flann.knnSearch(sd, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tkp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            tkp_final.append(tkp[i])

    return skp_final, tkp_final

def drawKeyPoints(img, template, skp, tkp, num=-1):
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1+w2
    nHeight = max(h1, h2)
    hdif = (h1-h2)/2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif+h2, :w2] = template
    newimg[:h1, w2:w1+w2] = img

    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
        pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
    return newimg


# % updated 10.11.2010



def match():
    # img = cv2.imread(sys.argv[1])
    # temp = cv2.imread(sys.argv[2])
    img = cv2.imread("mug.jpg")
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = misc.imresize(img,(500,500,3))



    temp = cv2.imread("mug4.JPG")
    #temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    #print temp.shape
    temp = misc.imresize(temp,(512,512))

    try:
    	dist = int(sys.argv[3])
    except IndexError:
    	dist = 200
    try:
    	num = int(sys.argv[4])
    except IndexError:
    	num = -1
    skp, tkp = findKeyPoints(img, temp, dist)
    newimg = drawKeyPoints(img, temp, skp, tkp, num)
    cv2.imshow("image", newimg)
    cv2.waitKey(0)

def siftpoints():
    img = cv2.imread('mug4.JPG')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)


    img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('sift_keypoints.jpg',img)
    
    cv2.waitKey(0)

def cornerpoints(filename,thres=0.1):

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    if gray.shape!=(512,512):
        gray = cv2.resize(gray, (512,512), interpolation=cv2.INTER_LINEAR)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    cords = []
    nrow, ncol = dst.shape
    #print nrow,ncol
    gap = min(nrow,ncol)*0.1

    lasti=lastj=-2
    for i in range(nrow):
        for j in range(ncol):
            #print i,j
            if dst[i,j]>thres*dst.max():
                if max(abs(i-lasti),abs(j-lastj))>gap:
                    cords.append((i,j))
                    lasti=i
                    lastj=j
                    #print i,j
    
    

    return cords
    # Threshold for an optimal value, it may vary depending on the image.
    #img[dst>thres*dst.max()]=[0,0,255]

    # kms =KMeans()
    # kms.fit(cords)
    # centers = kms.cluster_centers_

    # return centers
    # for a in centers:
    #     i = int(a[0])
    #     j = int(a[1])
    #     print i,j
    #     img[i,j]=[0,0,255]
    #cv2.imshow('img',img)
    #cv2.waitKey(0)




if __name__ == '__main__':
    
    filename1 = 'mug.jpg'
    filename2 ='mug5.png'
    #img = cv2.imread(filename2)
    #cords1 = cornerpoints(filename1,0.0001)
    #cords2 = cornerpoints(filename2,0.1)

    #np.savetxt('cords1.txt',cords1)
    #np.savetxt('cords2.txt',cords2)
    
    img = np.zeros(shape=(512,512,3))
    cords = open('cords1.txt','r')
    
    for line in cords.readlines():
        a = line.split()
        #print float(a[0]), float(a[1])
        img[float(a[0]),float(a[1])] = [255,255,255]
    
    cords.close()
    # nrow=ncol=len(X)    
    # dst = np.zeros((nrow,ncol))
    # for i in range(nrow):
    #     for j in range(i+1,ncol):
    #         dst[i,j] = np.linalg.norm(np.array(X[i])-np.array(X[j]))
    #         dst[j,i] = dst[i,j]

    # #cv2.imshow('dst',dst)
    # dst = dst/dst.max()
    #newimg = LoG(img,.4)
    
    cv2.imshow('dst',img)
    cv2.waitKey(0)


    # img = cv2.imread('mug2.jpg')
    # edges = cv2.Canny(gray,100,200)

    # #print transpose(nonzero(edges))
    # plt.subplot(121),plt.imshow(gray,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
