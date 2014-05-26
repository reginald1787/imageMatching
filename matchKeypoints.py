
import numpy as np
import cv2
from matplotlib import pyplot as plt

def drawMatches(img, template, skp, tkp, num=-1):
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1+w2
    nHeight = max(h1, h2)
    hdif = (h1-h2)/2
    newimg = np.zeros((nHeight, nWidth), np.uint8)
    newimg[hdif:hdif+h2, :w2] = template
    newimg[:h1, w2:w1+w2] = img

    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
        pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
        cv2.line(newimg, pt_a, pt_b, (155, 155, 155))
    return newimg
             
def SIFT(img):
    # Initiate SIFT detector
    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img,None)

    return kp,des

def corners(img):
    fast = cv2.FastFeatureDetector()
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    # find and draw the keypoints
    kp = fast.detect(img,None)

    kp,des = descriptor.compute(img,kp)

    return kp,des

def match(flag='corners'):
    img1 = cv2.imread('mug.jpg',0)          # queryImage
    img2 = cv2.imread('mug_rotate.jpg',0) # trainImage

    if flag=='SIFT':
    # SIFT match
        kp1,des1 = SIFT(img1)
        kp2,des2 = SIFT(img2)
    # corner match
    else:
        kp1,des1 = corners(img1)
        kp2,des2 = corners(img2)

    matchKeyPoints(img1,img2,kp1,kp2,des1,des2)


def matchKeyPoints(img1,img2,kp1,kp2,des1,des2):
    MIN_MATCH_COUNT = 10


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    #Now we set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. Otherwise simply show a message saying not enough matches are present.
    #If enough matches are found, we extract the locations of matched keypoints in both the images. They are passed to find the perpective transformation. 
    #Once we get this 3x3 transformation matrix, we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it.

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        #print type(img2)
        #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.CV_AA)
        #print type(img2)



    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    #Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    # print img1.shape,img2.shape
    # print kp1[0].pt, kp1[0].size,kp1[0].angle,kp1[0].response, kp1[0].octave, kp1[0].class_id
    # print kp2
    # print good
    # print matchesMask

    #img1=cv2.drawKeypoints(img1,kp1)
    #img2=cv2.drawKeypoints(img2,kp2)
    #img3 = np.zeros((5000,5000),np.uint8)
    #drawMatches(img1,kp1,img2,kp2,good,img3,**draw_params)
    img3 = drawMatches(img1,img2,kp1,kp2)

    plt.imshow(img3, 'gray')
    # #plt.imshow(img2,'gray') 
    plt.show()



if __name__ == '__main__':
    match()
