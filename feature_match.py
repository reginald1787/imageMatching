

import numpy as np
import cv2
from matplotlib import pyplot as plt

# # static void _prepareImgAndDrawKeypoints( InputArray img1, const std::vector<KeyPoint>& keypoints1,
# #                                          InputArray img2, const std::vector<KeyPoint>& keypoints2,
# #                                          InputOutputArray _outImg, Mat& outImg1, Mat& outImg2,
# #                                          const Scalar& singlePointColor, int flags )

# class Size:
#     def __init__(self,width,height):
#         self.width = width
#         self.height = height

# # class Point:
# #     def __init__(self,x,y):
# #         self.x = x
# #         self.y = y


# def _prepareImgAndDrawKeypoints(img1, keypoints1,
#                                 img2, keypoints2,
#                                 _outImg, outImg1, outImg2,
#                                 singlePointColor, flags ):
 
#     #Mat outImg 
#     # Size img1size = img1.size(), img2size = img2.size() 
#     # Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height) ) 
#     img1size = Size(img1.shape[0],img1.shape[1]) 
#     img2size = Size(img2.shape[0],img2.shape[1]) 
#     size = Size( img1size.width + img2size.width, max(img1size.height, img2size.height) ) 
#     #if flags and DrawMatchesFlags::DRAW_OVER_OUTIMG:
#     if flags:

#         #outImg = _outImg.getMat() 
#         outImg = _outImg
#         if size.width > outImg.shape[0] or size.height > outImg.shape[1]:
#             raise Exception("Error::outImg has size less than need to draw img1 and img2 together") 
#         #outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) ) 
#         #outImg2 = outImg( Rect(img1size.width, 0, img2size.width, img2size.height) ) 
     
#     else:
     
#         #_outImg.create( size, CV_MAKETYPE(img1.depth(), 3) ) 
#         #_outImg = np.zeros((size.width,size.height,3),np.uint8)
#         #outImg = _outImg.getMat() 
#         #outImg = Scalar::all(0)
#         #outImg = _outImg 
#         # outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) ) 
#         # outImg2 = outImg( Rect(img1size.width, 0, img2size.width, img2size.height) )
#         outImg1 = np.zeros((img1size.width,img1size.height),np.uint8) 
#         outImg2 = np.zeros((img2size.width,img2size.height),np.uint8) 


#         if( img1.type() == cv2.CV_8U ):
#             cv2.cvtColor( img1, outImg1, cv2.COLOR_GRAY2BGR ) 
#         else:
#             #img1.copyTo( outImg1 ) 
#             outImg1 = img1

#         if( img2.type() == cv2.CV_8U ):
#             cv2.cvtColor( img2, outImg2, cv2.COLOR_GRAY2BGR ) 
#         else:
#             #img2.copyTo( outImg2 ) 
#             outImg2 = img2
     

#     #draw keypoints
#     #if( !(flags and DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS) ):
#     if flags == False:
     
#         #Mat _outImg1 = outImg( Rect(0, 0, img1size.width, img1size.height) ) 
#         _outImg1 = np.zeros((img1size.width,img1size.height,3),np.uint8) 
#         #cv2.drawKeypoints( _outImg1, keypoints1, _outImg1, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG ) 
#         cv2.drawKeypoints( _outImg1, keypoints1, _outImg1, singlePointColor) 

#         #Mat _outImg2 = outImg( Rect(img1size.width, 0, img2size.width, img2size.height) ) 
#         _outImg2 = np.zeros((img2size.width,img2size.height,3),np.uint8)
#         #cv2.drawKeypoints( _outImg2, keypoints2, _outImg2, singlePointColor, flags + DrawMatchesFlags::DRAW_OVER_OUTIMG ) 
#         cv2.drawKeypoints( _outImg2, keypoints2, _outImg2, singlePointColor) 
     
 

# # static inline void _drawMatch( InputOutputArray outImg, InputOutputArray outImg1, InputOutputArray outImg2 ,
# #                           const KeyPoint& kp1, const KeyPoint& kp2, const Scalar& matchColor, int flags )

# def _drawMatch(outImg, outImg1, outImg2 ,
#                kp1, kp2, matchColor, flags):
 
#     #RNG& rng = theRNG() 
#     #bool isRandMatchColor = matchColor #== Scalar::all(-1) 
#     #Scalar color = isRandMatchColor ? Scalar( rng(256), rng(256), rng(256) ) : matchColor 
#     color = matchColor

#     cv2.drawKeypoints( outImg1, kp1, color, flags ) 
#     cv2.drawKeypoints( outImg2, kp2, color, flags ) 

#     pt1 = kp1.pt,
#     pt2 = kp2.pt,
#     #dpt2 = Point2f( std::min(pt2.x+outImg1.size().width, float(outImg.size().width-1)), pt2.y ) 
#     dpt2 = cv2.Point2f(min(pt2.x+outImg1.shape[0], float(outImg.shape[0]-1)), pt2.y ) 

#     cv2.line( outImg,
#           (round(pt1.x*draw_multiplier), round(pt1.y*draw_multiplier)),
#           (round(dpt2.x*draw_multiplier), round(dpt2.y*draw_multiplier)),
#           color, 1, cv2.CV_AA) 
 


# def drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
#                   InputArray img2, const std::vector<KeyPoint>& keypoints2,
#                   const std::vector<std::vector<DMatch> >& matches1to2, InputOutputArray outImg,
#                   const Scalar& matchColor, const Scalar& singlePointColor,
#                   const std::vector<std::vector<char> >& matchesMask, int flags ):


# def drawMatches(img1, keypoints1,
#                 img2, keypoints2,
#                 matches1to2, outImg,
#                 matchColor, singlePointColor,
#                 matchesMask, flags ):
 
#     if not matchesMask and len(matchesMask) != len(matches1to2):
#         raise Exception("Error:: matchesMask must have the same size as matches1to2!") 

#     #Mat outImg1, outImg2 
#     outImg1 = outImg
#     outImg2 = outImg
#     _prepareImgAndDrawKeypoints( img1, keypoints1, img2, keypoints2,
#                                  outImg, outImg1, outImg2, singlePointColor, flags ) 

#     #draw matches
#     #for( size_t i = 0  i < matches1to2.size()  i++ )
#     for  i  in  range(len(matches1to2)):
#         #for j in range(matches1to2[i].size()):
         
#             i1 = matches1to2[i].queryIdx 
#             i2 = matches1to2[i].trainIdx 
#             if not matchesMask or matchesMask[i]:
             
#                 kp1 = keypoints1[i1]
#                 kp2 = keypoints2[i2] 
#                 _drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor, flags ) 


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
             
             
def matchSIFT():
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('mug.jpg',0)          # queryImage
    img2 = cv2.imread('mug.jpg',0) # trainImage

    #img1 = cv2.resize(img1,(512,512))

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

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
    matchSIFT()
