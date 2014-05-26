import cv
import cv2
from numpy import *
#***************************************** 
# Check colinearity of a set of pts 
# input: p (pts to be checked) 
# num (ttl number of pts) 
# return True if some pts are coliner 
# False if not 
#***************************************** 
def isColinear(num, p):  
	 #int i,j,k  
	 #bool iscolinear  
	 #double value  
	 pt1 = cv.CreateMat(3,1,CV_64FC1)  
	 pt2 = cv.CreateMat(3,1,CV_64FC1)  
	 pt3 = cv.CreateMat(3,1,CV_64FC1)  
	 line = cv.CreateMat(3,1,CV_64FC1)  
	 
	 iscolinear = False  
	 # check for each 3 points combination 
	 #for(i=0  i<num-2  i++)  
	 for i in range(num-2):
		 cv.Mset(pt1,0,0,p[i].x)  
		 cv.Mset(pt1,1,0,p[i].y)  
		 cv.Mset(pt1,2,0,1)  
		 #for(j=i+1  j<num-1  j++)  
		 for j in range(i+1,num-1):
			 cv.Mset(pt2,0,0,p[j].x)  
			 cv.Mset(pt2,1,0,p[j].y)  
			 cv.Mset(pt2,2,0,1)  
			 # compute the line connecting pt1 & pt2 
		 	 cv.CrossProduct(pt1, pt2, line)  
			 #for(k=j+1  k<num  k++)  
			 for k in range(j+1,num):
				 cv.Mset(pt3,0,0,p[k].x)  
				 cv.Mset(pt3,1,0,p[k].y)  
				 cv.Mset(pt3,2,0,1)  
				 # check whether pt3 on the line 
				 value = cv.DotProduct(pt3, line)  
				 if(abs(value) < 10e-2):  
					 iscolinear = True  
					 break  
			   
			   
			 if(iscolinear == True):
			 	break  

		 if(iscolinear == True):
		 	break  
	   
	 # cvReleaseMat(&pt1)  
	 # cvReleaseMat(&pt2)  
	 # cvReleaseMat(&pt3)  
	 # cvReleaseMat(&line)  
	 return iscolinear  


#********************************************************************** 
# finding the normalization matrix x' = T*x, where T= s,0,tx, 0,s,ty, 0,0,1  
# compute T such that the centroid of x' is the coordinate origin (0,0)T 
# and the average distance of x' to the origin is sqrt(2) 
# we can derive that tx = -scale*mean(x), ty = -scale*mean(y), 
# scale = sqrt(2)/(sum(sqrt((xi-mean(x)^2)+(yi-mean(y))^2))/n) 
# where n is the total number of points 
# input: num (ttl number of pts) 
# p (pts to be normalized) 
# output: T (normalization matrix) 
# p (normalized pts) 
# NOTE: because of the normalization process, the pts coordinates should 
# has accurcy as "float" or "double" instead of "int" 
#********************************************************************** 
#void Normalization(int num, CvPoint2D64f *p, CvMat *T)  

def Normalization(num,p,T):
	 # double scale, tx, ty  
	 # double meanx, meany  
	 # double value  
	 # int i  
	 x = cv.CreateMat(3,1,CV_64FC1)  
	 xp = cv.CreateMat(3,1,CV_64FC1)  
	 
	 meanx = 0  
	 meany = 0  
	 #for(i=0  i<num  i++)  
	 for i in range(num):
		 meanx += p[i].x  
		 meany += p[i].y  
	   
	 meanx /= float(num)  
	 meany /= float(num)  
	 
	 value = 0  
	 #for(i=0  i<num  i++) 
	 for i in range(num):
	 	value += sqrt(pow(p[i].x-meanx, 2.0) + pow(p[i].y-meany, 2.0))  
	 value /= float(num)  
	 
	 scale = sqrt(2.0)/value  
	 tx = -scale * meanx  
	 ty = -scale * meany  
	 
	 cv.SetZero(T)  
	 cv.mSet(T,0,0,scale)   
	 cv.mSet(T,0,2,tx)  
	 cv.mSet(T,1,1,scale)  
	 cv.mSet(T,1,2,ty)  
	 cv.mSet(T,2,2,1.0)  
	 
	 #Transform x' = T*x 
	 #for(i=0  i<num  i++)  
	 for i in range(num):
		 cv.mSet(x,0,0,p[i].x)  
		 cv.mSet(x,1,0,p[i].y)  
		 cv.mSet(x,2,0,1.0)  
		 #cvMatMul(T,x,xp)  
		 cv.matMulDeriv(T,x,xp)
		 p[i].x = cv.mGet(xp,0,0)/cvmGet(xp,2,0)  
		 p[i].y = cv.mGet(xp,1,0)/cvmGet(xp,2,0)  
	   
	 
	 # cvReleaseMat(&x)  
	 # cvReleaseMat(&xp)  
	  


  
#**************************************************************** 
# Compute the homography matrix H 
# i.e., solve the optimization problem min ||Ah||=0 s.t. ||h||=1 
# where A is 2n*9, h is 9*1 
# input: n (number of pts pairs) 
# p1, p2 (coresponded pts pairs x and x') 
# output: 3*3 matrix H 
#**************************************************************** 
#void ComputeH(int n, CvPoint2D64f *p1, CvPoint2D64f *p2, CvMat *H)
def ComputeH(n,p1, p2, H):  
	 #int i  
	 A = cv.CreateMat(2*n, 9, CV_64FC1)  
	 U = cv.CreateMat(2*n, 2*n, CV_64FC1)  
	 D = cv.CreateMat(2*n, 9, CV_64FC1)  
	 V = cv.CreateMat(9, 9, CV_64FC1)  
	 
	 cv.SetZero(A)  
	 #for(i=0  i<n  i++)  
	 for i in range(n):
	 # 2*i row 
		 cv.Mset(A,2*i,3,-p1[i].x)  
		 cv.Mset(A,2*i,4,-p1[i].y)  
		 cv.Mset(A,2*i,5,-1)  
		 cv.Mset(A,2*i,6,p2[i].y*p1[i].x)  
		 cv.Mset(A,2*i,7,p2[i].y*p1[i].y)  
		 cv.Mset(A,2*i,8,p2[i].y)  
		 # 2*i+1 row 
		 cv.Mset(A,2*i+1,0,p1[i].x)  
		 cv.Mset(A,2*i+1,1,p1[i].y)  
		 cv.Mset(A,2*i+1,2,1)  
		 cv.Mset(A,2*i+1,6,-p2[i].x*p1[i].x)  
		 cv.Mset(A,2*i+1,7,-p2[i].x*p1[i].y)  
		 cv.Mset(A,2*i+1,8,-p2[i].x)  
	   
	 
	 # SVD 
	 # The flags cause U and V to be returned transposed 
	 # Therefore, in OpenCV, A = U^T D V 
	 cv.SVD(A, D, U, V, cv.CV_SVD_U_T|cv.CV_SVD_V_T)  
	 
	 # take the last column of V^T, i.e., last row of V 
	 #for(i=0  i<9  i++) 
	 for i in range(9):
	 	cv.Mset(H, i/3, i%3, cv.mGet(V, 8, i))  
	 
	 # cvReleaseMat(&A)  
	 # cvReleaseMat(&U)  
	 # cvReleaseMat(&D)  
	 # cvReleaseMat(&V)  
  
 
#***************************************************************************** 
# RANSAC algorithm 
# input: num (ttl number of pts) 
# m1, m2 (pts pairs) 
# output: inlier_mask (indicate inlier pts pairs in (m1, m2) as 1  outlier: 0) 
# H (the best homography matrix) 
#***************************************************************************** 
# def RANSAC_homography(int num, CvPoint2D64f *m1, CvPoint2D64f *m2, CvMat *H, 
# CvMat *inlier_mask):  


def RANSAC_homography(num,m1,m2,H,inlier_mask):
	 #int i,j  
	 N = 1000, s = 4, sample_cnt = 0  
	 e=0, p = 0.99  
	 #int numinlier, MAX_num  
	 #double curr_dist_std, dist_std  
	 #bool iscolinear  
	 #CvPoint2D64f *curr_m1 = new CvPoint2D64f[s]  
	 #CvPoint2D64f *curr_m2 = new CvPoint2D64f[s]  
	 curr_m1 =[0 for i in range(s)]
	 curr_m2 = curr_m1
	 #int *curr_idx = new int[s]
	 curr_idx = curr_m1  
	 
	 curr_inlier_mask = cv.CreateMat(num,1,CV_64FC1)  
	 curr_H = cv.CreateMat(3,3,CV_64FC1)  
	 T1 = cv.CreateMat(3,3,CV_64FC1)  
	 T2 = cv.CreateMat(3,3,CV_64FC1)  
	 invT2 = cv.CreateMat(3,3,CV_64FC1)  
	 tmp_pt = cv.CreateMat(3,1,CV_64FC1)  
	 
	 # RANSAC algorithm (reject outliers and obtain the best H) 
	 srand(134)  
	 MAX_num = -1  
	 while(N > sample_cnt):  
	 # for a randomly chosen non-colinear correspondances 
		 iscolinear = True  
		 while(iscolinear == True):  
			 iscolinear = False  
			 #for(i=0  i<s  i++)  
			 for i in range(s):
			 # randomly select an index 
			 	curr_idx[i] = rand()%num  
			 		#for(j=0  j<i  j++)  
			 		for j in range(i):
			 			if(curr_idx[i] == curr_idx[j]):
			 				iscolinear = True  
			 				break  
	   
	   
	 			if(iscolinear == True):
	 				break  
				curr_m1[i].x = m1[curr_idx[i]].x  
				curr_m1[i].y = m1[curr_idx[i]].y  
				curr_m2[i].x = m2[curr_idx[i]].x  
				curr_m2[i].y = m2[curr_idx[i]].y  
	   
			 # Check whether these points are colinear 
			 if(iscolinear == False): 
			 	iscolinear = isColinear(s, curr_m1)  
	   
	 # Nomalized DLT 
	 Normalization(s, curr_m1, T1)  #curr_m1 <- T1 * curr_m1 
	 Normalization(s, curr_m2, T2)  #curr_m2 <- T2 * curr_m2 
	 
	 # Compute the homography matrix H = invT2 * curr_H * T1 
	 ComputeH(s, curr_m1, curr_m2, curr_H)  
	 cv.Invert(T2, invT2)  
	 cv.MatMulDeriv(invT2, curr_H, curr_H)  # curr_H <- invT2 * curr_H 
	 cv.MatMulDeriv(curr_H, T1, curr_H)  # curr_H <- curr_H * T1 
	 
	 # Calculate the distance for each putative correspondence 
	 # and compute the number of inliers 
	 # numinlier = 
		# ComputeNumberOfInliers(num,m1,m2,curr_H,curr_inlier_mask,&curr_dist_std)  
	 
	 
	 # # Update a better H 
	 # if(numinlier > MAX_num || (numinlier == MAX_num && curr_dist_std < 
		# dist_std))  
		#  MAX_num = numinlier  
		#  cvCopy(curr_H, H)  
		#  cvCopy(curr_inlier_mask, inlier_mask)  
		#  dist_std = curr_dist_std  
	   
	 
	 # update number N by Algorithm 4.5 
	 # e = 1 - float(numinlier) / float(num)  
	 # N = (int)(log(1-p)/log(1-pow(1-e,s)))  
	 # sample_cnt++  
	   
	 
	 # Optimal estimation using all the inliers 
	 # delete curr_m1, curr_m2, curr_idx  
	 # cvReleaseMat(&curr_H)  
	 # cvReleaseMat(&T1)  
	 # cvReleaseMat(&T2)  
	 # cvReleaseMat(&invT2)  
	 # cvReleaseMat(&tmp_pt)  
	 # cvReleaseMat(&curr_inlier_mask)  
  
 
#int main(int argc, char *argv[]) 

def homography():
	# NOTE: because of the normalization process, the pts coordinates 
	# should has accurcy as "float" or "double" instead of "int" 
	# load the color image1 and image2 
	# create gray scale image 


	# detect corner 
 	# corner points are stored in CvPoint cornerp1 and cornerp2 

 	# feature matching by NCC 
 	# matched pairs are stored in CvPoint2D64f matched1 and matched2

 	# generate a new image displaying the two images 
