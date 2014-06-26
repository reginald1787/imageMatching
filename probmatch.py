#probmatch.py

import numpy as np
from scipy.linalg import sqrtm 
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import cv2
import cv
from BIC import *
from pylab import figure, show, rand
from matplotlib.patches import Ellipse
import re




def edges(filename=None,img=None):
	#img = cv2.imread('IMG_0513.JPG',0)          # queryImage
	#img = cv2.imread('36/36_i120.png',0) # trainImage
	#img = cv2.resize(img,(512,512))
	img = cv2.imread(filename,0)
	edge = cv2.Canny(img,100,200)
	# print type(edge),edge.shape
	# print type(img),img.shape
	# plt.subplot(221),plt.imshow(img,cmap='gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(222),plt.imshow(edge,cmap='gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	img=cv2.drawKeypoints(edge,[],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow('keypoints',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

   
	#edge = edge/float(edge.max())
	#print edge[edge>0]
	kp=[]
	for i in range(edge.shape[0]):
		for j in range(edge.shape[1]):
			if edge[i][j]>0.5:
				kp.append((i,j))

	kp = np.array(kp)
	#print kp.shape
	# plt.subplot(223)
	# plt.plot(kp[:,0],kp[:,1],'.',color='c')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.show()
	return kp

def SIFT(filename=None, img=None):
	img = cv2.imread(filename,0)
	#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Initiate SIFT detector
	sift = cv2.SIFT()
	# find the keypoints and descriptors with SIFT
	kp, des = sift.detectAndCompute(img,None)

	print len(kp)
	img=cv2.drawKeypoints(img,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow('keypoints',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return kp#,des

def corners(filename=None, img=None):
	img = cv2.imread(filename,0)
	fast = cv2.FastFeatureDetector()
	descriptor = cv2.DescriptorExtractor_create("SIFT")
	# find and draw the keypoints
	kp = fast.detect(img,None)

	#img=cv2.drawKeypoints(img,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# cv2.imshow('keypoints',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#kp,des = descriptor.compute(img,kp)

	return kp#,des


def mahalanobis_dst(p,mean,cov):
	cov = cov.reshape(2,2)
	x = (p-mean).reshape(2,1)
	return np.sqrt(x.T.dot(np.linalg.inv(cov)).dot(x))[0,0]


def search_next_gaussian(img,data,visited_g):
	if len(visited_g)==len(data[img][0]):
		#print 'no cadidates'
		return 

	pic = data[img]
	m = pic[0]
	#w = pic[1]
	#c = pic[2]

	minsim = 9999
	minloc = -1
	for i in range(len(m)):
		if i not in visited_g:
			p = m[i]
			maxsim1 = 0
			for j in range(len(data)):
				if j!=img:
					m2 = data[j][0]
					w2 = data[j][1]
					c2 = data[j][2]
					maxsim2 = 0
					for k in range(len(m2)):
						sim = w2[k]/(mahalanobis_dst(p,m2[k],c2[k])+0.1)
						if sim>maxsim2:
							maxsim2 = sim
					if maxsim2>maxsim1:
						maxsim1 = maxsim2
			if maxsim1<minsim:
				minsim = maxsim1
				minloc = i

	#print 'next Gaussian:', m[minloc]
	return minloc


def check(region_m,region_c,test):
	for i in range(len(test)):
		pt = test[i]
		if mahalanobis_dst(pt,region_m,region_c)<1:
			#print 'find point in test:',pt,region
			return True,pt
	return False,None

def search_nearest_gaussian(pts,data):
	# n = len(data)
	# maxsim = 0
	# for j in range(n):
	# 	if j not in visited_pic:
	# 		m = data[j][0]
	# 		w = data[j][1]
	# 		c = data[j][2]
	# 		for i in range(len(m)): 
	# 			sim = 10*w[i]/(mahalanobis_dst(p,m[i],c[i])+0.1)
	# 			if sim > maxsim:
	# 				maxsim = sim
	# 				loc = (j,i)

	# #print maxsim
	# if maxsim >0 :
	# 	#print 'selected Gaussian:',loc
	# 	return loc
	# else:
	# 	#print 'no nearest Gaussian:'
	# 	return
		
	m = data[0]
	w = data[1]
	c = data[2]
	maxsim = 0
	loc = -1
	for i in range(len(m)):
		sim=0
		for p in pts:
			sim += w[i]/(mahalanobis_dst(p,m[i],c[i])+0.1)
			
		if sim > maxsim:
				maxsim = sim
				loc = i
	if maxsim>0:
		return loc
	else:
		return

def search_highest_hypothesis(rankscores):
	return rankscores[-1]


def match(data,test,ranks,rankscores,pts):
	## initialize
	print '\n'
	print 'starting new match for points:',pts
	fig = figure()
	

	## select hypothesis

	img =  search_highest_hypothesis(ranks)

	## search nearest Gaussian
	
	g_loc = search_nearest_gaussian(pts,data[img])

	print 'current hypothesis:\tPic:',img,'Gaussian:,',g_loc
	

	visited_g =[g_loc]
	changepic = False
	plotdata(fig,pts,data,img,rankscores)
	

	ax = fig.add_subplot(111, aspect='equal')
	ax.plot(data[img][0][g_loc][0],data[img][0][g_loc][1],'+',color='b')


	## search next Gaussian most differentiate with otehrs

	while True:
		nextloc = search_next_gaussian(img,data,visited_g)
		if nextloc == None:
			print 'all Gaussian have been found!'
			#print 'pic:',img,'gmmscore:',gmmscore(data[img],test)
			break 
		region_m = data[img][0][nextloc]
		region_c = data[img][2][nextloc]
		find,pt = check(region_m,region_c,test)
		if find:
			pts.append(pt)
			visited_g.append(nextloc)
			#ax.plot(pt[0],pt[1],'+',color='b')
			print 'continue search next Gaussian:',pt
			# continue
		else:
			print 'no Gaussian matched, turn to next pictures!'
			#print 'pic:',img,'gmmscore:',gmmscore(data[img],test)
			changepic = True
			break


	score = gmmscore(pts,data[img])
	print 'search end,\t' 'Pic:', img, 'Similarity:', score
	
	rankscores[img] = score
	if changepic:
		k = ranks.index(img)
		while k>0:
			ranks[k] = ranks[k-1]
			k-=1
		ranks[k] = img
	else:
		ranks = sorted(range(len(rankscores)), key=lambda k:rankscores[k])

	print 'ranking scores:', rankscores,'ranks',ranks[::-1]



	
	plotdata(fig,pts,data,img,rankscores)
	show()

	if score>0.9:
		return

	if changepic:
			
		match(data,test,ranks,rankscores,pts) 

	#show()

	return


def gmmscore(pts,gmm):
	m = gmm[0]
	w = gmm[1]
	c = gmm[2]

	scores = 0.0
	for pt in pts:
		val = 0
		for i in range(len(m)):
			val+=w[i]/(mahalanobis_dst(pt,m[i],c[i])+0.1)
		scores+=val/len(m)

	if scores>1:
		return .95
	return scores

def writedata():
	#kp_mat = np.zeros((144,144))
	with open('results/points.txt','w') as f:
		for  k in range(1,1001):
			kp_mat = []
			X = []
			for i in range(1,9):
				for j in range(1,4):
					kp = corners('/home/lzhong/Downloads/Aldebaran/siftmatch/withprob/grey4/%s/%s_l%sc%s.png'%(k,k,i,j))

					#print len(kp)
					for point in kp:
						point = tuple(point.pt)
						kp_mat.append(point)

			X = np.array(kp_mat,dtype=np.int)
			print X.shape
			s = ''
			for point in X:
				s+=str(point[0])+','+str(point[1])+'\t'
		
			f.write(s+'\n')

def readdata():
	data = []
	with open('results/gmm/points.txt','r') as f:
		mean = []
		weight = []
		count=0
		for line in f.readlines():
			count+=1
			x = line.split('|')
			
			#print len(x)

			mean = np.array([w.split(',') for w in x[0].split()],dtype=np.float)
			weight = np.array([w for w in x[1].split(',') if w!=''],dtype=np.float) 
			num = x[2].split()
			cov = []
			for c in num:
				co = np.array([w for w in c.split(',') if w!=''],dtype=np.float)
				cov.append(co.reshape(2,2))
			data.append([mean,weight,cov])
			#break

		print 'number of pics:',count

	data = np.array(data)
	return data

	# selected = set([8,25,56,114,159,227,281,357,400,498,553,741,809,922,967])
	# with open('results/gmm/points.txt','w') as wf:
	# 	with open('results/points.txt','r') as f:
	# 		count = 0
	# 		for line in f.readlines():
	# 			count+=1
	# 			points = line.split()
	# 			#print len(points)
	# 			X = np.array([s.split(',') for s in points],dtype=np.int)
	# 			# print count, X.shape
	# 			# pic = plt.figure()
	# 			# plt.plot(X[:,0],X[:,1],'x',color='g')
	# 			# plt.savefig('results/pics/%s.png'%count)
				
	# 			if count in selected:

	# 				clf= BIC_GMM(X)
	# 				#fig = figure()
	# 				#ax = fig.add_subplot(111, aspect='equal')
	# 				#ax.plot(X[:,0],X[:,1],'x',color='g')
					
	# 				#print len(zip(clf.means_,clf.weights_,clf.covars_))
	# 				#print len(clf.covars_)
	# 				# for (mean,weight,cov) in zip(clf.means_,clf.weights_,clf.covars_):
	# 				# 	#print cov.shape
	# 				# 	v, w = np.linalg.eigh(cov)
	# 				# 	angle = np.arctan2(w[0][1], w[0][0])
	# 				# 	angle = 180 * angle / np.pi  # convert to degrees
	# 				# 	v /= np.linalg.norm(v)
	# 				# 	v *=40
	# 				# 	#mprint v
	# 				# 	ell = Ellipse(mean, v[0], v[1], 180 + angle, color='r')
	# 				# 	ell.set_clip_box(ax.bbox)
	# 				# 	ell.set_alpha(30*weight**2)
	# 				# 	ax.add_artist(ell)
	# 				# ax.plot(clf.means_[:,0],clf.means_[:,1],'.',color='b')
	# 				# #ax.set_xlim(0, 150)
	# 				# #ax.set_ylim(0, 150)
	# 				# #show()
	# 				# plt.savefig('results/gmm/%s.png'%count)
	# 				# break
	# 				s=''
	# 				for p in clf.means_:
	# 					s+=str(p[0])+','+str(p[1])+'\t'
	# 				s+='|'
	# 				for w in clf.weights_:
	# 					s+=str(w)+','
	# 				s+='|'
	# 				#print clf.covars_.shape
	# 				for c in clf.covars_:
	# 					mat = c.reshape(4,)
	# 					for num in mat:
	# 						s+=str(num)+','
	# 					s+='\t'
	# 				wf.write(s+'\n')
					


def plotdata(fig,pts,data,img,alpha):
	#fig = figure()
	ax = fig.add_subplot(111, aspect='equal')
	# colors = ['#088A29','#FF8000','#B40404','#FFFF00','#0A2A29','#A9F5F2','#0174DF','#08088A',
	# 			'#8181F7','#8000FF','#2A0A29','#FF0040','#F5A9D0','#A9F5A9','#01DF01']

	colors = [(a,b,c) for a,b,c in zip(rand(len(data)),rand(len(data)),rand(len(data)))]
	colors = [c/sum(c) for c in colors]
	#colors = [c*255/2 for c in colors ]
	for j in range(len(data)):
		m = data[j][0]
		w = data[j][1]
		c = data[j][2]
		for i in range(len(m)):
			mean = m[i]
			weight = w[i]
			cov = c[i]
			v, x = np.linalg.eigh(cov)
			angle = np.arctan2(x[0][1], x[0][0])
			angle = 180 * angle / np.pi  # convert to degrees
			v /= np.linalg.norm(v)
			v *=40
			#mprint v
			if j==img:
				facecolor = colors[j]
			else:
				facecolor = 'None'
			ell = Ellipse(mean, v[0], v[1], 180 + angle, edgecolor=colors[j],fc=facecolor)
			#print 'set alpha:',alpha[j], 'for:',j
			ell.set_alpha(alpha[j])
			#ell.set_alpha(0.8)
			ax.add_artist(ell)
	
	ax.set_xlim(0, 200)
	ax.set_ylim(0, 150)
	for p in pts:
		ax.plot(p[0],p[1],'*',color='b')
	plt.hold(True)	
	#plt.draw()
	#show()
	

def main():
	data = readdata()
	


	k = 10
	i = 2
	j = 2
	test = []
	kp = corners('/home/lzhong/Downloads/Aldebaran/siftmatch/withprob/grey4/%s/%s_l%sc%s.png'%(k,k,i,j))
	for point in kp:
		point = tuple(point.pt)
		test.append(point)
	test = np.array(test)
	
	#fig = figure()
	n = len(data)
	rankscores = np.array([gmmscore(test[0],gmm) for gmm in data])
	ranks = sorted(range(len(rankscores)), key=lambda k:rankscores[k])
	match(data,test,ranks,rankscores,[test[0]])
	# 	if i==926:
	
	# 		# plt.plot(test[:,0],test[:,1],'.',color='b')
	# 		
	# 		# plt.axis('equal')
	
		
	# 	print i, gmmscore(clf,test)
	#print np.mean(predict)
	# print np.mean(clf.score(test)),np.mean(clf.score_samples(test)[0])
	# plt.plot(mean[:,0],mean[:,1],'x')
	# plt.show() 

	

if __name__ == '__main__':
	#readdata()
	main()
