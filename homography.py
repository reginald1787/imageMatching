#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <string.h>

import numpy as np

class Affine:
 
    def __init__():
        #  m_Value 
          
        # int m_Index 
        #  m_AA[4] 
        # Affine() 
        self.m_minGX = 99999999   
    #Affine *m_Next 

  

def computeA(E,D,F,t,theta1):
    T = t.dot(t.T)

    r = D[0,0]**2
    s = D[1,1]**2
    P = 1 -T[0,0] + (E[0,0]**2)/r + (E[1,0]**2)/s
    Q = 1 -T[1,1] + (E[0,1]**2)/r + (E[1,1]**2)/s
  
    cos = np.arccos((E[0,1]*E[0,0]/r + E[1,0]*E[1,1]/s - T[0,1])/(P*Q))
    theta2 = theta1 - cos

    a = (r*P*np.cos(theta1) -E[0,0])/r 
    b = (s*P*np.sin(theta1) -E[1,0])/s 
    c = (r*P*np.cos(theta2) -E[0,1])/r 
    d = (s*P*np.sin(theta2) -E[1,1])/s

    return np.array([a,b,c,d]).reshape(2,2) 

def computeH(A,t,h31,h32):
    H = np.ones((3,3))
    H[:2,:2] = A
    H[:2,2] = t
    H[2,:2] = [h31,h32]
    return H

def computeD(R,N=3):
    D = np.zeros(N+1)
    n = len(R)
    for k in range(N+1):
        for point in R:
            D[k]+=pow(point[0],N-k)*pow(point[1],k)

    return D

def divide(Q,n=4,width=512,height=512):
    R = [[] for i in range(n)]
    for point in Q:
        if point[0]<width/2 and point[1]<height<2:
            R[1].append(point)
        elif point[0]<width/2:
            R[2].append(point)
        elif point[1]<height/2:
            R[3].append(point)
        else:
            R[4].append(point)

    return R



def project(H,P):
    Q = []
    for point in P:
        p = np.array([point[0],point[1],1]).reshape(3,1)
        q = H.dot(p)
        qx = q[0]/float(q[2])
        qy = q[1]/float(q[2])
        Q.append((qx,qy))
    return Q

def costFunc(H,P,Q,n=4):
    HP = project(H,P)

    RQ = divide(Q)
    RP = divide(HP)

    F = 0

    for i in range(n):
        DP = computeD(RP[i])
        DQ = computeD(RQ[i])
        F += (DP-DQ)**2 

    return F

def Normalize(p, nP,  dP, Scale, shift):
 
    # calculate center of the points set.
    mean =  np.array([0.0, 0.0])  
    #for(int i=0 i<nP i++)
    for i in range(np):
        mean[0] += p[2*i] 
        mean[1] += p[2*i+1] 
     
    mean[0] /= float(nP) 
    mean[1] /= float(nP) 


    maxX=-90000,minX=900000,maxY=-900000, minY=900000 
    #for(int i=0 i<nP i++)
    for i in range(np):
        #if(i<10)
        #    priNt ("in Normalize(): (%.3f,%.3f)\n", p[2*i], p[2*i+1]) 

        # move center of the points set to (0,0)
        dP[2*i]   = p[2*i]   - mean[0] 
        dP[2*i+1] = p[2*i+1] - mean[1] 
        # find the max mins
        if(dP[2*i] > maxX):
            maxX = dP[2*i] 
        if(dP[2*i] < minX):
            minX = dP[2*i] 

        # find the max mins
        if(dP[2*i+1] > maxY):
            maxY = dP[2*i+1] 
        if(dP[2*i+1] < minY):
            minY = dP[2*i+1] 
     

    # doing the normalization here
    # nX = (np.fabs(maxX) > np.fabs(minX))?np.fabs(maxX):np.fabs(minX) 
    # nY = (np.fabs(maxY) > np.fabs(minY))?np.fabs(maxY):np.fabs(minY) 
    nX = max(np.fabs(maxX), np.fabs(minX))
    nY = max(np.fabs(maxY), np.fabs(minY))

    #priNt ("nX = %.4f, nY = %.4f \n\n", nX, nY) 
    #for(int i=0 i<nP i++)
    for i in range(np): 
        dP[2*i]   = dP[2*i]/nX 
        dP[2*i+1] = dP[2*i+1]/nY 
     
    Scale[0] = nX 
    Scale[1] = nY 
    shift[0] = mean[0] 
    shift[1] = mean[1] 
 
#---------------------------------------------------------------------------------------------yy
def CheckQuality(mat2x2, p1, nP1,  p2, nP2, tp1, intv):
 

    #
    #

    # transform points set 1
    #for(int i=0 i<nP1 i++)
    for i in range(nP1): 
        tp1[2*i]  = mat2x2[0]*p1[2*i] + mat2x2[1]*p1[2*i+1] 
        tp1[2*i+1]= mat2x2[2]*p1[2*i] + mat2x2[3]*p1[2*i+1] 
     


    value = 0 
    mD = 9999999 
    maxD = -100 
    dist 

    #for(int i=0 i<nP1 i+= intv)
    for i in range(0,nP1,intv): 
        mD = 99999 
        #for(int j=0 j<nP2 j+=intv)
        for j in range(0,nP2,intv): 
            dist = (tp1[2*i] - p2[2*j])*(tp1[2*i] - p2[2*j] ) + 
                   (tp1[2*i+1]-p2[2*j+1])*(tp1[2*i+1]-p2[2*j+1]) 
            if(dist < mD):
                mD = dist 
         
        if(mD > maxD):
            maxD = mD  
     

    #priNt ("HELLO!!\n\n\n\n") 
    maxD2 = -100 

    #for(int i=0 i<nP2 i+=intv)
    for i in range(0,nP2,intv): 

     
        mD = 99999 
        #for(int j=0 j<nP1 j+=intv)
        for j in range(0,nP1,intv): 
 
            dist = (tp1[2*j] - p2[2*i])*  (tp1[2*j] - p2[2*i] ) + 
                   (tp1[2*j+1]-p2[2*i+1])*(tp1[2*j+1]-p2[2*i+1]) 
            if(dist < mD):
                mD = dist 
         
        if(mD > maxD2):
            maxD2 = mD  
     
    
    #w
    #priNt ("\n\n\n HELLO 2 2 2 2 \n\n\n") 


    return np.sqrt(maxD)+np.sqrt(maxD2) 

 
#---------------------------------------------------------------------------------------------yy
def GetAffine2D( p1,   p2,   nP1,   nP2,   S2,   cubicM2,   dScale,   tP1,   angles,   nAngles,
                  A,   B,   C,   D,   MM,   GX, head,   tempP1,
                  resultA2x2,   resultTT,  resultV):
 
    #mean1[] =  0.0,0.0 , mean2[]= 0,0  
    mean1 = [0.0,0.0]
    mean2 = 0.0
    # eigenvalues and vectors
    #eign[2], eigv[4] 
    #eign =np.zeros((1,2),dtype=np.float32)
    #eigv = np.zeros((1,4),dtype=np.float32)
    # S matrices in function Reduction
    #S1 = np.zeros((1,4),dtype=np.float32)
    #W1[4], W2[4]
    W1 = np.zeros((1,4),dtype=np.float32) 
    W2 = np.zeros((1,4),dtype=np.float32)
    # cubic moemnts
    #cubicM1[] =  0.0, 0.0,0.0,0.0  
    cubicM1 = np.zeros((1,4),dtype=np.float32)
    # only for P2 because P2 are already normalzied outside of GetAffine2D
    # *tP2 
    tP2 = p2 


    # 
    #minValues[20] 
    minIndex = np.zeros((1,20)) 
    nMins = 0 

    # calculate means
    #mean1[0] =0,  mean1[1] = 0 
    #for(int i=0 i<nP1 i++)
    for i in range(nP1): 
        mean1[0] += p1[i][0] 
        mean1[1] += p1[i][1] 
     
    mean1[0] = mean1[0]/float(nP1) 
    mean1[1] = mean1[1]/float(nP1) 

   

    # P2 are already normalized
    # /*
    # mean2[0] =0  mean2[1] = 0 
    # for(int i=0 i<nP2 i++)
     
    #     mean2[0] += p2[i*2] 
    #     mean2[1] += p2[i*2+1] 
     
    # mean2[0] = mean2[0]/()nP2 
    # mean2[1] = mean2[1]/()nP2 
    # */
    #------------------------------

    #priNt ("GetAffine2x2: mean1: %.3f, %.3f \n\n", mean2[0], mean2[1]) 

    # normalize points
    #for(int i=0 i<nP1 i++)
    for i in range(np1): 
        #if(i<10)
        #    priNt ("tP2:(%.3f,%.3f)  ", tP2[i*2], tP2[i*2+1]) 
        
        tP1[i]  = ( p1[i][0]   - mean1[0], p1[i][0] - mean1[1]  )
        
     

    # p2 are already normalized
    #for(int i=0 i<nP2 i++)
    # 
        #tP2[i*2]    =p2[i*2]    -mean2[0] 
        #tP2[i*2+1]  =p2[i*2+1]  -mean2[1] 
    #    tP2[i*2]    =p2[i*2] 
    #    tP2[i*2+1]  =p2[i*2+1] 
    # 

    # doing the function reduction here

    # build covariance matrix
    #C1 = np.zeros((1,4),dtype = np.float32) 
    #for(int i=0 i<4 i++) C1[i]=0.0 
    C1 = np.zeros((2,2))
    #for(int i=0 i<nP1 i++)
    for i in range(nP1): 
        C1[0,0] += tP1[i][0]**2 
        C1[1,1] += tP1[i][1]**2 
        C1[0,1] += tP1[i][0]*tP1[i][1] 
     
    C1[1,0] = C1[0,1] 

#    for(int i=0 i<nP2 i++)
#     
#        C2[0] += tP2[i*2]*tP2[i*2] 
#        C2[3] += tP2[i*2+1]*tP2[i*2+1] 
#        C2[1] += tP2[i*2]*tP2[i*2+1] 
#     
#    C2[2] = C2[1] 

    # the values of C1 and C2 are correct compared to the matlab code
    #
#    priNt ("\n\nC1:(%.3f,%.3f,%.3f,%.3f)",C1[0],C1[1],C1[2], C1[3]) 
#    priNt ("\nC2:(%.3f,%.3f,%.3f,%.3f) \n",C2[0],C2[1],C2[2], C2[3]) 
        
    # eigen
    #Eigen2by2Sym(C1,eign,eigv) 
    eign,eigv = np.linalg.eigh(C1)
#    priNt ("\n\n\neign:(%f,%f), eigv:(%.3f,%.3f,%.3f,%.3f)\n\n\n",eign[0],eign[1],eigv[0],eigv[1], eigv[2], eigv[3]) 
    # calculate 1/sqrt(eigenvalues)
    eign[0] = 1.0/np.sqrt(eign[0]),  eign[1] = 1.0/np.sqrt(eign[1]) 

    # S1[0] = eigv[0]*eigv[0]*eign[0] + eigv[1]*eigv[1]*eign[1] 
    # S1[1] = eigv[0]*eigv[2]*eign[0] + eigv[1]*eigv[3]*eign[1] 
    # S1[2] = S1[1] 
    # S1[3] = eigv[2]*eigv[2]*eign[0] + eigv[3]*eigv[3]*eign[1] 

    temp = np.zeros((2,2))
    temp[0,0] = eigv[0,0]
    temp[1,1] = eigv[1,1]
    S1 = eigv.dot(temp).dot(eigv.T)
    #Eigen2by2Sym(C2,eign,eigv) 
#    priNt ("\n\n\neign:(%f,%f), eigv:(%.3f,%.3f,%.3f,%.3f)\n\n\n",eign[0],eign[1],eigv[0],eigv[1], eigv[2], eigv[3]) 
    #eign[0] = 1.0/sqrt(eign[0])  eign[1] = 1.0/sqrt(eign[1])  

    #S2[0] = eigv[0]*eigv[0]*eign[0] + eigv[1]*eigv[1]*eign[1] 
    #S2[1] = eigv[0]*eigv[2]*eign[0] + eigv[1]*eigv[3]*eign[1] 
    #S2[2] = S2[1] 
    #S2[3] = eigv[2]*eigv[2]*eign[0] + eigv[3]*eigv[3]*eign[1] 
    # end of reduction function 

# values are correct to this point

    #priNt ("\n\nS1:(%.3f,%.3f,%.3f,%.3f)\n", S1[0], S1[1], S1[2], S1[3]) 
    #priNt ("S2:(%.4f,%.4f,%.4f,%.4f)\n", S2[0], S2[1], S2[2], S2[3]) 


    # calculating cubic moments
    #for(int i=0 i<nP1 i++)
    for i in range(nP1): 
        cubicM1[0] += pow(float(tP1[2*i]),3) 
        cubicM1[1] += pow(float(tP1[2*i]),2)*tP1[2*i+1] 
        cubicM1[2] += tP1[2*i]*pow(float(tP1[2*i+1]),2) 
        cubicM1[3] += pow(float(tP1[2*i+1]),3) 
     

#    for(int i=0 i<nP2 i++)
#     
#        cubicM2[0] += pow(()tP2[2*i],3) 
#        cubicM2[1] += pow(()tP2[2*i],2)*tP2[2*i+1] 
#        cubicM2[2] += tP2[2*i]*pow(()tP2[2*i+1],2) 
#        cubicM2[3] += pow(()tP2[2*i+1],3) 
#     
    # 
    # for(int i=0 i<4 i++)
     
    #     cubicM1[i] = cubicM1[i]/()nP1 

    cubicM1 = cubicM1/float(nP1)
#        cubicM2[i] = cubicM2[i]/()nP2   
     
    # correct up to here
    #priNt ("\ncubicM1 :(%f,%f,%f,%f)",cubicM1[0],cubicM1[1],cubicM1[2],cubicM1[3]) 
    #priNt ("\ncubicM2 :(%f,%f,%f,%f)\n",cubicM2[0],cubicM2[1],cubicM2[2],cubicM2[3]) 

    # W2 = inv(S2)
    Inverse2x2(S2, W2) 
    # for(int i=0 i<4 i++)  
    #     W1[i] = S1[i] 

    W1 = S1

    #priNt ("\nW2:(%f,%f,%f,%f)",W2[0],W2[1],W2[2],W2[3]) 
    #priNt ("\nW1:(%f,%f,%f,%f)\n",W1[0],W1[1],W1[2],W1[3]) 

    # Calculate A, B, C, D
    #for(int i=0 i<nAngles*2 i++)
    for i in range(nAngles*2): 
        A[i] = angles[4*i]*W2[0]*W1[0] + angles[4*i+1]*W2[0]*W1[1] + angles[4*i+2]*W2[1]*W1[0] + angles[4*i+3]*W2[1]*W1[1] 
        B[i] = angles[4*i]*W2[0]*W1[2] + angles[4*i+1]*W2[0]*W1[3] + angles[4*i+2]*W2[1]*W1[2] + angles[4*i+3]*W2[1]*W1[3] 
        C[i] = angles[4*i]*W2[2]*W1[0] + angles[4*i+1]*W2[2]*W1[1] + angles[4*i+2]*W2[3]*W1[0] + angles[4*i+3]*W2[3]*W1[1] 
        D[i] = angles[4*i]*W2[2]*W1[2] + angles[4*i+1]*W2[2]*W1[3] + angles[4*i+2]*W2[3]*W1[2] + angles[4*i+3]*W2[3]*W1[3] 
     

    # /*for(int i=2*nAngles-10 i<nAngles*2 i++)
     
    #      priNt ("A[%d]=%.4f  ",i,A[i]) 
    #      priNt ("B[%d]=%.4f  ",i,B[i]) 
    #      priNt ("C[%d]=%.4f  ",i,C[i]) 
    #      priNt ("D[%d]=%.4f  ",i,D[i]) 
    #      priNt ("\n") 
    #  */
    
    
    # calculting MM.
    #for(int i=0 i<nAngles*2 i++)
    for i in range(nAngles*2): 

        MM[4*i]     = pow(float(A[i],3))*cubicM1[0]    + 3*pow(float(A[i],2))*B[i] * cubicM1[1] + 3*A[i]*B[i]*B[i]*cubicM1[2] + pow(float(B[i],3))*cubicM1[3] 
        MM[4*i+1]   = A[i]*A[i]*C[i]*cubicM1[0] + (2*A[i]*B[i]*C[i]+A[i]*A[i]*D[i])*cubicM1[1] + (2*A[i]*B[i]*D[i]+B[i]*B[i]*C[i])*cubicM1[2] + B[i]*B[i]*D[i]*cubicM1[3] 
        MM[4*i+2]   = C[i]*C[i]*A[i]*cubicM1[0] + (2*C[i]*D[i]*A[i]+C[i]*C[i]*B[i])*cubicM1[1] + (2*C[i]*D[i]*B[i]+D[i]*D[i]*A[i])*cubicM1[2] + D[i]*D[i]*B[i]*cubicM1[3] 
        MM[4*i+3]   = pow(float(C[i],3))*cubicM1[0]    + 3*C[i]*C[i]*D[i]*cubicM1[1]     + 3*C[i]*D[i]*D[i]*cubicM1[2] + pow(float(D[i],3))*cubicM1[3] 

        # this is the part that calculate KX. 
        MM[4*i]     -= cubicM2[0] 
        MM[4*i+1]   -= cubicM2[1] 
        MM[4*i+2]   -= cubicM2[2] 
        MM[4*i+3]   -= cubicM2[3] 
        
        if(i>nAngles*2-4):
                
            #priNt ("KX[%d]=%.8f,KX[%d]=%.8f,KX[%d]=%.8f,KX[%d]=%.8f,\n",4*i,MM[4*i],4*i+1,MM[4*i+1],4*i+2,MM[4*i+2],4*i+3,MM[4*i+3]) 
         
        # calculate GX
        
            GX[i]   = pow(float((MM[4*i]/(np.fabs(cubicM2[0])+0.0001)),2)) 
                    + pow(float((MM[4*i+1]/(np.fabs(cubicM2[1])+0.0001)),2)) 
                    + pow(float((MM[4*i+2]/(np.fabs(cubicM2[2])+0.0001)),2)) 
                    + pow(float((MM[4*i+3]/(np.fabs(cubicM2[3])+0.0001)),2))  
     

    # /*for(int i=nAngles*2-3  i<2*nAngles i++)
     
    #     priNt ("GX[%d]=%.4f   ",i,GX[i]) 
    #  */

    # find the global/local minimums of GX
    minV = 9999999 
    minI = -1 
    tempidx1 = 2 

    #for(int i=1 i<nAngles*2-1 i++)
    for i in range(1,nAngles*2-1): 

        if(GX[i]<=GX[i-1] and GX[i]<=GX[i+1] and GX[i] < head[2].m_minGX ):
           
            tempidx1 = 2 
            #for(int j=1 j>=0 j--)
            for j in range(1,-1,-1):
                if(GX[i] < head[j].m_minGX):
                   tempidx1 = j  
               
            #for(int j=2 j>= j++)
            #
            #
            if(tempidx1 ==2):
             
                head[tempidx1].m_minGX = GX[i] 
                head[tempidx1].m_Index = i 
             
            elif(tempidx1 == 1):
             
                head[2].m_minGX = head[1].m_minGX 
                head[2].m_Index = head[1].m_Index 
                #
                head[1].m_minGX = GX[i] 
                head[1].m_Index = i 
             
            elif(tempidx1 == 0):
             
                
                head[2].m_minGX = head[1].m_minGX 
                head[2].m_Index = head[1].m_Index 
                #
                head[1].m_minGX = head[0].m_minGX 
                head[1].m_Index = head[0].m_Index 

                head[0].m_minGX = GX[i] 
                head[0].m_Index = i 
             
         
     

    #minValues[0] = minV 
    #minIndex[0] = minI 
    nMins = 3 

    #theta 
    # RR[4], iS2[4], RRS1[4] 
    # AA[4] 
    RR = np.zeros((1,4),dtype=np.float32)
    iS2 = np.zeros((1,4),dtype=np.float32)
    AA = np.zeros((1,4),dtype=np.float32)
    RRS1 = np.zeros((1,4),dtype=np.float32)
    minValue = 9999999#,value 


    
    # identity matrix
    AA[0] = 1,  AA[1] = 0,  AA[2] = 0,  AA[3] = 1 

    #for(int i=0 i<4 i++)
    #    resultA2x2[i] = AA[i] 
    resultA2x2 = AA

    minValue = CheckQuality(AA, tP1, nP1, tP2, nP2,tempP1, dScale) 
    
    #priNt ("\n\n\nQuality of identity: %.4f\n\n", minValue) 
    #priNt ("Indices: %d, %d, %d \n", head[0].m_Index, head[1].m_Index, head[2].m_Index) 
    #priNt ("GXMAX 3: %.3f,%.3f,%.3f\n", GX[head[0].m_Index],GX[head[1].m_Index],GX[head[2].m_Index]) 
    #priNt ("GX165~167: %.3f,%.3f,%.3f,%.3f\n", GX[164],GX[165],GX[166],GX[167]) 



    #int index11 

    #for(int i=0 i<nMins i++)
    for i in range(nMins): 

        # # build the rotational matrix
        # # get theta
        # theta = angles[head[i].m_Index] 

        # # build the rotational matrix
        # RR[0] = cos(theta)  RR[1] = -sin(theta)  RR[2] = sin(theta)  RR[3] = cos(theta) 
        # # if it's a reflection
        index11 = head[i].m_Index 
        #for(int k=0 k<4 k++)
        for k in range(4):
            RR[k] = angles[4*index11+k] 


        if(head[i].m_Index > nAngles):
         
            RR[2] = -1*RR[2] 
            RR[3] = -1*RR[3] 
#            priNt ("REFLECTION!!!!!!!!!!\n") 
         
        # This is the part calculating
        # AA = inv(S2)*RR*S1 
        matrixMult2x2(RR,S1,RRS1) 
        Inverse2x2(S2,iS2) 
        #priNt ("Theta = %.4f\n",theta) 
        #priNt ("RR =(%.4f,%.4f,%.4f,%.4f )\n",RR[0],RR[1],RR[2],RR[3]) 
        #priNt ("inverseS2=(%.3f,%.3f,%.3f,%.3f )\n",iS2[0],iS2[1],iS2[2],iS2[3]) 
        #priNt ("RRS1=(%.3f,%.3f,%.3f,%.3f )\n",RRS1[0],RRS1[1],RRS1[2],RRS1[3]) 

        # AA here is Affine2x2(if criteria satisfies.....)
        matrixMult2x2(iS2,RRS1,AA) 
        
        value = CheckQuality(AA,tP1, nP1, tP2, nP2, tempP1, dScale) 
        #
        #priNt ("[%.4f\t%.4f]\n",AA[0], AA[1]) 
        #priNt ("[%.4f\t%.4f]\n",AA[2], AA[3]) 
        #priNt ("Quality %d is %.4f\n",i,value) 
        if(value < minValue):
         
            minValue = value 
            # for(int j=0 j<4 j++)
            #     resultA2x2[j] = AA[j] 
            resultA2x2 = AA
           # priNt ("Min values found. The index is %d.\n", head[i].m_Index) 
         
     

    #priNt ("resultA2x2=(%.4f,%.4f,%.4f,%.4f)\n",resultA2x2[0],resultA2x2[1],resultA2x2[2],resultA2x2[3] ) 


    tMean[2] 
    tMean[0] = resultA2x2[0]*mean1[0] + resultA2x2[1]*mean1[1] 
    tMean[1] = resultA2x2[2]*mean1[0] + resultA2x2[3]*mean1[1] 

    resultTT[0] = mean2[0] - tMean[0] 
    resultTT[1] = mean2[1] - tMean[1] 
    resultV = minValue 
    
 

def mexFunction(nlhs, plhs,  nrhs, 
                nP1,nP2,uP1,uP2,range1,range2,dint):
 
    # syntax 
    # [HMatrix] = homography(P1, P2, range1, range2, dScale)

    # if(nrhs !=5 )
     
    #     mexErrMsgTxt("[HMatrix, Scale1, Translate1, Scale2, Translate2, transformedP1] = homography(P1, P2, range1, range2, dScale)\n\n") 
    #     return 
     
    # int nP1, nP2 
    # # points set normalized 
    #  *P1, *P2 
    # covarance of normalized points set 2
     # cov2[] =  0.0,0.0,0.0,0.0  
     # cubicM2[] =  0.0, 0.0, 0.0, 0.0  
    cov2 = np.zeros((1,4),dtype=np.float32)
    cubicM1 = np.zeros((1,4),dtype=np.float32)
    # un-normalized points set
     # *uP1,*uP2 
     # *transformedP1 
     # *newP1 
     # *tmpP1, *tmpP1_2, *tmpP2 
     # *tempdist 


    # normalization factors for p1 and p2
    # HH1[2], HH2[2], TT1[2], TT2[2] 
    HH1 = np.zeros((1,2),dtype=np.float32)
    HH2 = np.zeros((1,2),dtype=np.float32)
    TT1 = np.zeros((1,2),dtype=np.float32)
    TT2 = np.zeros((1,2),dtype=np.float32)
    # ranges
    # *range1, *range2 
    # interval for calculating the distance. 
    #int dint 

    # get numbers of points 
    # nP1 = mxGetN(prhs[0]) 
    # nP2 = mxGetN(prhs[1]) 

    # # read the points in. 
    # uP1 = mxGetPr(prhs[0]) 
    # uP2 = mxGetPr(prhs[1]) 
    # # read range
    # range1 = mxGetPr(prhs[2]) 
    # range2 = mxGetPr(prhs[3]) 
    # # read resolution
    # dint = mxGetScalar(prhs[4]) 
    
    # alloc
    newP1   = np.zeros((1,nP1*2)) 
    tmpP1   = np.zeros((1,nP1*2)) 
    tmpP1_2 = np.zeros((1,nP1*2)) 
    tmpP2   = np.zeros((1,nP2*2)) 
    P1      = np.zeros((1,nP1*2)) 
    P2      = np.zeros((1,P2*2)) 
    tempdist = np.zeros((1,nP1)) 

    transformedP1 = np.zeros((1,nP1*2)) 
    # generating angles. 
    nAngles = int(360.0/0.1) 
    d_inc = 0.1/180.0*3.14159265 
    theta = 0 

     # *RMatrix = new [nAngles*4*2] 
     # *A = new [nAngles*2] 
     # *B = new [nAngles*2] 
     # *C = new [nAngles*2] 
     # *D = new [nAngles*2] 
     # *MM = new [nAngles*4*2] 
     # *GX = new [nAngles*2] 

    RMatrix = np.zeros((1,nAngles*4*2))
    A = np.zeros((1,nAngles*2)) 
    B = np.zeros((1,nAngles*2)) 
    C = np.zeros((1,nAngles*2)) 
    D = np.zeros((1,nAngles*2)) 
    MM = np.zeros((1,nAngles*4*2))
    GX = np.zeros((1,nAngles*2)) 

    #priNt ("LOOK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n\n") 
    #for(int i=0 i<nAngles  i++, theta+= d_inc)
    for i in range(nAngles):
        theta+=d_inc 
        RMatrix[4*i  ] = cos(theta)          
        RMatrix[4*i+1] = -sin(theta)          
        RMatrix[4*i+2] = -RMatrix[4*i+1]          
        RMatrix[4*i+3] = RMatrix[4*i] 
        #if(i<20)
          #  priNt ("cos(%.5f)=%.4f  ", theta,RMatrix[4*i]) 
     
    #priNt ("LOOK_++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n") 
    # reflection part
    #for(int i=nAngles i<2*nAngles i++)
    for i range(nAngles,2*nAngles): 
        RMatrix[4*i  ] = RMatrix[4*(i-nAngles)] 
        RMatrix[4*i+1] = RMatrix[4*(i-nAngles)+1] 
        RMatrix[4*i+2] = -1*RMatrix[4*(i-nAngles)+2] 
        RMatrix[4*i+3] = -1*RMatrix[4*(i-nAngles)+3] 
     
     

    #for(int i=0 i<10 i++)
   #  
    #    priNt ("(%.3f,%.3f) ",uP1[2*i],uP1[2*i+1]) 
    # 
    #
    #priNt ("\n") 

    #----------------create 3 componenets

    #Affine head[3] 
    head = []


    #priNt ("Normalizing points.....\n\n") 
    Normalize(uP1, nP1, P1, HH1, TT1) 
    Normalize(uP2, nP2, P2, HH2, TT2) 

    # Calculate the covariance of normalized points set 2
    #for(int i=0 i<nP2 i++)
    for i in range(nP2): 
        cov2[0] += P2[i*2]*P2[i*2] 
        cov2[3] += P2[i*2+1]*P2[i*2+1] 
        cov2[1] += P2[i*2]*P2[i*2+1] 
     
    cov2[2] = cov2[1] 
    
    eign = np.zeros((1,2))
    eigv = np.zeros((1,4))
    S2 = np.zeros((1,4)) 

    # get eigenvector and values
    Eigen2by2Sym(cov2,eign,eigv) 
    eign[0] = 1.0/np.sqrt(eign[0]),  eign[1] = 1.0/np.sqrt(eign[1])  

    S2[0] = eigv[0]*eigv[0]*eign[0] + eigv[1]*eigv[1]*eign[1] 
    S2[1] = eigv[0]*eigv[2]*eign[0] + eigv[1]*eigv[3]*eign[1] 
    S2[2] = S2[1] 
    S2[3] = eigv[2]*eigv[2]*eign[0] + eigv[3]*eigv[3]*eign[1] 

    # cubic moment of P2
    #for(int i=0 i<nP2 i++)
    for i in range(nP2): 
        cubicM2[0] += pow(float(P2[2*i]),3) 
        cubicM2[1] += pow(float(P2[2*i]),2)*P2[2*i+1] 
        cubicM2[2] += P2[2*i]*pow(float(P2[2*i+1]),2) 
        cubicM2[3] += pow(float(P2[2*i+1]),3) 
     
    # for(int i=0 i<4 i++)
    #     cubicM2[i] = cubicM2[i]/()nP2 

    cubicM2 = cubicM2/float(np2)
    
    #for(int i=0 i<10 i++)
    # 
    #    priNt ("(%.3f,%.3f)  ",P1[2*i],P1[2*i+1]) 
    # 
    #priNt ("\n") 


    # counter 
    count = 0 
    # the denominator to calculte the H(p)
    #aa = new [nP1]
    aa = np.zeros((1,nP1)) 
    minD = 999999 
    rAA = np.zeros((1,4))
    rTT = np.zeros((1,2))#  rValue 
    minValue = 999999 

    ghw = np.zeros((1,3)) 
    minAA = np.zeros((1,4)) 
    minTT = np.zeros((1,2)) 

    #priNt ("range 11 --> (%.3f, %.3f, %.3f)\n",range1[0],range1[1],range1[2]) 
    #priNt ("range 22 --> (%.3f, %.3f, %.3f)\n",range2[0],range2[1],range2[2]) 

    # iterate through the two ranges
    #for( r1 = range1[0]  r1 <= range1[2]  r1 += range1[1])
    for r1 in range(range1[0],range1[2]+1,range1[1]): 
        #for( r2 = range2[0]  r2 <= range2[2]  r2 += range2[1])
        for r2 in range(range2[0],range2[2]+1,range2[1]): 
            count ++ 
            # calculate the denominator
            minD = 999999 
            #memset(aa,0,sizeof()*nP1)
            aa = np.zeros((1,nP1)) 
            #for(int i=0 i<nP1 i++)
            for i in range(nP1): 
                aa[i] = np.fabs(r1*P1[i*2] + r2*P1[i*2+1] + 1) 
                if(aa[i] < minD):
                    minD = aa[i]  
             # end of for int i
            

            if(r1==0.18 and r2 == 0.12):
             
                print "The minD of 0.18/0.12 is %f\n\n",minD
             

            # minD here is ww 
            #priNt ("(r1 %f,%f) ww = %f, P1(%f,%f) \n\n", r1,r2, minD, P1[0], P1[1]) 
            if(minD > 0.0001):
             
                # calculate new P1
                #for(int i=0 i<nP1 i++)
                for i in range(nP1): 
                    newP1[2*i]   = P1[2*i]/aa[i] 
                    newP1[2*i+1] = P1[2*i+1]/aa[i] 
                    #if(i<10)
                    #    priNt ("newP1:(%.3f,%.3f)  ",newP1[2*i],newP1[2*i+1]) 
                 
                # put GetAffine2D here
#                priNt ("cov2 = (%.4f,%.4f,%.4f,%.4f)\n",cov2[0],cov2[1],cov2[2],cov2[3]) 
                # reset head
                #for(int i=0 i<3 i++)    head[i].m_minGX = 9999999 
                for i in range(3):
                    temphead = Affine()
                    head.append(temphead)

                GetAffine2D(newP1, P2, nP1, nP2, S2, cubicM2, dint, tmpP1, RMatrix, nAngles, A, B, C, D, MM, GX, head,tmpP1_2,
                              rAA,rTT, rValue) 
                
                #priNt ("\n(r1:%.2f,r2:%.2f)\t[%.2f  %.2f] [%.3f]\n", r1,r2, rAA[0], rAA[1], rTT[0]) 
                #priNt ("\t\t\t[%.2f  %.2f] [%.3f]    rValue = %.3f\n", rAA[2], rAA[3], rTT[1], rValue) 

                # calculate transformed P1
                #for(int i=0 i<nP1 i++)
                # 
                #    transformedP1[2*i]   = rAA[0]*newP1[2*i]+rAA[1]*newP1[2*i+1] + rTT[0] 
                #    transformedP1[2*i+1] = rAA[2]*newP1[2*i]+rAA[3]*newP1[2*i+1] + rTT[1] 
                # #

                if(rValue < minValue): 
                 
                    minValue = rValue 
                    #for(int i=0 i<4 i++)
                    for i in range(4): 
                        minAA[i] = rAA[i] 
                        #priNt ("minAA[%d]=%.5f",i,minAA[i]) 
                     
                    minTT[0] = rTT[0],  minTT[1] = rTT[1] 
                    ghw[0] = r1 , ghw[1] = r2 
                    #priNt ("minValue happened!! %.5f\n",minValue) 
                 
#                priNt ("%.4f\t%.4f\n",r1,r2)  
              # end of if minD > 0.0001
            else:
             
                print "minD is too small! @(%f) minD = %f\n",%(r1,r2) 
                
#                 
                     
             

         # end of for r2
     # end of for r1
    

    #priNt ("minTT=(%.4f, %.4f)\n", minTT[0],minTT[1]) 
    #priNt ("minAA=(%.4f,%.4f,%.4f,%.4f)\n", minAA[0], minAA[1], minAA[2], minAA[3]) 
    # 
    #priNt ("Finalizing..........\n") 

    # returning affine matrix
    #
    a = np.zeros((1,9)) ,b = np.zeros((1,9)), finalMatrix= np.zeros((1,4),dtype=np.float32) #0.0, 0.0, 0.0,   0.0, 0.0, 0.0,  0.0,0.0,0.0  
    a[0] = minAA[0] ,    a[1] = minAA[1]  ,   a[2] = minTT[0] 
    a[3] = minAA[2] ,   a[4] = minAA[3]  ,   a[5] = minTT[1] 
    a[6] = 0.0      ,    a[7] = 0.0       ,   a[8] = 1.0 

    b[0] = 1.0     ,     b[1] = 0.0       ,   b[2] = 0.0 
    b[3] = 0.0     ,     b[4] = 1.0      ,    b[5] = 0.0 
    b[6] = ghw[0] ,     b[7] = ghw[1]   ,    b[8] = 1.0    
# /*
#     for(int i=0 i<9 i++)
     
#         if(i%3==0)
#             priNt ("\n") 
#         priNt ("%.4f   ",a[i]) 
     
#     priNt ("a and then b") 
#     for(int i=0 i<9 i++)
     
#         if(i%3==0)
#             priNt ("\n") 
#         priNt ("%.4f   ",a[i]) 
     
# */


    #plhs[0] = mxCreateDoubleMatrix(3,3, mxREAL) 
    ptr = np.zeros((1,9),dtype=float32)
    # *ptr = mxGetPr(plhs[0]) 
    # for(int i=0 i<9 i++)
    #     ptr[i] = 0.0 
    
    print ("\n") 
    #for(int i=0 i<3 i++)
    for i in range(3): 
        #for(int j=0 j<3 j++)
        for j in range(3): 
        #    for(int k=0 k<3 k++)
            for k in range(3):
                finalMatrix[i*3+j] += a[i*3+k]*b[(k)*3+j] 
            #priNt ("%.4f\t",finalMatrix[i*3+j]) 
         
        #priNt ("\n") 
     
    #priNt ("\n") 

    # normalize ptr
    #for(int i=0 i<9 i++)
    for i in range(9): 
        ptr[i] = finalMatrix[i]/finalMatrix[8] 
        #priNt ("ptr[%d]=%.5f \n",i,finalMatrix[i]) 
     



    # HH1
    plhs[1] = mxCreateDoubleMatrix(2,1,mxREAL) 
    ptr = mxGetPr(plhs[1]) 
    ptr[0] = HH1[0] 
    ptr[1] = HH1[1] 


    plhs[2] = mxCreateDoubleMatrix(2,1,mxREAL) 
    ptr = mxGetPr(plhs[2]) 
    ptr[0] = TT1[0] 
    ptr[1] = TT1[1] 

    plhs[3] = mxCreateDoubleMatrix(2,1,mxREAL) 
    ptr = mxGetPr(plhs[3]) 
    ptr[0] = HH2[0] 
    ptr[1] = HH2[1] 

    plhs[4] = mxCreateDoubleMatrix(2,1,mxREAL) 
    ptr = mxGetPr(plhs[4]) 
    ptr[0] = TT2[0] 
    ptr[1] = TT2[1] 

    # transformed P1
    plhs[5] = mxCreateDoubleMatrix(3,nP1,mxREAL) 
    ptr = mxGetPr(plhs[5]) 
    
    memset(ptr,0,3*nP1*sizeof()) 
    
    for(int i=0 i<nP1 i++)
     
        for(int j=0 j<3 j++)
         
            for(int k=0 k<2 k++)
             
                ptr[3*i+j] += finalMatrix[3*j+k]*P1[2*i+k]  
             
            ptr[3*i+j] +=finalMatrix[3*j+2]*1 
         
        ptr[3*i]   = ptr[3*i]/ptr[3*i+2] 
        ptr[3*i+1] = ptr[3*i+1]/ptr[3*i+2] 
        ptr[3*i+2] = 1.0 
        
     


    return 
 

