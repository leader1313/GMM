#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det, pinv
from mpl_toolkits.mplot3d import Axes3D





def dot(x,y,z):
    a = np.dot(x,y)
    b = np.dot(a,z)
    return b

def uni_Gaussian_Distribution(x, mean, sigma):
    E = np.exp((-1/(2*((sigma)**2)))*((x-mean)**2))
    y = (1/(((2*np.pi)*((sigma)**2))**(1/2))) * E
    return y

def Gaussian_Distribution(x, D, mean, sigma):
    E = np.exp((-1/2)*dot((x-mean).T,(inv(sigma)),(x-mean)))
    y = (1/((2*np.pi)**(D/2)))*(1/((det(sigma))**(1/2)))*E        
    return y

def uni_Gaussian_bias(x, mean, sigma):
    B = np.exp((-1/(2*((sigma)**2)))*((x-mean)**2))
    return B

def Gaussian_bias(x, mean, sigma):
    B = np.exp((-1/2)*dot((x-mean).T,(inv(sigma)),(x-mean)))
    return B

def responsibility( t, n, k, Weight, Phi, cov, prior):
    global N,K,D
    wk_bn = np.dot(Weight[:,k].T,Phi[n])
    p = prior
    sum_r = np.zeros(D)[...,None]
    r_k = p[k]*uni_Gaussian_Distribution(t, wk_bn, cov)
    for j in range(K):
        r1 = p[j]*uni_Gaussian_Distribution(t, np.dot(Weight[:,j].T, Phi[n]), cov)
        sum_r += r1
    return r_k/sum_r

#=========================sample data 정의================================

N = 100
n = int(N/2)
X1 = np.linspace(0, np.pi*2, n)[:,None]
Y1 = np.sin(X1) + np.random.randn(n)[:,None] * 0.2
X2 = np.linspace(0, np.pi*2, n)[:,None]
Y2 = np.cos(X2) + np.random.randn(n)[:,None] * 0.2

X = np.append(X1,X2, axis = 0)
Y = np.append(Y1,Y2, axis = 0)

plt.plot(X, Y, "g*")

# True
true_y1 = np.sin(X1)
true_y2 = np.cos(X2)

plt.plot(X1, true_y1, 'k')
plt.plot(X2, true_y2, 'k')

# plt.show()
#==================== random initialize parameter =======================


K = 2                                       # model 수
D = X.shape[1]                              # Dimensional of Data
M = 9                                       # Number of model
prior = np.random.rand(K)                   # 어느 클래스에 속할 확률 vector
Weight = np.random.rand(M,K)                # mean vector
cov = np.zeros(1)+2                  # variance vector

Phi = np.zeros((N,M))
phi_mean = np.zeros(D)+1
phi_sigma = np.diag((np.zeros(D)+1))

for m in range(M):
    for n in range(N):
        Phi[n,m] = Gaussian_bias(X[n],m*phi_mean,phi_sigma)

#============================ EM algorithm =================================

r = np.zeros((N,K))             # responsibility
sum_r = np.zeros((D,K))
sum_rd = np.zeros(1)


T = 20
for t in range(T) :
    # E-step
    R = np.zeros((K,N,N))
    for n in range(N):
        for k in range(K):
            r[n,k] = responsibility(Y[n], n, k, Weight, Phi, cov, prior)
            R[k,n,n] = r[n,k]
    
    sum_r = np.zeros((D,K))
    sum_rd = np.zeros(1)
    
    

    #M-step
    for k in range(K):
        for n in range(N):
            re = r[n,k]
            sum_r[:,k] += re

        prior[k] = sum_r[:,k]/N

        invPRP = inv(dot(Phi.T,R[k],Phi))
        PtRy = dot(Phi.T,R[k],Y)
        Weight[:,k][:,None] = np.dot(invPRP, PtRy)
    
        for n in range(N):
            re = r[n,k]
            d = (Y[n]-np.dot(Weight[:,k].T, Phi[n]))**2

            sum_rd += re * d

    cov = sum_rd/N
    
    # print("mean%i : %0.4f , p%i : %0.4f" %(t, np.dot(Weight[:,k].T, Phi[n]), t, p) )
    print("cov%i : " %(t),cov)

    print("="*40)
    test_num = 100
    predict1 = np.zeros(test_num)[:,None]
    predict2 = np.zeros(test_num)[:,None]
    test_x = np.linspace(0, np.pi *2, test_num)[:, None]
    Pi = np.zeros((test_num,M))
    for m in range(M):
        for n in range(test_num):
            Pi[n,m] = Gaussian_bias(test_x[n],m*phi_mean,phi_sigma)
    
    for n in range(test_num):
        predict1[n] = np.dot(Weight[:,0].T,Pi[n])
        predict2[n] = np.dot(Weight[:,1].T,Pi[n])
    ss = np.sqrt(cov)

    if t % 5 == 0 :
        plt.figure()
        plt.plot(X, Y, "g*")
        line1 = plt.plot(test_x, predict1, 'b')
        plt.plot(test_x, predict1 + ss,"--", color = line1[0].get_color() )
        plt.plot(test_x, predict1 - ss,"--", color = line1[0].get_color() )
        
        line2 = plt.plot(test_x, predict2, 'r')
        plt.plot(test_x, predict2 + ss,"--", color = line2[0].get_color() )
        plt.plot(test_x, predict2 - ss,"--", color = line2[0].get_color() )
        
        plt.show()
    

    

    



