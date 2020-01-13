#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det
from mpl_toolkits.mplot3d import Axes3D


def uni_Gaussian_Distribution(x, mean, sigma):
    E = np.exp((-1/(2*((sigma)**2)))*((x-mean)**2))
    y = (1/(((2*np.pi)*((sigma)**2))**(1/2))) * E
    return y

#=========================sample data 정의

N = 200

X1 = np.linspace(0, np.pi*2, N)[...,None]
Y1 = np.sin(X1) + 0.2*np.random.randn(N)
X2 = np.linspace(0, np.pi*2, N)[...,None]
Y2 = np.cos(X2) + 0.2*np.random.randn(N)

X = np.append(X1,X2, axis = 0)
Y = np.append(Y1,Y2, axis = 0)[...,None]

#Data
Data1 = 0.5*np.random.randn(int(N/2))
Data2 = 0.5*np.random.randn(int(N/2))+3
Data = np.append(Data1,Data2)[...,None]
print(Data.shape)
y_data = np.zeros(N)
plt.plot(Data, y_data, "r*")

# True
T_data = np.linspace(-3,5,N)
T1 = uni_Gaussian_Distribution(T_data,0,0.5)
T2 = uni_Gaussian_Distribution(T_data,3,0.5)
T3 = (T1+T2)/2
ax = plt.axes()
plt.plot(T_data, T3, 'k')

plt.show()
#==================== random initialize parameter =======================
x= Data

K = 2                           # 클래스 수
D = x.shape[1]               # Dimensional of Data
p = np.random.rand(D,K)         # 어느 클래스에 속할 확률 vector
m = np.random.rand(D,K)         # mean vector
# m = np.array([0.5,0.5])[None,...]

cov = np.zeros((K,D,D))         # Covariance matrix
for i in range(K):
    cov1 = np.random.rand(D)*np.eye(D)
    cov[i] = cov1
# cov = np.array([[0.5],[2.5]])[...,None]

def dot(x,y,z):
    a = np.dot(x,y)
    b = np.dot(a,z)
    return b

def Gaussian_Distribution(x, D, mean, sigma):
    E = np.exp((-1/2)*dot((x-mean),(inv(sigma)),(x-mean).T))
    y = (1/((2*np.pi)**(D/2)))*(1/((det(sigma))**(1/2)))*E        
    return y

def responsibility(x,n,k, mean, cov, pi):
    global N,K,D
    m = mean
    p = pi
    r = np.zeros(D)
    r_k = p[:,k]*Gaussian_Distribution(x[n], D, m[:, k], cov[k])
    for j in range(K):
        r1 = p[:,j]*Gaussian_Distribution(x[n], D, m[:, j], cov[j])
        r += r1
    return r_k/r


#============================ EM algorithm =================================
G=np.zeros(len(T_data))
r = np.zeros((N,K))
sum_r = np.zeros((D,K))
sum_rx = np.zeros((D,K))
sum_rd = np.zeros((K,D,D))

T = 12
for t in range(T) :
    # E-step
    for n in range(N):
        for k in range(K):
            r[n,k] = responsibility(x,n,k, m,cov,p)

    sum_r = np.zeros((D,K))
    sum_rx = np.zeros((D,K))
    sum_rd = np.zeros((K,D,D))
    #M-step
    for f in range(K):
        for h in range(N):
            re = r[h,f]
            rex = re * x[h]

            sum_r[:,f] += re
            sum_rx[:,f] += rex

        m[:,f] = sum_rx[:,f]/sum_r[:,f]
        p[:,f] = sum_r[:,f]/N
        for h in range(N):
            re = r[h,f]
            red = re * np.dot((x[h]-m[:,f]),(x[h]-m[:,f]).T)

            sum_rd[f] += red

        cov[f] = sum_rd[f]/sum_r[:,f]
        
        print("mean%i : %0.4f , p%i : %0.4f" %(f, m[:,f], f, p[:,f]) )
        print("cov%i : " %(f),cov[f])

    print("="*40)
    for v in range(len(T_data)):
        G1 = Gaussian_Distribution(T_data[v],D,m[:,0],cov[0])
        G2 = Gaussian_Distribution(T_data[v],D,m[:,1],cov[1])
        g = p[:,0]*G1+p[:,1]*G2
        G[v] = g

    if t % 1 == 0 :
        plt.figure()
        plt.plot(Data, y_data, "r*")
        plt.plot(T_data, T3, 'k')
        plt.plot(T_data, G, "c")
        plt.show()
    

    

    



