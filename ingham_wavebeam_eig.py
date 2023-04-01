import numpy as np
#import pylab as p
import scipy.linalg as s
#import symeig as sym
import sys
#ZZ=np.loadtxt('toto_eig_ell1_num10000.dat')
#ZZ=np.loadtxt('toto_eig.dat')
ZZ=np.loadtxt('toto_eig_ell1_num10000_bkl.dat')
#ZZ=np.loadtxt('toto_eig_ell10_num10000.dat')
#ZZ=np.loadtxt('toto_eig_ell2_num10000.dat')
ind=np.loadtxt('tata_eig_ell1_num10000_bkl.dat',dtype=int)
print([[ind[2*i],ZZ[ind[2*i]],ZZ[ind[2*i+1]]-ZZ[ind[2*i]]] for i in range(len(ind)//2)])
sys.exit()
num2=np.size(ZZ)
num=num2
#print(num)
#A=np.zeros((num,num),dtype=complex)
T=3 #2.1
# for i in range(num):
# 	for j in range(num):
# 		if(i==j):
# 			A[i,j] = T
# 		else:
# 			A[i,j] = (np.exp(1j*(ZZ[i]-ZZ[j])*T)-1.)/(1j*(ZZ[i]-ZZ[j]))

#imA=p.load('immat2.dat')

#A=reA+complex(0,1)*imA
N=num#np.shape(A)
#N=N[0]
n_loc = 100

AA=np.zeros((n_loc,n_loc),dtype=complex)

#[a:a+nloc] ind[]


#for a in n.arange(100,2001,100):
#for a in np.arange(2,num+1,1):
for a in np.arange(N-n_loc):
  #AA=A[0:a,0:a]  
	for ii in range(n_loc):
		for jj in range(n_loc):
			i = ii+a
			j = jj+a
			if(i==j):
				AA[ii,jj] = T
			else:
				AA[ii,jj] = (np.exp(1j*(ZZ[i]-ZZ[j])*T)-1.)/(1j*(ZZ[i]-ZZ[j]))
	#AA=A[a:a+n_loc,a:a+n_loc]
	V=np.linalg.eigvals(AA)
	#V=s.eigh(AA,eigvals_only=True,eigvals=(0,1))
	#V=sym.symeig(AA,eigenvectors=0,range=(0,1))
	reV=V.real
	imV=V.imag
	reVmin=reV.min()
	reVmax=reV.max()
	imVmin=imV.min()
	imVmax=imV.max()
	#print("%d %1.20lg %.20lg %1.20lg %.20lg" % (np.size(V),reVmin,reVmax,imVmin,imVmax))
	print("%d %1.20lg %.20lg %1.20lg %.20lg" % (a+1,reVmin,reVmax,imVmin,imVmax))
	#print("%d %1.20lg" % (a,V[0]))
