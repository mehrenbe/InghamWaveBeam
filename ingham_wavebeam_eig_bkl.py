import numpy as np
#import pylab as p
import scipy.linalg as s
#import symeig as sym
import sys
#ZZ=np.loadtxt('toto_eig_ell1_num10000.dat')
#ZZ=np.loadtxt('toto_eig.dat')
ZZ=np.loadtxt('toto_eig_ell1_num10000_bkl.dat')
#ZZ=np.loadtxt('toto_eig_ell10_num10000_bkl.dat')
#ZZ=np.loadtxt('toto_eig_ell10_num10000.dat')
#ZZ=np.loadtxt('toto_eig_ell2_num10000.dat')
ind=np.loadtxt('tata_eig_ell1_num10000_bkl.dat',dtype=int)
#ind=np.loadtxt('tata_eig_ell10_num10000_bkl.dat',dtype=int)
#ind=np.loadtxt('tata_eig.dat',dtype=int)
#print([[ind[i],ZZ[ind[i]],ZZ[ind[i]+1]-ZZ[ind[i]]] for i in range(len(ind))])
#sys.exit()
num2=np.size(ZZ)
num=num2
#print(num)
#A=np.zeros((num,num),dtype=complex)
T=3. #2.1
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

ind2 = np.zeros(len(ind)+1,dtype=int)
#ind2[0] = 0
ind2[-1] = num2
ind2[0:len(ind)] = ind
#print(ind2)
#[a:a+nloc] a<=ind[i] and ind[i*]+1<a+nloc

# for a in np.arange(N-n_loc):
# 	sa = 0
# 	while (ind2[sa]<a):
# 		sa+=1
# 	sb = sa
# 	while (ind2[sb]<a+n_loc-1):
# 		sb+=1
# 	sb-=1	
# 	if(sa<=sb):
# 		print(a,a+n_loc,sa,sb,ind2[sa],ind2[sb])
#sys.exit()			
			


#for a in n.arange(100,2001,100):
#for a in np.arange(2,num+1,1):
for a in np.arange(N-n_loc):
  #AA=A[0:a,0:a]  
	
	sa = 0
	while (ind2[sa]<a):
		sa+=1
	sb = sa
	while (ind2[sb]<a+n_loc-1):
		sb+=1
	sb-=1	
	if(sa<=sb):
		#print(a,a+n_loc,sa,sb,ind2[sa],ind2[sb])
		use_Q = 1
	else:
		use_Q = 0	
	
	for ii in range(n_loc):
		for jj in range(n_loc):
			i = ii+a
			j = jj+a
			if(i==j):
				AA[ii,jj] = T
			else:
				AA[ii,jj] = (np.exp(1j*(ZZ[i]-ZZ[j])*T)-1.)/(1j*(ZZ[i]-ZZ[j]))
	if(use_Q==1):
		#for ii in range(sa,sa+1):
		Q = np.identity(n_loc)
		#for ii in range(sa,sb+1):
			#print("sa",sa,sb,a,ii,ind2[ii])			
			#Q = np.array([[1.,0.],[0.,1.]])
			#Q = np.array([[1.,1./(ZZ[ind2[ii]]-ZZ[ind2[ii]+1])],[1.,-1./(ZZ[ind2[ii]]-ZZ[ind2[ii]+1])]])
			#Q = np.array([[1.,0.],[1.,ZZ[ind2[ii]+1]-ZZ[ind2[ii]]]])
			#Q = np.array([[1.,0.],[1.,ZZ[ind2[ii]+1]-ZZ[ind2[ii]]]])
			#
			#Q[ind2[ii]-a:ind2[ii]-a+2,ind2[ii]-a:ind2[ii]-a+2] = np.array([[1.,-1./(ZZ[ind2[ii]+1]-ZZ[ind2[ii]])],[0.,1./(ZZ[ind2[ii]+1]-ZZ[ind2[ii]])]])
			#
			#print(ind2[ii]-a,ind2[ii]+1-a)			
			#A_loc = np.array([[AA[ind2[ii]-a,ind2[ii]-a],AA[ind2[ii]-a,ind2[ii]+1-a]],
			#[AA[ind2[ii]+1-a,ind2[ii]-a],AA[ind2[ii]+1-a,ind2[ii]+1-a]]])
			#print(np.linalg.eigvals(AA))
			#AA = np.dot(np.transpose(Q),AA)
			#AA = np.dot(AA,Q)
			#A_loc = np.dot(np.transpose(Q),A_loc)
			#A_loc = np.dot(A_loc,Q)
			#AA[ind2[ii]-a,ind2[ii]-a] = A_loc[0,0]
			#AA[ind2[ii]-a,ind2[ii]+1-a] = A_loc[0,1]
			#AA[ind2[ii]+1-a,ind2[ii]-a] = A_loc[1,0]
			#AA[ind2[ii]+1-a,ind2[ii]+1-a] = A_loc[1,1]
			#print(np.linalg.eigvals(AA))
			#sys.exit()
		AA = np.dot(np.transpose(Q),AA)
		AA = np.dot(AA,Q)
			
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
