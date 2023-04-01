#python ingham_wavebeam_eig_diag_bkl.py
import numpy as np
import matplotlib.pyplot as plt

#for column in data.T:
#  plt.plot(data[:,0], column)

#b="(($2))";set title "electric energy"
#b="3";set title "L2 norm"
#b="(0.5*($3)**2)";set title "0.5*(L2 norm)^2"
#b="(($5))";set title "L1 norm"
#b="(0.5*$5)";set title "L1 norm"
#b="(0.5*($4+$2**2))";set title "total energy"
#p 'topeSW_N64_d2limT400.dat' u 1:@b w l t 'd=2 lim',\
#'topeSW_N64_d2DaTeT400.dat' u 1:@b w l t 'd=2 DaTe',\
#'topeSW_N64_d2nolimT400.dat' u 1:@b w l t 'd=2 no lim',\
#'topeSW_N64_splT400.dat' u 1:@b w l t 'cubic splines',\
#'topeSW_N64_wenoT400.dat' u 1:@b w l t 'SLWENO5',\
#set output "peSW_ee_fig2_N64.pdf"
#p 'topeSW_N64_d2limT400.dat' u 1:@b w l t 'd=2 lim',\
#'topeSW_N64_d4limT400.dat' u 1:@b w l t 'd=4 lim',\
#'topeSW_N64_d4DaTeT400.dat' u 1:@b w l t 'd=4 DaTe',\
#'topeSW_N64_d4nolimT400.dat' u 1:@b w l t 'd=4 no lim',\

#fig, axs = plt.subplots(9, 1,figsize=(15,7.5))
fig, axs = plt.subplots(8, 1,figsize=(7.5,15))
ymin = 0. #1.e-6 #81.
ymax = 10. #3.5 #0.2 #87.
xmin = 0.
xmax= 100.

left = 0.115  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.95     # the top of the subplots of the figure
wspace = 0.  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.1  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height


def phi1(x):
	#return x[:,1]
	return x[:,2]

def phi0(x):
	return np.sqrt(x[:,0])

	#return np.abs(x/1.9970030457005845692-1.)
	#return np.fabs(x[:,4]/x[0,4]-1.) #L1
	#return np.fabs(x[:,2]/x[0,2]-1.) #L2
	#return np.fabs((x[:,3]+x[:,1]**2)/(x[0,3]+x[0,1]**2)-1.) #te
	#return np.fabs(0.5*(x[:,3]+x[:,1]**2)/(3.*np.pi)-1.)


plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

#for i in range(8):
	#for j in range(1):
		#axs[i,j].set_yscale('log')
#	axs[i].set_ylim((ymin,ymax))

# axs[0].set_ylim((0.,2.1))
# axs[1].set_ylim((0.,2.1))
# axs[2].set_ylim((0.,2.1))
# axs[3].set_ylim((0.,2.1))
# axs[4].set_ylim((0.,0.025))
# axs[5].set_ylim((0.,0.025))
# axs[6].set_ylim((0.,9.))
# axs[7].set_ylim((0.,9.))

# axs[0].set_ylim((3.7,4.1))
# axs[1].set_ylim((3.7,4.1))
# axs[2].set_ylim((3.9,4.7))
# axs[3].set_ylim((3.9,4.7))
# axs[4].set_ylim((3.,11.))
# axs[5].set_ylim((3.,11.))
# axs[6].set_ylim((11.,20.))
# axs[7].set_ylim((11.,20.))


data1 = np.loadtxt('to_eig_ell1_pmax10000_T2p1_Nloc100_bkl.dat')
data2 = np.loadtxt('to_eig_ell1_pmax10000_T2p1_Nloc100_bkl_dd.dat')
data3 = np.loadtxt('to_eig_ell1_pmax10000_T3_Nloc100_bkl.dat')
data4 = np.loadtxt('to_eig_ell1_pmax10000_T3_Nloc100_bkl_dd.dat')
data5 = np.loadtxt('to_eig_ell10_pmax10000_T2p1_Nloc100_bkl.dat')
data6 = np.loadtxt('to_eig_ell10_pmax10000_T2p1_Nloc100_bkl_dd.dat')
data7 = np.loadtxt('to_eig_ell10_pmax10000_T10_Nloc100_bkl.dat')
data8 = np.loadtxt('to_eig_ell10_pmax10000_T10_Nloc100_bkl_dd.dat')
#data9 = np.loadtxt('toto_ell50_pmax10000.dat')


axs[0].plot(phi0(data1),phi1(data1),label='$\ell=1,\ T=2.1,\ N_{loc}=100$')
axs[1].plot(phi0(data2),phi1(data2),label='$\ell=1,\ T=2.1,\ N_{loc}=100\ dd$')
axs[2].plot(phi0(data3),phi1(data3),label='$\ell=1,\ T=3,\ N_{loc}=100$')
axs[3].plot(phi0(data4),phi1(data4),label='$\ell=1,\ T=3,\ N_{loc}=100\ dd$')
axs[4].plot(phi0(data5),phi1(data5),label='$\ell=10,\ T=2.1,\ N_{loc}=100$')
axs[5].plot(phi0(data6),phi1(data6),label='$\ell=10,\ T=2.1,\ N_{loc}=100\ dd$')
axs[6].plot(phi0(data7),phi1(data7),label='$\ell=10,\ T=10,\ N_{loc}=100$')
axs[7].plot(phi0(data8),phi1(data8),label='$\ell=10,\ T=10,\ N_{loc}=100\ dd$')
#axs[8].plot(phi0(data9),phi1(data9),label='ell=50')
#axs[0,j].plot(data5[:,0],phi1(data2),label='slweno5')
#axs[0,j].plot(data4[:,0],phi(data3),label='cubic\nsplines')
#axs[0].set_title('32x32')
#axs[0].legend(loc=(-0.5,0.15))
axs[0].legend(loc=(50,50.))
axs[0].legend()
#axs[0,j].set_ylim((ymin,ymax))
for i in range(8):
	axs[i].set_xlim((xmin,xmax))
	axs[i].grid(True)
	#axs[i].legend(loc=(50,50.))
	axs[i].legend()
for i in range(7):
	xticks = axs[i].xaxis.get_major_ticks()
	xticks[-1].label1.set_visible(False)
	xticks[0].label1.set_visible(False)
	xticks[1].label1.set_visible(False)
	xticks[2].label1.set_visible(False)
	xticks[3].label1.set_visible(False)
	xticks[4].label1.set_visible(False)
plt.suptitle('Maximal eigenvalue vs $n^{1/2}$')
#plt.suptitle('Minimal eigenvalue vs $n^{1/2}$')
plt.savefig('toto.png')
plt.show()
