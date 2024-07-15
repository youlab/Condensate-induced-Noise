# get the binodal points of a two polymer - solvent system on the phi1 - phi2 plane
# Author: Zhengqing Zhou, Duke University

import matplotlib.pyplot as plt
import numpy as np
import time as timer
from scipy.optimize import root
from solve_2phase import *
from scipy.interpolate import interp1d

c1 = "#4682b4"
c2 = "#5da269"
c3 = "#e366a4"

chi_12 = 0.8
chi_11 = 2.3
chi_22 = 0
Na = 5
Nb = 4
Nc = 1
fh = FloryHuggins(chi_12=chi_12, chi_11=chi_11, chi_22=chi_22,
                    Na=Na, Nb=Nb, Nc=Nc)
# solve the critical point
guess = [0.36,0.17]
solution = root(fh.critical_point,guess,tol=1e-10)
print(f"critical point: {solution.x}, solve success: {solution.success}")
print(f"equality: {fh.critical_point(solution.x)}")
cp=solution.x
#Init guesses for a paired phases on the same tie line: phi_1A, phi_1B, phi_2A, phi_2B
init_configs=[{'phi_1A' : 0.10805, 'phi_2A' : 0, 'phi_1B' : 0.59, 'phi_2B' : 0},]
#define the range of sampling phi1A (until hitting the critical point)
ranges = [np.linspace(init_configs[0]['phi_1A'],cp[0],num=300,endpoint=True),]
n_iter = len(ranges)
total_num = len(np.hstack(ranges))

binodal_surf = np.empty([total_num],float)
phi_1A_surf = np.empty([total_num],float)
phi_2A_surf = np.empty([total_num],float)
phi_1B_surf = np.empty([total_num],float)
phi_2B_surf = np.empty([total_num],float)

count = 0
t0 = timer.perf_counter()
#iteratively solve for pairs of binodal points
for i in range(n_iter):
    init_config=init_configs[i]
    phi_1As=ranges[i]
    for j,phi_1A in enumerate(phi_1As):

        ch = Ternary_2phase(phi_1A=phi_1A, chi_12 = chi_12, chi_11 = chi_11, chi_22 = chi_22,
                            Na=Na, Nb=Nb,Nc=Nc)

        if j == 0:
            phi_2A = init_config['phi_2A']
            phi_1B = init_config['phi_1B']
            phi_2B = init_config['phi_2B']
            x0 = np.array([phi_2A,phi_1B,phi_2B])
        else:
            x0 = [roots[0],roots[1],roots[2]]

        solution = root(ch.equilibrium,x0,
                        jac=ch.eq_jacobian,
                        tol=1e-10)

        roots = solution.x
        phi_1s=np.array([phi_1A,roots[1]])
        phi_2s = np.array([roots[0], roots[2]])
        if (np.abs(phi_1s[0]-phi_1s[1])>1e-3
            and solution.success):
            phi_1B = phi_1s[1]
            phi_2A = phi_2s[0]
            phi_2B = phi_2s[1]
            found = True
        else:
            phi_1A = np.nan
            phi_1B = np.nan
            phi_2A = np.nan
            phi_2B = np.nan
            found = False


        binodal_surf[count] = bool(found)
        phi_1A_surf[count] = phi_1A
        phi_2A_surf[count] = phi_2A
        phi_1B_surf[count] = phi_1B
        phi_2B_surf[count] = phi_2B

        count += 1
    print("%i th iteration ended, detected binodal points = %i"%(i+1,np.nansum(binodal_surf)))
t1 = timer.perf_counter()
print("search finished, time spent: %1.1fs"%(t1-t0))

#plot all the solved binodal points
fig_pd,ax_pd=plt.subplots(1,1,figsize=(4,4))
for j in range(len(phi_1A_surf)):
    ax_pd.plot([phi_1A_surf[j], phi_1B_surf[j]], [phi_2A_surf[j], phi_2B_surf[j]],lw=0.5,
            color="#808080", alpha=0.5)
ax_pd.scatter(phi_1A_surf, phi_2A_surf,c=c1,alpha=0.5)
ax_pd.scatter(phi_1B_surf, phi_2B_surf,c=c2,alpha=0.5)
phi_1s = np.logspace(-3.5,0,num=200,endpoint=True)
phi_2s = np.logspace(-3.5,0,num=200,endpoint=True)
XX,YY = np.meshgrid(phi_1s,phi_2s)
F = np.ones_like(XX)*np.nan
detH = np.ones_like(XX)*np.nan
for idx, _ in np.ndenumerate(XX):
    x = XX[idx]
    y = YY[idx]
    if x+y>1:
        continue
    F[idx] = ch.freeEnergy(x,y)
    detH[idx] = ch.Spinodal(x,y)
cs = ax_pd.contour(XX,YY,detH,levels=[0],zorder=10,colors="k")
ax_pd.set_xlim([0,1])
ax_pd.set_ylim([0,1])
ax_pd.set_xticks([0,0.5,1])
ax_pd.set_yticks([0,0.5,1])
ax_pd.set_xlabel(r"$\phi_1$")
ax_pd.set_ylabel(r"$\phi_2$")
fig_pd.tight_layout()

#interpolate (and extrapolate) from the solved binodal points to get the entire binodal curve with even sampled points
#interpolation are done by generating phi1A, phi2A, phi1B, phi2B as function of b, which is the intercept of the tie-line
#k is the slope of the tie-line
phi_1As=phi_1A_surf
phi_1Bs=phi_1B_surf
phi_2As=phi_2A_surf
phi_2Bs=phi_2B_surf
k = (phi_2Bs-phi_2As)/(phi_1Bs-phi_1As)
b = phi_2Bs - k*phi_1Bs
idx=np.isnan(b)
phi_1As = phi_1As[~idx]
phi_2As = phi_2As[~idx]
phi_1Bs = phi_1Bs[~idx]
phi_2Bs = phi_2Bs[~idx]
k = k[~idx]
b = b[~idx]
k_b = interp1d(x=b,y=k,kind='linear',fill_value='extrapolate')
phi1A_b = interp1d(x=b,y=phi_1As,kind='linear',fill_value='extrapolate')
phi2A_b = interp1d(x=b,y=phi_2As,kind='linear',fill_value='extrapolate')
phi1B_b = interp1d(x=b,y=phi_1Bs,kind='linear',fill_value='extrapolate')
phi2B_b = interp1d(x=b,y=phi_2Bs,kind='linear',fill_value='extrapolate')
bmax=np.max(b)
bmin=np.min(b)
bmin_crit = root(lambda x: phi2A_b(x), bmin)
bmax_crit = root(lambda x: phi1A_b(x)-phi1B_b(x),bmax)
print(f"maximum b: {bmax_crit.x}, solve success:{bmax_crit.success}")
bb=np.linspace(bmin_crit.x,bmax_crit.x,1000)
pp1A = phi1A_b(bb)
pp1B = phi1B_b(bb)
pp2A = phi2A_b(bb)
pp2B = phi2B_b(bb)
kk = k_b(bb)
# every individual intercept b can be projected
# to one value of phi1A, phi2A, phi1B, phi2B, and slope k
# it is thus a good practice to interpolate the binodal points based on b
fig0,ax0=plt.subplots(1,1,figsize=(4,4))
ax0.plot(bb,pp1A)
ax0.scatter(b,phi_1As,s=10,label="$\phi_{1A}$")
ax0.plot(bb,pp1B)
ax0.scatter(b,phi_1Bs,s=10,label="$\phi_{1B}$")
ax0.plot(bb,pp2A)
ax0.scatter(b,phi_2As,s=10,label="$\phi_{2A}$")
ax0.plot(bb,pp2B)
ax0.scatter(b,phi_2Bs,s=10,label="$\phi_{2B}$")
ax0.plot(bb,kk)
ax0.scatter(b,k,s=10,label="k")
ax0.legend()
data=np.hstack([bb,kk,pp1A,pp2A,pp1B,pp2B]).T
#save the data for future stochastic simulations
#np.savetxt("pd_data/phase diagram.dat",data)

#plot the interpolated phase diagram
fig_pd2,ax_pd2=plt.subplots(1,1,figsize=(4,2.8))
for j in range(len(bb)):
    ax_pd2.plot([pp1A[j], pp1B[j]], [pp2A[j], pp2B[j]],lw=0.5,
            color="#808080", alpha=0.5,zorder=0)
ax_pd2.scatter(pp1A, pp2A,c=c1,alpha=0.5,zorder=15)
ax_pd2.scatter(pp1B, pp2B,c=c2,alpha=0.5,zorder=15)
cs = ax_pd2.contour(XX,YY,detH,levels=[0],zorder=10,colors="k")
ax_pd2.scatter(cp[0], cp[1], c=c3, s=50, marker="s",zorder=20)
ax_pd2.set_xlim([0,0.7])
ax_pd2.set_ylim([0,0.4])
ax_pd2.set_xticks([0,0.3,0.6])
ax_pd2.set_yticks([0,0.2,0.4])
ax_pd2.set_xlabel(r"$\phi_1$")
ax_pd2.set_ylabel(r"$\phi_2$")
fig_pd2.tight_layout()
plt.show()
