"""
Parallel simulation of one- or two-solute phase separation under the framework of Flory-Huggins theory

Author: Zhengqing Zhou, Department of Biomedical Engineering, Duke University
Updated: Mar 25th, 2024
"""

import numpy as np
import pandas as pd
import time as timer
from scipy.interpolate import interp1d
from scipy.optimize import root
from solve_2phase import *
import multiprocessing as mp
import sys

def run(nn1, nn2, k_transcription1, k_transcription2, phi1A_b, phi2A_b, phi1B_b, phi2B_b,
        k_b, bmin, bmax, fh, v, seed):
    """

    :param nn1: the index of phi1 (corresponds to the range of np.arange(0.01,0.41,0.01)
    :param nn2: the index of phi2: nn2==0 for phi2=0 | nn2==1 for phi2=0.1
    :param k_transcription1: transcription rate of gene 1 (corresponds to protein 1)
    :param k_transcription2: transcription rate of gene 2 (corresponds to protein 2)
    :param phi1A_b: phi1A as a function of the tie-line intercept b, obtained from the phase diagram
    :param phi2A_b:
    :param phi1B_b:
    :param phi2B_b:
    :param k_b: the slope of the tie line as a function of the intercept b
    :param bmin: lower limit of intercept possible
    :param bmax: upper limit of intercept possible
    :param fh: defined Flory-Huggins parameters
    :param v: unit volume fraction
    :param seed: seed for generating random numbers
    :return:
    """
    np.random.seed(seed)
    k_translation = 5e-3  # Translation rate: per second (protein synthesis)
    tau_mRNA = 1e2  # mRNA half-life: ~ 5 min (Bernstein 2004 PNAS)
    tau_protein = 3.6e3  # Protein half-life: ~ 1 hour
    V_tot = 1
    Na = int(fh.Na)
    Nb = int(fh.Nb)

    # function to find the tie-lines with a specific phi1, phi2 value
    # point (phi1, phi2) should be located on the tie-line
    def find_tie_line(bb, phi1, phi2):
        return k_b(bb) * phi1 + bb - phi2

    # update the dilute and dense phase protein concentration, and the condensate volume
    def update_PS(total_protein_count1, total_protein_count2, b_x):
        phi1 = total_protein_count1 * Na * v / V_tot
        phi2 = total_protein_count2 * Nb * v / V_tot
        sol = root(find_tie_line, b_x, args=(phi1, phi2), tol=1e-10)
        b_x_tmp = sol.x
        PS = False
        # solve fails when the intercept exceeds the range of possible intercepts on the binodal curve
        # suggesting the point is outside the phase separation region
        if b_x_tmp > bmax or b_x_tmp < bmin:
            phi1A = phi1
            phi2A = phi2
            phi1B = 0
            phi2B = 0
            return PS, True, b_x, 1, 0, phi1A, phi2A, phi1B, phi2B
        # solve fails when the dilute phase (A) has higher volume fraction than the bulk concentration
        # or when the dense phase has lower volume fraction
        elif phi1A_b(b_x_tmp) >= phi1 or phi1B_b(b_x_tmp) <= phi1:
            phi1A = phi1
            phi2A = phi2
            phi1B = 0
            phi2B = 0
            return PS, True, b_x, 1, 0, phi1A, phi2A, phi1B, phi2B
        else:
            PS = True
            success = True
            if find_tie_line(b_x_tmp, phi1, phi2) > 1e-10:
                print("solution failed", phi1, phi2)
                success = False
            # consider the two phases are successfully solved,
            # and retrieve the concentrations of the proteins in the two phases and the condensate volume
            phi1B_tmp = phi1B_b(b_x_tmp)
            phi1A_tmp = phi1A_b(b_x_tmp)
            Va_tmp = (phi1B_tmp - phi1) / (phi1B_tmp - phi1A_tmp)

            dense_phase_volume = V_tot - Va_tmp
            phi1A = phi1A_b(b_x_tmp)
            phi2A = phi2A_b(b_x_tmp)
            phi1B = phi1B_b(b_x_tmp)
            phi2B = phi2B_b(b_x_tmp)
            return PS, success, float(b_x), float(Va_tmp), float(dense_phase_volume),\
                float(phi1A), float(phi2A), float(phi1B), float(phi2B)

    # Initial conditions
    mRNA_count1 = int(k_transcription1/(1/tau_mRNA))
    mRNA_count2 = int(k_transcription2/(1/tau_mRNA))
    total_protein_count1 = int(mRNA_count1*k_translation / (1 / tau_protein))
    total_protein_count2 = int(mRNA_count2*k_translation / (1 / tau_protein))
    dilute_phase_volume = 1
    dense_phase_volume = 0
    phi1A = total_protein_count1*Na*v/V_tot
    phi2A = total_protein_count2 * Nb * v / V_tot
    phi1B = 0
    phi2B = 0
    b_x = (bmin+bmax)/2
    args = update_PS(total_protein_count1, total_protein_count2, b_x)
    success = bool(args[1])
    if success:
        b_x, dilute_phase_volume, dense_phase_volume, phi1A, phi2A, phi1B, phi2B = args[2:]
    time = 0
    end_time = 100*60
    solve_success=[success]
    # trajectory of the simulation
    trajectory = [np.array([time, mRNA_count1, mRNA_count2, total_protein_count1, total_protein_count2,
                   dilute_phase_volume, dense_phase_volume, phi1A, phi2A, phi1B, phi2B],dtype=float)]

    # Gillespie algorithm
    t0=timer.perf_counter()
    while time<end_time:

        rates = np.array([
            k_transcription1,
            k_transcription2,
            k_translation * mRNA_count1,
            k_translation * mRNA_count2,
            1/tau_mRNA * mRNA_count1,
            1/tau_mRNA * mRNA_count2,
            1/tau_protein * total_protein_count1,
            1/tau_protein * total_protein_count2,
        ])

        total_rate = np.sum(rates)
        time_increment = np.random.exponential(1 / total_rate)

        # Choose the reaction based on its probability
        reaction = np.random.choice(range(len(rates)), p=rates / total_rate)

        if reaction == 0:  # Transcription of 1
            mRNA_count1 += 1
        elif reaction ==1:  # Transcription of 2
            mRNA_count2 += 1
        elif reaction == 2:  # Translation of 1
            total_protein_count1 += 1
            args = update_PS(total_protein_count1,total_protein_count2,b_x)
            PS = bool(args[0])
            success = bool(args[1])
            if success:
                b_x, dilute_phase_volume, dense_phase_volume, phi1A, phi2A, phi1B, phi2B = args[2:]
        elif reaction == 3:  # Translation of 2
            total_protein_count2 += 1
            args = update_PS(total_protein_count1,total_protein_count2,b_x)
            PS = bool(args[0])
            success = bool(args[1])
            if success:
                b_x, dilute_phase_volume, dense_phase_volume, phi1A, phi2A, phi1B, phi2B = args[2:]
        elif reaction == 4:  # mRNA degradation of 1
            mRNA_count1 -= 1
        elif reaction == 5:  # mRNA degradation of 2
            mRNA_count2 -= 1
        elif reaction == 6:  # protein degradation of 1
            total_protein_count1 -= 1
            args = update_PS(total_protein_count1,total_protein_count2,b_x)
            PS = bool(args[0])
            success = bool(args[1])
            if success:
                b_x, dilute_phase_volume, dense_phase_volume, phi1A, phi2A, phi1B, phi2B = args[2:]
        elif reaction == 7:  # protein degradation of 2
            total_protein_count2 -= 1
            args = update_PS(total_protein_count1,total_protein_count2,b_x)
            PS = bool(args[0])
            success = bool(args[1])
            if success:
                b_x, dilute_phase_volume, dense_phase_volume, phi1A, phi2A, phi1B, phi2B = args[2:]
        time += time_increment
        state=[time, mRNA_count1, mRNA_count2, total_protein_count1, total_protein_count2,
                   dilute_phase_volume, dense_phase_volume, phi1A, phi2A, phi1B, phi2B]
        state=np.array(state,dtype=float)
        trajectory.append(state)
        solve_success.append(success)
    # Extracting results for plotting
    trajectory=np.vstack(trajectory)
    # instead of recording the protein numbers, record the bulk concentrations instead
    trajectory[:,3]=trajectory[:,3]*Na*v/V_tot
    trajectory[:,4]=trajectory[:,4]*Nb*v/V_tot
    #determine if the simulation is successful (throw away when the root function return wrong values)
    solution_success = np.prod(solve_success)
    #calculate the time-weighted averaged mean and variance of all the variables
    trajectory_for_calculation = trajectory[100:,:]
    t_tmp = trajectory_for_calculation[:,0]
    delta_t = t_tmp[1:]-t_tmp[:-1]
    mean = np.average(trajectory_for_calculation[:-1,1:],axis=0,weights=delta_t)
    var = np.average(np.power(trajectory_for_calculation[:-1,1:]-mean,2),axis=0,weights=delta_t)
    std = np.sqrt(var)
    return solution_success, mean, std, trajectory, nn1, nn2

def run_merge(args):
    return run(*args)

if __name__=="__main__":
    #parameters of the system
    chi_12 = 0.8
    chi_11 = 2.3
    chi_22 = 0
    Na = 5
    Nb = 4
    Nc = 1
    suffix = int(sys.argv[1])

    fh = FloryHuggins(chi_12=chi_12, chi_11=chi_11, chi_22=chi_22, Na=Na, Nb=Nb, Nc=Nc)

    k_translation = 5e-3  # Translation rate: per second (protein synthesis)
    tau_mRNA = 1e2  # mRNA half-life: ~ 5 min (Bernstein 2004 PNAS)
    tau_protein = 3.6e3  # Protein half-life: ~ 1 hour
    V_tot = 1
    v = 2e-5
    #load the phase diagram
    binodal = np.loadtxt("phase diagram.dat")
    b = binodal[0]
    k = binodal[1]

    phi_1As = binodal[2]
    phi_2As = binodal[3]
    phi_1Bs = binodal[4]
    phi_2Bs = binodal[5]
    bmax = np.max(b)
    bmin = np.min(b)
    k_b = interp1d(x=b, y=k, kind='linear', fill_value='extrapolate')
    phi1A_b = interp1d(x=b, y=phi_1As, kind='linear', fill_value='extrapolate')
    phi2A_b = interp1d(x=b, y=phi_2As, kind='linear', fill_value='extrapolate')
    phi1B_b = interp1d(x=b, y=phi_1Bs, kind='linear', fill_value='extrapolate')
    phi2B_b = interp1d(x=b, y=phi_2Bs, kind='linear', fill_value='extrapolate')

    labels = ["mRNA1", "mRNA2", "phi1_bar","phi2_bar","V_dilute","V_dense",
              "phi1A", "phi2A", "phi1B", "phi2B"]

    N_sim = 100 # number of parallel simulations
    Phi1 = np.arange(0.01,0.41,0.01) # expected phi1 values when no noise exist
    record_phi1 = np.array([0.01,0.05,0.1,0.12,0.15,0.3])
    record_id = np.array(record_phi1/0.01-1,dtype=int) #ids for recording the simulation trajectories
    Phi2 = np.array([0, 0.1]) # expected phi2 values when no noise exist
    n1 = len(Phi1)
    n2 = len(Phi2)
    t0 = timer.perf_counter()

    Mean_kr1 = Phi1*V_tot/(Na*v*tau_mRNA*tau_protein*k_translation)
    Mean_kr2 = Phi2 * V_tot / (Nb * v * tau_mRNA * tau_protein * k_translation)
    args=[]
    for j in range(n2):
        Kr2 = Mean_kr2[j]
        for i in range(n1):
            mean1 = Mean_kr1[i]
            sigma1 = mean1 * 0.1
            k = (mean1 / sigma1) ** 2
            theta = (sigma1 ** 2) / mean1
            Kr1 = np.random.gamma(k, theta, size=(N_sim,))# the transcription rate is assumed to follow gamma distribution
            for s in range(N_sim):
                args.append(
                    [i,j,Kr1[s], Kr2, phi1A_b, phi2A_b, phi1B_b, phi2B_b, k_b, bmin, bmax, fh, v, s])
    # record the simulation results (mean, std, CV, and final value of the variables)
    # only the final values are used for downstream analysis
    # Condensation indicates whether phase separation ever took place in the simulation
    # id1, id2 corresponds to the index to Phi1 and Phi2
    df={"Condensation":[],"id1":[],"id2":[]}
    for i in range(10):
        df["mean_%s" % labels[i]]=[]
        df["std_%s" % labels[i]]=[]
        df["final_%s" % labels[i]] = []
        df["CV_%s" % labels[i]]=[]

    print("parallel simulation started")
    cores = mp.cpu_count()
    pool = mp.Pool(processes=int(cores))
    for success,mean,std,traj,nn1,nn2 in pool.imap(run_merge, args):
        nn1=int(nn1)
        nn2=int(nn2)
        if not success:
            continue
        for i in range(10):
            df["final_%s"%labels[i]].append(traj[-1,i+1])
            df["mean_%s" % labels[i]].append(mean[i])
            df["std_%s" % labels[i]].append(std[i])
            df["CV_%s" % labels[i]].append(std[i]/mean[i] if mean[i]>0 else 0)
        condensate_formation = bool(mean[5]>0)
        df["Condensation"].append(condensate_formation)
        df["id1"].append(nn1)
        df["id2"].append(nn2)
    df=pd.DataFrame(df)
    df.to_csv("Parallel_MC_fastPS_ext_%s.csv"%(suffix))
    pool.close()
    print("Parallel Computation finished, time used: %i min"%((timer.perf_counter()-t0)/60))