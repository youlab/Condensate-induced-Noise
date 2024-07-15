"""
Processing of the simulated stochastic phase separation
Calculate the coefficient of variance of different variables
using their final values in each independent simulation

Input files:

Parallel_MC_fastPS_fixed_p2_ext_1.csv
Parallel_MC_fastPS_fixed_p2_ext_2.csv
Parallel_MC_fastPS_fixed_p2_ext_3.csv

Author: Zhengqing Zhou, Department of Biomedical Engineering, Duke University
Updated: Mar 25th, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time as timer
from scipy.interpolate import interp1d
from scipy.optimize import root
from solve_phasediagram.solve_2phase import *
from solve_phasediagram.solve_2phase_SC import FloryHuggins as FloryHuggins_SC

if __name__=="__main__":

    n1 = 40 # number of phi1 choices (np.linspace(0.01,0.41,0.01))
    n2 = 2# number of phi2 choices: 0 or 0.1

    lower_limit = -4 # cv smaller than 1e-4 is taken as 1e-4

    Phi1 = np.arange(0.01,0.41,0.01,dtype=float)
    Phi2 = np.array([0, 0.1])
    colors=['#808080',"#4682b4"]
    # corresponding variable names in the input and output files
    # phi1_bar / bulk_phi1: average protein 1 concentration cross the whole cell
    # phi1A / dilute_phi1: protein 1 concentration in the dilute phase / cytoplasma
    # V_dense / BMC_vol: volume of the condensate
    var_shown = ["phi1_bar","phi1A", "V_dense"]
    var_record = ["bulk_phi1","dilute_phi1","BMC_vol"]
    titles=["single-polymer condensate","two-polymer condensate"]
    # specifying the recorded information that will be input into the visualization_MCFH_external.py file for visualization
    # recording the coefficient of variance (std/mean) of the three variables across three individual simulations
    df_record1 = {"phi1": [],
                 "log(CV_bulk_phi1)_t1": [], "log(CV_bulk_phi1)_t2": [], "log(CV_bulk_phi1)_t3": [],
                 "log(CV_dilute_phi1)_t1": [], "log(CV_dilute_phi1)_t2": [], "log(CV_dilute_phi1)_t3": [],
                 "log(CV_BMC_vol)_t1": [], "log(CV_BMC_vol)_t2": [], "log(CV_BMC_vol)_t3": [], }
    df_record2 = {"phi1": [],
                 "log(CV_bulk_phi1)_t1": [], "log(CV_bulk_phi1)_t2": [], "log(CV_bulk_phi1)_t3": [],
                 "log(CV_dilute_phi1)_t1": [], "log(CV_dilute_phi1)_t2": [], "log(CV_dilute_phi1)_t3": [],
                 "log(CV_BMC_vol)_t1": [], "log(CV_BMC_vol)_t2": [], "log(CV_BMC_vol)_t3": [], }
    # i==0: single-polymer phase separation, with phi2=0
    # i==1: two-polymer phase separation, with phi2=0.1
    for i in range(2):
        df_record = df_record1 if i==0 else df_record2
        df_record["phi1"]=Phi1.tolist()
        for trial in range(1,4,1):
            df = pd.read_csv("./MC_FH/Parallel_MC_fastPS_ext_%i.csv" % (trial))
            indices = df["id1"].tolist()
            df_i=df[df["id2"]==i]# only load the data related to a specific phi2 value
            for j in range(n1):
                idx=np.where(np.array(df_i["id1"])==j)[0]
                df_ij=df_i.iloc[idx,:]
                for k,label in enumerate(var_shown):
                    y = np.array(df_ij["final_%s"%label])
                    mean = np.mean(y)
                    std = np.std(y)
                    if mean == 0:
                        cv = 0
                    else:
                        cv = std / mean
                    if cv<10**lower_limit:# cv smaller than 1e-4 is taken as 1e-4
                        cv=10**lower_limit

                    df_record["log(CV_%s)_t%i"%(var_record[k],trial)].append(np.log10(cv))
    # save the data
    df_record1=pd.DataFrame(df_record1)
    df_record2 = pd.DataFrame(df_record2)
    with pd.ExcelWriter('MCFH_external_noise.xlsx', engine='xlsxwriter') as writer:
        df_record1.to_excel(writer, index=False, sheet_name=titles[0])
        df_record2.to_excel(writer, index=False, sheet_name=titles[1])
