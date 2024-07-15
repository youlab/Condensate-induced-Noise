# Condensate-induced-Noise

[![DOI](https://zenodo.org/badge/797473364.svg)](https://zenodo.org/doi/10.5281/zenodo.12741347)

This is the repository for the codes used in the paper _Biomolecular condensates regulate cellular electrochemical equilibria_ to phase separation when coupled with stochastic gene expression.

There are two folders in this repository. _phasediagram_ contains files to generate the phase diagrams based on Flory-Huggins theory of ternary mixture. The phase diagram is further used in stochastic simulations.

The directory _parallel simulation_ contains script _parallel_MCFH_external.py_, which is used to simulate the phase separation when coupled with stochastic gene expression using gillespie algorihm. Two other scripts, _save_MCFH_data_ext.py_, and _visualization_MCFH_external.py_, should be ran consecutively after using the parallel computation script to generate the _Parallel_MC_fastPS_ext_ csv files for data processing and visualization.
