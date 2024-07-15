#Numerical Solver for Flory Huggins Theory in two-component (polymer-solvent) systems capable of separting into two phases
#Integrating two methods online with significant modifications
#Source 1: Dr. Ryan McGorty at University of San Diego
#https://github.com/rmcgorty/PhaseSeparation
#Source 2: https://github.com/samueljmcameron/floryhugginsternary
#Author: Zhengqing Zhou, Duke University
#Most Recent Update: Mar 26th, 2024

import numpy as np

class FloryHuggins():
    def __init__(self, chi=2.3, Na=5, Nv=1):
        """
        Initialise all attributes.

        Parameters
        ----------
        chi : float
            Interaction between the solute and solvent.
        Na: int
            lengths of solute
        Nv: int
            grid size
        """

        self.chi = chi
        self.Na = Na
        self.Nv = Nv

        return

    def freeEnergy(self,phi):
        Na=self.Na
        chi=self.chi
        Nv=self.Nv
        out1=np.array(-np.ones_like(phi)*1e10,dtype=float)
        entropy1=phi/Na*np.log(phi, where=phi > 0, out=out1)
        out2 = np.array(-np.ones_like(phi)*1e10, dtype=float)
        entropy2 = (1-phi)*np.log((1-phi), where=(1-phi) > 0, out=out2)
        interaction_energy = -1/2*chi*np.square(phi)
        f = entropy1+entropy2+interaction_energy
        return f*Nv

    def mu(self,phi):
        Na=self.Na
        chi=self.chi
        Nv=self.Nv
        phi = np.array(phi, dtype=float)
        out1=np.array(-np.ones_like(phi)*1e10,dtype=float)
        dS1dphi=1/Na*np.log(phi, where=phi > 0, out=out1)+1/Na
        out2 = np.array(-np.ones_like(phi)*1e10, dtype=float)
        dS2dphi = -np.log((1-phi), where=(1-phi) > 0, out=out2)-1
        mu = dS1dphi+dS2dphi-chi*phi
        return mu*Nv

    def osmotic_pressure(self,phi):
        return self.mu(phi)*phi-self.freeEnergy(phi)

    def binodal(self,phi):#binodal points correspond to chemical and osmotic equilibrium (tangent construction)
        phi1,phi2=phi
        eq1 = self.mu(phi1)-self.mu(phi2)
        eq2 = self.osmotic_pressure(phi1) - self.osmotic_pressure(phi2)
        return [eq1,eq2]

    def Spinodal(self,phi):#spinodal point correspond to the second derivative of the free energy to be zeros
        dmudphi=1/(self.Na*phi)+1/(1-phi)-self.chi
        return dmudphi

