#Numerical Solver for Flory Huggins Theory in two- / three- component systems capable of separting into two phases
#Integrating two methods online with significant modifications
#Source 1: Dr. Ryan McGorty at University of San Diego
#https://github.com/rmcgorty/PhaseSeparation
#Source 2: https://github.com/samueljmcameron/floryhugginsternary
#Author: Zhengqing Zhou, Duke University
#Most Recent Update: Mar 26th, 2024

import numpy as np

class FloryHuggins():
    def __init__(self, chi_12=1, chi_11=1, chi_22=1, Na=10, Nb=10, Nc=1):
        """
        Initialise all attributes.

        Parameters
        ----------
        chi_12 : float
            The value of the chi_12 interaction parameter.
        chi_11 : float, optional
            The value of the chi_11 interaction parameter.
        chi_22 : float, optional
            The value of the chi_22 interaction parameter.
        Na, Nb, Nc: int
            lengths of solute 1 and 2, and solvent
        """

        self.chi_12 = chi_12
        self.chi_11 = chi_11
        self.chi_22 = chi_22
        self.Na = Na
        self.Nb = Nb
        self.Nc = Nc

        return

    def freeEnergy(self,phi_1,phi_2):#free energy of a mixture with volume fraction of phi1 and phi2
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        entropy1 = (phi_1/Na)*np.log(phi_1) if phi_1>0 else 0 # entropy of polymer 1
        entropy2 = (phi_2/Nb)*np.log(phi_2) if phi_2>0 else 0 # entropy of polymer 2
        entropy3 = ((1-phi_1-phi_2)/Nc)*np.log(1-phi_1-phi_2) if 1-phi_1-phi_2>0 else 0 # entropy of solvent
        interaction_energy = -(chi_11*np.square(phi_1)+chi_22*np.square(phi_2)+2*chi_12*phi_1*phi_2)/2
        f = entropy1+entropy2+entropy3+interaction_energy
        return f

    def mu1(self,phi_1,phi_2):#chemical potential of phi_1
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        dS1dphi1 = (1/Na)*np.log(phi_1)+(1/Na) if phi_1>0 else 0
        dS3dphi1 = -(1/Nc)*np.log(1-phi_1-phi_2)-(1/Nc) if 1-phi_1-phi_2>0 else 0
        mu1 = dS1dphi1+dS3dphi1-chi_11*phi_1-chi_12*phi_2
        return mu1

    def mu2(self,phi_1,phi_2):#chemical potential of phi_2
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        dS2dphi2 = (1/Nb)*np.log(phi_2)+(1/Nb) if phi_2>0 else 0
        dS3dphi2 = -(1/Nc)*np.log(1-phi_1-phi_2)-(1/Nc) if 1-phi_1-phi_2>0 else 0
        mu2 = dS2dphi2+dS3dphi2-chi_22*phi_2-chi_12*phi_1
        return mu2

    def osmotic_pressure(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        solvent_osmo=(1/Nc)*np.log(1-phi_1-phi_2) if 1-phi_1-phi_2>0 else 0
        b = phi_1*(1/Nc-1/Na)+phi_2*(1/Nc-1/Nb)+(chi_11*(phi_1**2)+chi_22*(phi_2**2)+2*chi_12*phi_1*phi_2)/2+solvent_osmo
        return b

    # 2nd order derivatives
    def dmu1_phi1(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out1=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad1 = np.divide(1, Na * phi_1, where=phi_1 > 0, out=out1)
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = np.divide(1, Nc * (1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)

        grad = grad1+grad2-chi_11
        return grad

    def dmu1_phi2(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out1=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad1 = np.divide(1, Nc * (1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out1)
        grad = grad1-chi_12
        return grad

    def dmu2_phi2(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out1=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad1 = np.divide(1, Nb * phi_2, where=phi_2 > 0, out=out1)
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = np.divide(1, Nc * (1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)

        grad = grad1+grad2-chi_22
        return grad

    def db_phi1(self,phi_1,phi_2): # derivative of osmotic pressure with respect to phi1
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out1=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad1 = np.divide(1, Nc * (1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out1)
        grad = (1/Nc - 1/Na)+chi_11*phi_1+chi_12*phi_2-grad1
        return grad

    def db_phi2(self,phi_1,phi_2): # derivative of osmotic pressure with respect to phi1
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out1=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad1 = np.divide(1, Nc * (1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out1)
        grad = (1/Nc - 1/Nb)+chi_22*phi_2+chi_12*phi_1-grad1
        return grad

    def Hessian(self,phi_1,phi_2):# Hessian matrix of free energy
        H11=self.dmu1_phi1(phi_1,phi_2)
        H12=self.dmu1_phi2(phi_1,phi_2)
        H21=self.dmu1_phi2(phi_1,phi_2)
        H22=self.dmu2_phi2(phi_1,phi_2)

        H = np.array([[H11,H12],
                      [H21,H22]])

        return H

    def Spinodal(self,phi_1,phi_2): # the spinnodal point correspond to zero determinant of the Hessian
        return np.linalg.det(self.Hessian(phi_1,phi_2))

    def dh11_phi1(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out1=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad1 = -np.divide(1, Na * np.square(phi_1), where=phi_1 > 0, out=out1)
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = +np.divide(1, Nc * np.square(1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)
        grad = grad1+grad2
        return grad

    def dh12_phi1(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = +np.divide(1, Nc * np.square(1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)
        grad = grad2
        return grad

    def dh22_phi1(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = +np.divide(1, Nc * np.square(1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)
        grad = grad2
        return grad

    def dh11_phi2(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = +np.divide(1, Nc * np.square(1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)
        grad = grad2
        return grad

    def dh12_phi2(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = +np.divide(1, Nc * np.square(1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)
        grad = grad2
        return grad

    def dh22_phi2(self,phi_1,phi_2):
        Na=self.Na
        Nb=self.Nb
        Nc=self.Nc
        chi_11=self.chi_11
        chi_12=self.chi_12
        chi_22=self.chi_22
        # Ensure phi_1 and phi_2 are numpy arrays for consistent behavior
        phi_1 = np.array(phi_1, dtype=float)
        phi_2 = np.array(phi_2, dtype=float)
        # Calculate grad1 and grad2 using np.where for element-wise comparison
        out1=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad1 = -np.divide(1, Nb * np.square(phi_2), where=phi_2 > 0, out=out1)
        out2=np.array(np.ones_like(phi_1)*1e10,dtype=float)
        grad2 = +np.divide(1, Nc * np.square(1 - phi_1 - phi_2), where=1 - phi_1 - phi_2 > 0, out=out2)
        grad = grad1+grad2
        return grad

    def ddet_dphi1(self,phi_1,phi_2):
        H = self.Hessian(phi_1,phi_2)
        eq = self.dh11_phi1(phi_1, phi_2) * H[1, 1] + self.dh22_phi1(phi_1, phi_2) * H[0, 0] \
        - 2 * H[0, 1] * self.dh12_phi1(phi_1, phi_2)
        return eq

    def ddet_dphi2(self,phi_1,phi_2):
        H = self.Hessian(phi_1,phi_2)
        eq = self.dh11_phi2(phi_1, phi_2) * H[1, 1] + self.dh22_phi2(phi_1, phi_2) * H[0, 0] \
        - 2 * H[0, 1] * self.dh12_phi2(phi_1, phi_2)
        return eq

    def critical_point(self,x):
        phi_1,phi_2=x
        H = self.Hessian(phi_1,phi_2)
        M = H*1
        M[1,0] = self.ddet_dphi1(phi_1,phi_2)
        M[1,1] = self.ddet_dphi2(phi_1,phi_2)
        return [np.linalg.det(H),np.linalg.det(M)]

class Ternary_2phase(FloryHuggins):
    """
    Child class of FloryHuggins.
    Generate Equations for chemical and osmotic pressure equilibrium between two phases

    Attributes
    ----------
    phi_1A : float
        Volume fraction of component 1, in phase A.
        Must satisfy 0 < phi_1A < 1.
    chi_12 : float
        The value of the chi_12 interaction parameter.
    chi_11 : float, optional
        The value of the chi_11 interaction parameter.
    chi_22 : float
        The value of the chi_22 interaction parameter.
    """

    def __init__(self, phi_1A, chi_12=1, chi_11=1, chi_22=1, Na=10, Nb=10, Nc=1,
                 largenumber=1e20, check_bounds=True):
        """
        Initialise all attributes.

        Parameters
        ----------
        chi_12 : float
            The value of the chi_12 interaction parameter.
        chi_11 : float, optional
            The value of the chi_11 interaction parameter.
        chi_22 : float, optional
            The value of the chi_22 interaction parameter.
        Na, Nb, Nc: int
            lengths of solute 1 and 2, and solvent
        func : callable, optional
            The function which maps phi_1 to phi_2 along
            a curve.

            '' func(x,*args) -> float or np.array ''

        dfunc : callable, optional
            The derivative of func.

            '' dfunc(x,*args) -> float or np.array ''

        args : tuple, optional
            Extra arguments which are to be passed to
            func and dfunc.
        """

        super(Ternary_2phase,self).__init__(chi_12,chi_11,chi_22,Na,Nb,Nc)
        self.phi_1A = phi_1A
        self.largenumber = largenumber
        self.check_bounds = check_bounds

        return

    def _bounds_checker(self, x_A, x_B):
        # return True if volume constraints are broken
        if (x_A<0).any() or (x_B<0).any() or np.sum(x_A)>1 or np.sum(x_B)>1:
            return True
        else:
            return False
        return

    def equilibrium(self, x):
        """
        a set of equations whose root corresponds to
        chemical and osmotic equilbrium between two phases

        Parameters
        ----------
        x : np.array of length 3
            Array with values of the volume fractions
            in the three phases, i.e.

            x = np.array([phi_2A,phi_1B,phi_2B])


        Returns
        -------
        gs : list of floats
            List of all the equations of chemical
            equilibrium, in the order g0_AB, g1_AB,
            g2_AB.

        """
        phi_1A = self.phi_1A
        phi_2A = x[0]
        phi_1B = x[1]
        phi_2B = x[2]

        if (self.check_bounds and
                self._bounds_checker(np.array([phi_1A,phi_2A]), np.array([phi_1B, phi_2B]))):
            outs = [self.largenumber, self.largenumber,
                    self.largenumber]
        else:
            out1 = self.mu1(phi_1A,phi_2A) - self.mu1(phi_1B,phi_2B)
            out2 = self.mu2(phi_1A,phi_2A) - self.mu2(phi_1B,phi_2B)
            out3 = self.osmotic_pressure(phi_1A,phi_2A) - self.osmotic_pressure(phi_1B,phi_2B)
            outs = [out1, out2, out3]
        return outs


    def eq_jacobian(self, x):
        """
        Computes the Jacobian matrix of equilibrium function to facilitate root solving.

        Parameters
        ----------
        x : np.array of length 3
            Array with values of the volume fractions
            in the three phases, i.e.

            x = np.array([phi_2A,phi_1B,phi_2B])

        Returns
        -------
        J : 3x3 np.array
            Jacobian of the rootfind_eqns method, with
            rows in the order 0_AB, 1_AB,
            2_AB and columns in the
            order phi_2A, phi_1B, phi_2B.

        """

        phi_1A = self.phi_1A
        phi_2A = x[0]
        phi_1B = x[1]
        phi_2B = x[2]


        J11 = self.dmu1_phi2(phi_1A,phi_2A)
        J12 = - self.dmu1_phi1(phi_1B,phi_2B)
        J13 = - self.dmu1_phi2(phi_1B,phi_2B)

        J21 = self.dmu2_phi2(phi_1A,phi_2A)
        J22 = - self.dmu1_phi2(phi_1B,phi_2B)
        J23 = - self.dmu2_phi2(phi_1B,phi_2B)

        J31 = self.db_phi2(phi_1A,phi_2A)
        J32 = - self.db_phi1(phi_1B,phi_2B)
        J33 = - self.db_phi2(phi_1B,phi_2B)

        return np.array([[J11, J12, J13],
                         [J21, J22, J23],
                         [J31, J32, J33]])
