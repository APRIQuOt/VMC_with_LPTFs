import numpy as np
import torch
from torch import nn
ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Hamiltonian():
    def __init__(self,L,offDiag,device=device):
        self.offDiag  = offDiag           # Off-diagonal interaction
        self.L        = L               # Number of spins
        self.device   = device
        self.Vij      = self.Vij=nn.Linear(self.L,self.L).to(device)
        self.buildlattice()

    def buildlattice():
        """Creates the matrix representation of the on-diagonal part of the hamiltonian
            - This should fill Vij with values"""
        raise NotImplementedError

#    def localenergy(self,samples,logp,logppj):
#        """
#        Takes in s, ln[p(s)] and ln[p(s')] (for all s'), then computes Hloc(s) for N samples s.
#        
#        Inputs:
#            samples - [B,L,1] matrix of zeros and ones for ground/excited states
#            logp - size B vector of logscale probabilities ln[p(s)]
#            logppj - [B,L] matrix of logscale probabilities ln[p(s')] where s'[i][j] had one state flipped at position j
#                    relative to s[i]
#        Returns:
#            size B vector of energies Hloc(s)
#        
#        """
#        # Going to calculate Eloc for each sample in a separate spot
#        # so eloc will have shape [B]
#        # recall samples has shape [B,L,1]
#        B=samples.shape[0]
#        eloc = torch.zeros(B,device=self.device)
#        # Chemical potential
#        with torch.no_grad():
#            tmp=self.Vij(samples.squeeze(2))
#            eloc += torch.sum(tmp*samples.squeeze(2),axis=1)
#        # Off-diagonal part
#        #logppj is shape [B,L]
#        #logppj[:,j] has one state flipped at position j
#        for j in range(self.L):
#            #make sure torch.exp is a thing
#            eloc += self.offDiag * torch.exp((logppj[:,j]-logp)/2)
#
#        return eloc

    def localenergyALT(self,samples,logp,sumsqrtp,logsqrtp):
        """
        Takes in s, ln[p(s)] and exp(-logsqrtp)*sum(sqrt[p(s')]), then computes Hloc(s) for N samples s.
        
        Inputs:
            samples  - [B,L,1] matrix of zeros and ones for ground/excited states
            logp     - size B vector of logscale probabilities ln[p(s)]
            logsqrtp - size B vector of average (log p)/2 values used for numerical stability 
                       when calculating sum_s'(sqrt[p(s')/p(s)]) 
            sumsqrtp - size B vector of exp(-logsqrtp)*sum(sqrt[p(s')]).
        Returns:
            size B vector of energies Hloc(s)
        
        """
        # Going to calculate Eloc for each sample in a separate spot
        # so eloc will have shape [B]
        # recall samples has shape [B,L,1]
        B=samples.shape[0]
        eloc = torch.zeros(B,device=self.device)
        # Chemical potential
        with torch.no_grad():
            tmp=self.Vij(samples.squeeze(2))
            eloc += torch.sum(tmp*samples.squeeze(2),axis=1)
        # Off-diagonal part
        
        #in this function the entire sum is precomputed and it was premultiplied by exp(-logsqrtp) for stability
        eloc += self.offDiag *sumsqrtp* torch.exp(logsqrtp-logp/2)

        return eloc
    
    def magnetizations(self, samples):
        B = samples.shape[0]
        L = samples.shape[1]
        mag = torch.zeros(B, device=self.device)
        abs_mag = torch.zeros(B, device=self.device)
        sq_mag = torch.zeros(B, device=self.device)
        stag_mag = torch.zeros(B, device=self.device)

        with torch.no_grad():
            samples_pm = 2 * samples - 1
            mag += torch.sum(samples_pm.squeeze(2), axis=1)
            abs_mag += torch.abs(torch.sum(samples_pm.squeeze(2), axis=1))
            sq_mag += torch.abs(torch.sum(samples_pm.squeeze(2), axis=1))**2
            
            samples_reshape = torch.reshape(samples.squeeze(2), (B, int(np.sqrt(L)), int(np.sqrt(L))))
            for i in range(int(np.sqrt(L))):
                for j in range(int(np.sqrt(L))):
                    stag_mag += (-1)**(i+j) * (samples_reshape[:,i,j] - 0.5)

        return mag, abs_mag, sq_mag, stag_mag / L

    def ground(self):
        """Returns the ground state energy E/L"""
        raise NotImplementedError
