import numpy as np
import math,time,json
import torch
from torch import nn
ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


class Sampler(nn.Module):

    def __init__(self,device=device):
        self.device=device
        super(Sampler, self).__init__()

    def save(self,fn):
        torch.save(self,fn)

    def logprobability(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        raise NotImplementedError

    @torch.jit.export
    def sample(self,B,L):
        # type: (int,int) -> Tensor
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
            logprobs - [B] matrix of logscale probabilities (float Tensor)
        """
        raise NotImplementedError

    @torch.jit.export
    def off_diag_labels(self,sample,nloops=1):
        # type: (Tensor,int) -> Tensor
        """
        Inputs:
            samples  - [B,L,1] matrix of zeros and ones for ground/excited states
        
        Returns:
            probs - size [B,L] tensor of probabilities of the excitation-flipped states
        """
        D=nloops
        B,L,_=sample.shape
        sflip = torch.zeros([B,L,L,1],device=self.device)
        #collect all of the flipped states into one array
        for j in range(L):
            #get all of the states with one spin flipped
            sflip[:,j] = sample*1.0
            sflip[:,j,j] = 1-sflip[:,j,j]
        #compute all of their logscale probabilities
        with torch.no_grad():
            probs=torch.zeros([B*L],device=self.device)
            tmp=sflip.view([B*L,L,1])
            for k in range(D):
                probs[k*B*L//D:(k+1)*B*L//D] = self.logprobability(tmp[k*B*L//D:(k+1)*B*L//D])

        return probs.reshape([B,L])

    @torch.jit.export
    def off_diag_labels_summed(self,sample,nloops=1):
        # type: (Tensor,int) -> Tuple[Tensor,Tensor]
        """
        Inputs:
            samples  - [B,L,1] matrix of zeros and ones for ground/excited states
        
        Returns:
            logsqrtp - size B vector of average (log p)/2 values used for numerical stability 
                       when calculating sum_s'(sqrt[p(s')/p(s)]) 
            sumsqrtp - size B vector of exp(-logsqrtp)*sum(sqrt[p(s')]).
        """
        probs = self.off_diag_labels(sample,nloops)
        #get the average of our logprobabilities and divide by 2
        logsqrtp=probs.mean(dim=1)/2
        #compute the sum with a constant multiplied to keep the sum close to 1
        sumsqrtp = torch.exp(probs/2-logsqrtp.unsqueeze(1)).sum(dim=1)
        return sumsqrtp,logsqrtp
    
# Functions for making Patches & doing probability traces
class Patch2D(nn.Module):
    def __init__(self,nx,ny,Lx,Ly,device=device):
        super().__init__()
        self.nx=nx
        self.ny=ny
        self.Ly=Ly
        self.Lx=Lx
        
        #construct an index tensor for the reverse operation
        indices = torch.arange(Lx*Ly,device=device).unsqueeze(0)
        self.mixed = self.forward(indices).reshape([Lx*Ly])
        #inverse
        self.mixed=torch.argsort(self.mixed)
        
    def forward(self,x):
        # type: (Tensor) -> Tensor
        nx,ny,Lx,Ly=self.nx,self.ny,self.Lx,self.Ly
        """Unflatten a tensor back to 2D, break it into nxn chunks, then flatten the sequence and the chunks
            Input:
                Tensor of shape [B,L]
            Output:
                Tensor of shape [B,L//n^2,n^2]
        """
        #make the input 2D then break it into 2x2 chunks 
        #afterwards reshape the 2x2 chunks to vectors of size 4 and flatten the 2d bit
        return x.view([x.shape[0],Lx,Ly]).unfold(-2,nx,nx).unfold(-2,ny,ny).reshape([x.shape[0],int(Lx*Ly//(nx*ny)),nx*ny])

    def reverse(self,x):
        # type: (Tensor) -> Tensor
        """Inverse function of forward
            Input:
                Tensor of shape [B,L//n^2,n^2]
            Output:
                Tensor of shape [B,L]
        """
        Ly,Lx=self.Ly,self.Lx 
        # Reversing is done with an index tensor because torch doesn't have an inverse method for unfold
        return x.reshape([x.shape[0],Ly*Lx])[:,self.mixed]
    
class Patch1D(nn.Module):
    def __init__(self,n,L):
        super().__init__()
        self.n=n
        self.L = L
    
    def forward(self,x):
        # type: (Tensor) -> Tensor
        """Break a tensor into chunks, essentially a wrapper of reshape
            Input:
                Tensor of shape [B,L]
            Output:
                Tensor of shape [B,L/n,n]
        """
        #make the input 2D then break it into 2x2 chunks 
        #afterwards reshape the 2x2 chunks to vectors of size 4 and flatten the 2d bit
        return x.reshape([x.shape[0],self.L//self.n,self.n])

    def reverse(self,x):
        # type: (Tensor) -> Tensor
        """Inverse function of forward
            Input:
                Tensor of shape [B,L/n,n]
            Output:
                Tensor of shape [B,L]
        """
        # original sequence order can be retrieved by chunking twice more
        #in the x-direction you should have chunks of size 2, but in y it should
        #be chunks of size Ly//2
        return x.reshape([x.shape[0],self.L])

@torch.jit.script
def genpatch2onehot(patch,p):
    # type: (Tensor,int) -> Tensor
    """ Turn a sequence of size p patches into a onehot vector
    Inputs:
        patch - Tensor of shape [?,p]
        p (int) - the patch size
    
    """
    #moving the last dimension to the front
    patch=patch.unsqueeze(0).transpose(-1,0).squeeze(-1).to(torch.int64)
    out=torch.zeros(patch.shape[1:],device=patch.device)
    for i in range(p):
        out+=patch[i]<<i
    return nn.functional.one_hot(out.to(torch.int64), num_classes=1<<p)
