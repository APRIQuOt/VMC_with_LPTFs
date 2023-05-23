from tvmc.util import Options,OptionManager
from tvmc.hamiltonians.hamiltonian import *

class Rydberg(Hamiltonian):
    
    DEFAULTS = Options(Lx=4,Ly=4,V=7.0,Omega=1.0,delta=1.0)
    def __init__(self,Lx,Ly,V,Omega,delta,device=device,**kwargs):
        self.Lx       = Lx              # Size along x
        self.Ly       = Ly              # Size along y
        self.V        = V               # Van der Waals potential
        self.delta    = delta           # Detuning
        # off diagonal part is -0.5*Omega
        super(Rydberg,self).__init__(Lx*Ly,-0.5*Omega,device)

    @staticmethod
    def Vij(Ly,Lx,V,matrix):
    #matrix will be size [Lx*Ly,Lx*Ly]
      for i in range(Ly):
        for j in range(Lx):
            #flatten two indices into one
            idx = Ly*j+i
            # only fill in the upper diagonal
            for k in range(idx+1,Lx*Ly):
                #expand one index into two
                i2 = k%Ly
                j2=k//Ly
                div = ((i2-i)**2+(j2-j)**2)**3
                #if div<=R:
                matrix[idx][k]=V/div

    def buildlattice(self):
        Lx,Ly=self.Lx,self.Ly
        
        #diagonal hamiltonian portion can be written as a matrix multiplication then a dot product        
        mat=np.zeros([self.L,self.L])
        Rydberg.Vij(Lx,Ly,self.V,mat)
        
        with torch.no_grad():
            self.Vij.weight[:,:]=torch.Tensor(mat)
            self.Vij.bias.fill_(-self.delta)

    def ground(self):
        return Rydberg.E[self.Lx*self.Ly]

OptionManager.register("rydberg",Rydberg.DEFAULTS)
