from tvmc.util import Options,OptionManager
from tvmc.models.BaseModel import *

class PRNN(Sampler):
    """
    Patched Recurrent Neural Network Implementation.
    
    The network is patched as the sequence is broken into patches of size p, then entire patches are sampled at once.
    This means the sequence length is reduced from L to L/p but the output layer must now use a softmax over 2**p possible
    patches. Setting p above 5 is not recommended.
    
    Note for _2D = True, p actually becomes a pxp patch so the sequence is reduced to L/p^2 and it's a softmax over
    2^(p^2) patches so p=2 is about the only patch size which makes sense
    
    """
    
    INFO = """RNN based sampler where the input sequence is broken up into 'patches' and the output is a sequence of conditional probabilities of all possible patches at position i given the previous 0 to i-1 patches. Each patch is used to update the RNN hidden state, which (after two Fully Connected layers) is used to get the probability labels.
    
    RNN Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- RNN hidden size.
    
        patch      (str)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/prod(patch).
                                Example values: 2x2, 2x3, 2, 4
        
        rnntype    (string)  -- Which type of RNN cell to use. Only ELMAN and GRU are valid options at the moment.
    """
    
    DEFAULTS=Options(L=16,patch=1,rnntype="GRU",Nh=256)
    TYPES={"GRU":nn.GRU,"ELMAN":nn.RNN,"LSTM":nn.LSTM}
    def __init__(self,L,patch,rnntype,Nh,device=device, **kwargs):
        
        super(PRNN, self).__init__(device=device)
        if type(patch)==str and len(patch.split("x"))==2:
            px,py = [int(a) for a in patch.split("x")]
            Lx,Ly=[int(L**0.5)]*2 if type(L) is int else [int(a) for a in L.split("x")]
            self.patch=Patch2D(px,py,Lx,Ly)
            self.L = int(Lx*Ly//(px*py))
            self.p=px*py
        else:
            p=int(patch)
            self.patch=Patch1D(p,L)
            self.L = int(L//p)
            self.p = p
        
        assert rnntype!="LSTM"
        #rnn takes input shape [B,L,1]
        self.rnn = PRNN.TYPES[rnntype](input_size=self.p,hidden_size=Nh,batch_first=True)
         
        self.lin = nn.Sequential(
                nn.Linear(Nh,Nh),
                nn.ReLU(),
                nn.Linear(Nh,1<<self.p),
                nn.Softmax(dim=-1)
            )
        self.Nh=Nh
        self.rnntype=rnntype
        
        #create a tensor of all possible patches
        self.options=torch.zeros([1<<self.p,self.p],device=self.device)
        tmp=torch.arange(1<<self.p,device=self.device)
        for i in range(self.p):
            self.options[:,i]=(tmp>>i)%2
            
        self.to(device)
    
    @torch.jit.export
    def logprobability(self,input,h0=None):
        # type: (Tensor,Optional[Tensor]) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
                
        #shape is modified to [B,L//4,4]
        input = self.patch(input.squeeze(-1))
        data=torch.zeros(input.shape,device=self.device)
        #batch first
        data[:,1:]=input[:,:-1]
        # [B,L//4,Nh] -> [B,L//4,16]
        
        if h0 is None:
            h0=torch.zeros([1,input.shape[0],self.Nh],device=self.device)
        out,h=self.rnn(data,h0)
        output = self.lin(out)
        
        #real is going to be a onehot with the index of the appropriate patch set to 1
        #shape will be [B,L//4,16]
        real=genpatch2onehot(input,self.p)
        
        #[B,L//4,16] -> [B,L//4]
        total = torch.sum(real*output,dim=-1)
        #[B,L//4] -> [B]
        logp=torch.sum(torch.log(total),dim=1)
        return logp
    
    @torch.jit.export
    def sample(self,B,L,h0=None):
        # type: (int,int,Optional[Tensor]) -> Tuple[Tensor,Tensor]
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
        """
        #length is divided by four due to patching
        L=L//self.p
        
        if h0 is None:  
            h=torch.zeros([1,B,self.Nh],device=self.device)
        else:
            h=h0
        #Sample set will have shape [B,L,p]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        input = torch.zeros([B,L+1,self.p],device=self.device)
        sample = torch.zeros([B,self.p],device=self.device)
        logp = torch.zeros([B],device=self.device)
        
        for idx in range(1,L+1):
            #out should be batch first [B,L,Nh]
            out,h=self.rnn(sample.unsqueeze(1),h)
            #check out the probability of all 1<<p vectors
            probs=self.lin(out[:,0,:]).view([B,1<<self.p])
            #sample from the probability distribution
            indices = torch.multinomial(probs,1,False).squeeze(1)
            #extract samples
            sample = self.options[indices]
            
            onehot = nn.functional.one_hot(indices, num_classes=1<<self.p)
            
            logp+= torch.log(torch.sum(onehot*probs,dim=-1))
            
            #set input to the sample that was actually chosen
            input[:,idx] = sample
        #remove the leading zero in the input    
        #sample is repeated 16 times at 3rd index so we just take the first one 
        return self.patch.reverse(input[:,1:]).unsqueeze(-1),logp
        
    @torch.jit.export
    def off_diag_labels(self,sample,nloops=1):
        # type: (Tensor,int) -> Tensor
        """label all of the flipped states  - set D as high as possible without it slowing down runtime
        Parameters:
            sample - [B,L,1] matrix of zeros and ones for ground/excited states
            B,L (int) - batch size and sequence length
            D (int) - Number of partitions sequence-wise. We must have L%D==0 (D divides L)
            
        Outputs:
            
            sample - same as input
            probs - [B,L] matrix of probabilities of states with the jth excitation flipped
        """
        D=nloops
        B,L,_=sample.shape
        sample0=sample
        #sample is batch first at the moment
        sample = self.patch(sample.squeeze(-1))
        
        sflip = torch.zeros([B,L,L//self.p,self.p],device=self.device)
        #collect all of the flipped states into one array
        for j in range(L//self.p):
            #have to change the order of in which states are flipped for the cache to be useful
            for j2 in range(self.p):
                sflip[:,j*self.p+j2] = sample*1.0
                sflip[:,j*self.p+j2,j,j2] = 1-sflip[:,j*self.p+j2,j,j2]
                 
        #compute all of their logscale probabilities
        data=torch.zeros(sample.shape,device=self.device)

        data[:,1:]=sample[:,:-1]

        #add positional encoding and make the cache

        h=torch.zeros([1,B,self.Nh],device=self.device)

        out,_=self.rnn(data,h)

        #cache for the rnn is the output in this sense
        #shape [B,L//4,Nh]
        cache=out
        probs=torch.zeros([B,L],device=self.device)
        #expand cache to group L//D flipped states
        cache=cache.unsqueeze(1)

        #the cache has to be shaped such that the batch parts line up
        cache=cache.repeat(1,L//D,1,1).reshape(B*L//D,L//self.p,cache.shape[-1])

        pred0 = self.lin(out)
        #shape will be [B,L//4,16]
        real=genpatch2onehot(sample,self.p)
        #[B,L//4,16] -> [B,L//4]
        total0 = torch.sum(real*pred0,dim=-1)

        for k in range(D):

            N = k*L//D
            #next couple of steps are crucial          
            #get the samples from N to N+L//D
            #Note: samples are the same as the original up to the Nth spin
            real = sflip[:,N:(k+1)*L//D]
            #flatten it out and set to sequence first
            tmp = real.reshape([B*L//D,L//self.p,self.p])
            #set up next state predction
            fsample=torch.zeros(tmp.shape,device=self.device)
            fsample[:,1:]=tmp[:,:-1]
            #grab your rnn output
            if k==0:
                out,_=self.rnn(fsample,cache[:,0].unsqueeze(0)*0.0)
            else:
                out,_=self.rnn(fsample[:,N//self.p:],cache[:,N//self.p-1].unsqueeze(0)*1.0)
            # grab output for the new part
            output = self.lin(out)
            # reshape output separating batch from spin flip grouping
            pred = output.view([B,L//D,(L-N)//self.p,1<<self.p])
            real = genpatch2onehot(real[:,:,N//self.p:],self.p)
            total = torch.sum(real*pred,dim=-1)
            #sum across the sequence for probabilities
            logp=torch.sum(torch.log(total),dim=-1)
            logp+=torch.sum(torch.log(total0[:,:N//self.p]),dim=-1).unsqueeze(-1)
            probs[:,N:(k+1)*L//D]=logp

        return probs

OptionManager.register("rnn",PRNN.DEFAULTS)


