from tvmc.util import Options,OptionManager
from tvmc.models.PTF import *
from tvmc.models.RNN import PRNN


class LPTF(Sampler):
    
    """
    Sampler class which uses a transformer for long term information and a smaller subsampler for short term information
    This can either be in the form of an RNN or a transformer (likely patched).
    
    The sequence is broken into 2D patches, each patch is expanded to a tensor of size Nh (be repeating it),\
    then a positional encoding is added. You then apply masked self-attention to the patches num_layers times, with the final
    outputs fed in as the initial hidden state of an rnn.
    
    Going with 4x4 patches, you can use these patches as a sequence to get a factorized probability of the entire
    4x4 patch by feeding the 2x2 patches in one at a time and outputting a size 16 tensor 
    (probability of all possible next 2x2 patches) for each patch. The output is obtained by applying two FC layers to
    the hidden state of the rnn.
    
    
    Here is an example of how everything comes together
    
    Say you have a 16x16 input and Nh=128, this input is broken into 16 4x4 patches which are repeated 8 times and
    given a positional encoding. Masked self attention is done between the 16 patches (size Nh) for N layers, then
    16 RNNs are given the outputs in parallel as the hidden state. Now the original input is broken into 16 sets of 4 2x2
    patches. These length 4 sequences are given to the rnns (16 in parallel all sharing the same weights) and the outputs
    are then grouped together such that you end up with a length 64 sequence of vectors of size 16. this gives your probability.
    You can easily calculate it by taking the output (of 16) corresponding to each 2x2 patch and multiplying all 64 of them
    together (or adding them in logscale).
    
    """
    INFO = """Transformer based sampler where the input sequence is broken up into large 'patches' and the output is a sequence of conditional probabilities of all possible patches at position i given the previous 0 to i-1 patches. Each patch is projected into a token with an added positional encoding. The sequence of encoded patches is used as transformer input. This specific model is used for very large patches where doing a softmax over all possible patches is not feasable thus a subsampler must be used to factorize these probabilities.
    
    
    LPTF Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
                                Note: When using an RNN subsampler this Nh MUST match the rnn's Nh.
    
        patch      (int)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/patch.
            
        dropout    (float)   -- The amount of dropout to use in the transformer layers.
        
        num_layers (int)     -- The number of transformer layers to use.
        
        nhead     (int)     -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh.
        
        subsampler (Sampler) -- The inner model to use for probability factorization. This is set implicitly.
                                by including --rnn or --ptf arguments.
    
    """
    DEFAULTS=Options(L=64,patch=1,Nh=128,dropout=0.0,num_layers=2,nhead=8,full_seq=False)
    def __init__(self,subsampler,L,patch,Nh,dropout,num_layers,nhead,full_seq,device=device, **kwargs):
        super(Sampler, self).__init__()
        
        if type(patch)==str and len(patch.split("x"))==2:
            
            #patch and system sizes
            px,py = [int(a) for a in patch.split("x")]
            Lx,Ly=[int(L**0.5)]*2 if type(L) is int else [int(a) for a in L.split("x")]
            #token size and positional encoder
            self.pe = PE2D(Nh, Lx//px,Ly//py,device)
            #patching, sequence length and total patch size
            self.patch=Patch2D(px,py,Lx,Ly)
            self.L = int(L//(px*py))
            self.p=px*py
        else:
            p=int(patch)
            self.pe = PE1D(Nh,L//p,device)
            self.patch=Patch1D(p,L)
            self.L = int(L//p)
            self.p = p
            
        self.tokenize=nn.Sequential(
                nn.Linear(self.p,Nh),
                nn.Tanh()
        )
            
        self.allh=full_seq
            
        self.device=device
        #Encoder only transformer
        self.transformer = FastMaskedTransformerEncoder(Nh=Nh,dropout=dropout,num_layers=num_layers,nhead=nhead)       
        
        # Sampler class object which has both sample and logprobability functions
        self.subsampler = subsampler
        
        self.set_mask(self.L)
        
        self.to(device)
    
    def set_mask(self, L):
        # type: (int)
        """Initialize the self-attention mask"""
        self.L=L
        self.transformer.set_mask(L)
        self.pe.L=L

    @torch.jit.export
    def logprobability(self,input):
        # type: (Tensor) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        
        if input.shape[1]//self.p!=self.L:
            self.set_mask(input.shape[1]//self.p)
        #shape should be sequence first [L,B,Nh]
        
        #shape is modified to [L//p,B,p]
        input = self.patch(input.squeeze(-1)).transpose(1,0)
        
        data=torch.zeros(input.shape,device=self.device)
        #The first input should be zeros and the last patch is not used as input
        data[1:]=input[:-1]
        
        #[L//p,B,p] -> [L//p,B,Nh]
        encoded=self.pe(self.tokenize(data))
        #shape is preserved
        output = self.transformer(encoded)
        
        Lp,B,Nh=output.shape
        if self.allh:
            h0 = output
        else:
            # [L//p,B,Nh] -> [1,L//p*B,Nh]
            h0 = output.view([1,Lp*B,Nh])
        flattened_input = input.reshape([Lp*B,self.p])
        # [L//p*B,p],[1,L//p*B,Nh] -> [L//p,B]
        logsubsample = self.subsampler.logprobability(flattened_input,h0).view([Lp,B])
        
        #[L//p,B] -> [B]
        logp=torch.sum(logsubsample,dim=0)
        return logp
    
    @torch.jit.export
    def sample(self,B,L,cache=None):
        # type: (int,int,Optional[Tensor]) -> Tuple[Tensor,Tensor]
        """ Generates a set states
        Inputs:
            B (int)            - The number of states to generate in parallel
            L (int)            - The length of generated vectors
        Returns:
            samples - [B,L,1] matrix of zeros and ones for ground/excited states
        """
        #sequence length is divided by patch size due to patching
        L=L//self.p
        
        #Sample set will have shape [L/p,B,p]
        #need one extra zero batch at the start for first pred hence input is [L/p+1,B,1] 
        input = torch.zeros([L+1,B,self.p],device=self.device)

        logp = torch.zeros([B],device=self.device)
        
        for idx in range(1,L+1):
            
            #[l,B,p] -> [l,B,Nh]            multiply by 1 to copy the tensor
            encoded_input = self.pe(self.tokenize(input[:idx,:,:]*1))
                        
            #Get transformer output (shape [l,B,Nh])
            output,cache = self.transformer.next_with_cache(encoded_input,cache)
            #get state and probability by sampling from the subsample (pass along the last elem reshaped to [1,B,Nh])
            if self.allh:
                sample,logsubsample = self.subsampler.sample(B,self.p,output)
            else:
                sample,logsubsample = self.subsampler.sample(B,self.p,output[-1].view([1,B,output.shape[-1]]))
            #Add your logscale conditional probability to the sum
            logp+=logsubsample
            #set input to the sample that was actually chosen
            input[idx] = sample.squeeze(-1)
            
        #remove the leading zero in the input    
        input=input[1:]
        #Unpatch the samples
        return self.patch.reverse(input.transpose(1,0)).unsqueeze(-1),logp
    
    
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
            
        #switch sample into sequence-first
        sample = sample.transpose(1,0)
            
        #compute all of the logscale probabilities of the original sample
        data=torch.zeros(sample.shape,device=self.device)
        data[1:]=sample[:-1]

        #[L//p,B,p] -> [L//p,B,Nh]
        encoded=self.pe(self.tokenize(data))

        #add positional encoding and make the cache
        out,cache=self.transformer.make_cache(encoded)
        probs=torch.zeros([B,L],device=self.device)
        #expand cache to group L//D flipped states
        cache=cache.unsqueeze(2)
        
        #the cache has to be repeated L//D times along the correct axis (otherwise there is a mismatch)
        cache=cache.repeat(1,1,L//D,1,1).transpose(2,3).reshape(cache.shape[0],L//self.p,B*L//D,cache.shape[-1])

        Lp,B,Nh=out.shape
        
        if self.allh:
            h0 = out
        else:
            # [L//p,B,Nh] -> [1,L//p*B,Nh]
            h0 = out.view([1,Lp*B,Nh])
        #flatten the batch & sequence dimensions into the batch dimension
        flattened_input = sample.reshape([Lp*B,self.p])
        # [L//p*B,p],[1,L//p*B,Nh] -> [L//p,B]
        logsubsample0 = self.subsampler.logprobability(flattened_input,h0).view([Lp,B])

        for k in range(D):

            N = k*L//D
            #next couple of steps are crucial          
            #get the samples from N to N+L//D
            #Note: samples are the same as the original up to the Nth spin
            real = sflip[:,N:(k+1)*L//D]
            #flatten it out and set to sequence first
            tmp = real.reshape([B*L//D,L//self.p,self.p]).transpose(1,0)
            #set up next state predction
            fsample=torch.zeros(tmp.shape,device=self.device)
            fsample[1:]=tmp[:-1]
            # add positional encoding
            tgt=self.pe(self.tokenize(fsample))
            #grab your transformer output
            out,_=self.transformer.next_with_cache(tgt,cache[:,:N//self.p],N//self.p)

            output = out[N//self.p:]

            #[(L-N)/p,B*L/D,Nh]
            Lp2,B2,Nh=output.shape

            if self.allh:
                h0 = out
            else:
                # [(L-N)/p,B*L/D,Nh] -> [1,((L-N)/p)*(B*L/D),Nh]
                h0 = output.view([1,Lp2*B2,Nh])
            #flatten the batch & sequence dimensions into the batch dimension
            flattened_input = tmp[N//self.p:].reshape([Lp2*B2,self.p])
            
            #get the subsampler output and unflatten it
            # [?] -> [(L-N)/p,B*L//D]
            logsubsample = self.subsampler.logprobability(flattened_input,h0).view([Lp2,B2])

            #[(L-N)/p,B*L//D] -> [B,L/D]

            #sum over (L-N)/p
            logp=torch.sum(logsubsample,dim=0).view([B,L//D])

            #sum over N/p
            logp+=torch.sum(logsubsample0[:N//self.p],dim=0).unsqueeze(-1)

            probs[:,N:(k+1)*L//D]=logp
                
        return probs
    
OptionManager.register("lptf",LPTF.DEFAULTS) 

