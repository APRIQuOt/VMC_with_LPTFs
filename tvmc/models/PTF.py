from tvmc.util import Options,OptionManager
from tvmc.models.BaseModel import *

############################################Transformer Encoder Module############################################################

#Original Idea: https://github.com/alex-matton/causal-transformer-decoder

class FastMaskedTransformerEncoder(nn.Module):
    """
    Base class for a fast, masked transformer
    
    """
    def __init__(self,Nh=128,dropout=0.0,num_layers=2,nhead=8,device=device):
        super(FastMaskedTransformerEncoder, self).__init__()
        #Encoder only transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=Nh, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)      
        self.nl=num_layers
        self.Nh=Nh
        self.nhead=nhead
        self.device=device
    
    def set_mask(self, L):
        # type: (int)
        """
        Set the transformer mask for a sequence of length L
        Inputs: 
            L (int) - the desired sequence length
        """
        # take the log of a lower triangular matrix
        self.mask = torch.log(torch.tril(torch.ones([L,L],device=self.device)))        
        
    def forward(self, input):
        # type: (Tensor)->Tensor
        """Run the transformer on a sequence of length L
            Inputs:
                input -  Tensor of shape [L,B,Nh]
            Outputs:    
                Tensor of shape [L,B,Nh]
        """
        return self.transformer(input,self.mask)
    
    def next_with_cache(self,tgt,cache=None,idx=-1):
        # type: (Tensor,Optional[Tensor],int) -> Tuple[Tensor,Tensor]
        """Efficiently calculates the next output of a transformer given the input sequence and 
        cached intermediate layer encodings of the input sequence
        
        Inputs:
            tgt - Tensor of shape [L,B,Nh]
            cache - Tensor of shape ?
            idx - index from which to start
            
        Outputs:
            output - Tensor of shape [?,B,Nh]
            new_cache - Tensor of shape ?
        """
        output = tgt
        new_token_cache = []
        #go through each layer and apply self attention only to the last input
        for i,layer in enumerate(self.transformer.layers):
            
            tgt=output
            #have to merge the functions into one
            src = tgt[idx:, :, :]
            mask = None if idx==-1 else self.mask[idx:]

            # self attention part
            src2 = layer.self_attn(
                src,#only do attention with the last elem of the sequence
                tgt,
                tgt,
                attn_mask=mask,  
                key_padding_mask=None,
            )[0]
            #straight from torch transformer encoder code
            src = src + layer.dropout1(src2)
            src = layer.norm1(src)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
            src = src + layer.dropout2(src2)
            src = layer.norm2(src)
            
            output = src
            new_token_cache.append(output)
            if cache is not None:
                #layers after layer 1 need to use a cache of the previous layer's output on each input
                output = torch.cat([cache[i], output], dim=0)

        #update cache with new output
        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache
    
    def make_cache(self,tgt):
        # type: (Tensor) -> Tuple[Tensor,Tensor]
        """
        Equivalent to forward, but the intermediate outputs are also returned
        Inputs:
            tgt - Tensor of shape [L,B,Nh]
        Outputs:
            output - Tensor of shape [L,B,Nh]
            new_cache - Tensor of shape [?,L,B,Nh]
        """
        output = tgt
        new_token_cache = []
        #go through each layer and apply self attention only to the last input
        for i, layer in enumerate(self.transformer.layers):
            output = layer(output,src_mask=self.mask)
            new_token_cache.append(output)
        #create cache with tensor
        new_cache = torch.stack(new_token_cache, dim=0)
        return output, new_cache
    
############################################################Positional Encodings#######################################################
    
class PE2D(nn.Module):
    """Sequence-First 2D Positional Encoder"""
    def __init__(self, d_model, Lx,Ly,device,n_encode=None):
        
        super().__init__()
        assert (d_model%4==0)
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(Lx*Ly, d_model)
        
        if type(n_encode)==type(None):
            n_encode=3*d_model//4
        for pos in range(Lx*Ly):
            x=pos//Ly
            y=pos%Ly
            # Only going to fill 3/4 of the matrix so the
            # occupation values are preserved
            for i in range(0, n_encode, 4):
                
                #x direction encoding
                pe[pos, i] =                 math.sin(x / (10000 ** ((2 * i)/n_encode)))
                pe[pos, i + 1] =                 math.cos(x / (10000 ** ((2 * (i + 1))/n_encode)))
                #y direction encoding
                pe[pos, i+2] =                 math.sin(y / (10000 ** ((2 * i)/n_encode)))
                pe[pos, i + 3] =                 math.cos(y / (10000 ** ((2 * (i + 1))/n_encode)))
                
        self.pe = pe.unsqueeze(1).to(device)
        self.L=Lx*Ly
    
    def forward(self, x):
        """
        Adds a 2D positional encoding of size d_model to x
        Inputs:
            Tensor of shape [L,B,?]
        Outputs:
            Tensor of shape [L,B,d_model]
        """
        if self.d_model%x.shape[-1]!=0:
            return x.repeat(1,1,self.d_model//x.shape[-1]+1)[:,:,:self.d_model] + self.pe[:x.shape[0]]
        return x.repeat(1,1,self.d_model//x.shape[-1]) + self.pe[:x.shape[0]]
    
class PE1D(nn.Module):
    """Sequence-First 1D Positional Encoder"""
    def __init__(self, d_model, L,device,n_encode=None):
        super().__init__()
        assert (d_model%4==0)
        self.d_model = d_model
        # create constant 'pe' matrix with values dependent on 
        # pos and i
        pe = torch.zeros(L, d_model)
        if type(n_encode)==type(None):
            n_encode=3*d_model//4
        for pos in range(L):
            # Only going to fill 3/4 of the matrix so the
            # occupation values are preserved
            for i in range(0, n_encode, 2):
                #position encoding
                pe[pos, i] =                 math.sin(pos / (10000 ** ((2 * i)/n_encode)))
                pe[pos, i + 1] =             math.cos(pos / (10000 ** ((2 * (i + 1))/n_encode)))
        self.pe = pe.unsqueeze(1).to(device)
        self.L=L
    
    def forward(self, x):
        """
        Adds a 1D positional encoding of size d_model to x
        Inputs:
            Tensor of shape [L,B,?]
        Outputs:
            Tensor of shape [L,B,d_model]
        """
        if self.d_model%x.shape[-1]!=0:
            return x.repeat(1,1,self.d_model//x.shape[-1]+1)[:,:,:self.d_model] + self.pe[:x.shape[0]]
        return x.repeat(1,1,self.d_model//x.shape[-1]) + self.pe[:x.shape[0]]

    
##########################################################PTF Model#############################################################

class PTF(Sampler):
    """ 
    
    Architecture wise this is how it works:
    
    You give it a state and it patches it into groups of size p. It then tells you the probability of each potential patch given all previous patches in your sequence using masked attention.
    
    This model has 2**p outputs, which describes the probability distrubition for the nth patch when given the first n-1 patches
    
    """
    
    INFO = """Transformer based sampler where the input sequence is broken up into 'patches' and the output is a sequence of conditional probabilities of all possible patches at position i given the previous 0 to i-1 patches. Each patch is projected into a token with an added positional encoding. The sequence of encoded patches is used as transformer input.
    
    PTF Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice
    
        Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
    
        patch      (str)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/prod(patch).
                                Example values: 2x2, 2x3, 2, 4
            
        dropout    (float)   -- The amount of dropout to use in the transformer layers
        
        num_layers (int)     -- The number of transformer layers to use
        
        nhead     (int)      -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh
    
        repeat_pre (bool)    -- repeat the precondition instead of projecting it out
    """
    
    DEFAULTS=Options(L=16,patch=1,Nh=128,dropout=0.0,num_layers=2,nhead=8,repeat_pre=False)
    def __init__(self,L,patch,Nh,dropout,num_layers,nhead,repeat_pre,device=device, **kwargs):
        super(Sampler, self).__init__()
        
        if type(patch)==str and len(patch.split("x"))==2:
            #patch and system sizes
            px,py = [int(a) for a in patch.split("x")]
            Lx,Ly=[int(L**0.5)]*2 if type(L) is int else [int(a) for a in L.split("x")]
            #token size and positional encoder
            t_size = Nh if type(Nh) is int else Nh[0]
            self.pe = PE2D(t_size, Lx//px,Ly//py,device)
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

        if type(Nh) is int:
            Nh = [Nh]*4
        else:
            Nh += [self.L*Nh[0]] if _2D else [self.L*Nh[0]] 
            
        self.device=device
        
        self.tokenize=nn.Sequential(
                nn.Linear(self.p,Nh[0]),
                nn.Tanh()
        )
        
        #Encoder only transformer
        self.transformer = FastMaskedTransformerEncoder(Nh=Nh[0],dropout=dropout,num_layers=num_layers,nhead=nhead)       
        
        self.nrepeat = Nh[2]//Nh[1] if repeat_pre else 1
        
        self.lin = nn.Sequential(
                nn.Linear(Nh[1],(Nh[1] if repeat_pre else Nh[2])),
                nn.ReLU(),
                nn.Linear(Nh[0],1<<self.p),
                nn.Softmax(dim=-1)
            )
        
        self.lin0,self.lin1=self.lin[:2],self.lin[2:]
        
        self.set_mask(self.L)
        
        #create a tensor of all possible patches
        self.options=torch.zeros([1<<self.p,self.p],device=self.device)
        tmp=torch.arange(1<<self.p,device=self.device)
        for i in range(self.p):
            self.options[:,i]=(tmp>>i)%2
        
        self.to(device)
    
    def set_mask(self, L):
        # type: (int)
        """Initialize the self-attention mask"""
        # take the log of a lower triangular matrix
        self.L=L
        self.transformer.set_mask(L)
        self.pe.L=L

    @torch.jit.export
    def logprobability(self,input,h0=None):
        # type: (Tensor,Optional[Tensor]) -> Tensor
        """Compute the logscale probability of a given state
            Inputs:
                input - [B,L,1] matrix of zeros and ones for ground/excited states
            Returns:
                logp - [B] size vector of logscale probability labels
        """
        
        if input.shape[1]//self.p!=self.L:
            self.set_mask(input.shape[1]//self.p)
        #pe should be sequence first [L,B,Nh]
        
        #shape is modified to [L//p,B,p]
        input = self.patch(input.squeeze(-1)).transpose(1,0)
        
        #The first input should be zeros and the last patch is not used as input
        data=torch.zeros(input.shape,device=self.device)
        data[1:]=input[:-1]
        
        #[L//p,B,p] -> [L//p,B,Nh]
        encoded=self.pe(self.tokenize(data))
        
        if h0 is not None:
            
            L,B,Nh=encoded.shape
            #sequence is preconditioned with h0
            h0 = self.lin0(h0)
            #repeat h0 if necessary
            h0=h0.repeat(1,1,self.nrepeat)
            #project out h0
            #[1,B,Nh0] -> [L,B,Nh]
            h=h0.reshape([1,B,Nh,L]).transpose(-1,0).squeeze(-1)
            #output is shape [L//p,B,Nh]
            #precondition information is added to the sequence
            output = self.transformer(encoded+h)
            output=self.lin1(output)          
        else:
            #shape is preserved
            output = self.transformer(encoded)
            # [L//p,B,Nh] -> [L//p,B,2^p]
            output = self.lin(output)
        
        #real is going to be a onehot with the index of the appropriate patch set to 1
        #shape will be [L//p,B,2^p]
        real=genpatch2onehot(input,self.p)
        
        #[L//p,B,2^p] -> [L//p,B]
        total = torch.sum(real*output,dim=-1)
        #[L//p,B] -> [B]
        logp=torch.sum(torch.log(total),dim=0)
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
        
        if h0 is not None:
            
            #h0 is shape [1,B,Nh0], Nh0=L*Nh
            # Project out the precondition information
            h0 = self.lin0(h0)
            # repeat it if necessary
            h0=h0.repeat(1,1,self.nrepeat)
            
            #[1,B,Nh0] -> [L,B,Nh]
            h0=h0.reshape([1,B,self.transformer.Nh,L]).transpose(-1,0).squeeze(-1)
        
        #Sample set will have shape [L/p,B,p]
        #need one extra zero batch at the start for first pred hence input is [L+1,B,1] 
        input = torch.zeros([L+1,B,self.p],device=self.device)

        logp = torch.zeros([B],device=self.device)
        
        #make cache initially an empty tensor
        cache = torch.zeros([self.transformer.nl,0,B,self.transformer.Nh],device=self.device)
        
        for idx in range(1,L+1):
            
            #pe should be sequence first [l,B,Nh]
            # multiply by 1 to copy the tensor
            encoded_input = self.pe(self.tokenize(input[:idx,:,:]*1))
                        
            #check out the probability of all 16 vectors
            if h0 is not None:
                h = h0[:idx,:,:]
                #output is shape [?,B,Nh]
                output,cache = self.transformer.next_with_cache(encoded_input+h,cache)
                probs=self.lin1(output[-1,:,:]).view([B,1<<self.p])
            else:
                #Get transformer output
                output,cache = self.transformer.next_with_cache(encoded_input,cache)
                probs=self.lin(output[-1,:,:]).view([B,1<<self.p])

            #sample from the probability distribution
            indices = torch.multinomial(probs,1,False).squeeze(1)
            #extract samples
            sample = self.options[indices]
            
            onehot = nn.functional.one_hot(indices, num_classes=1<<self.p)
            logp+= torch.log(torch.sum(onehot*probs,dim=-1))
            
            #set input to the sample that was actually chosen
            input[idx] = sample
            
        #remove the leading zero in the input    
        input=input[1:]
        #sample is repeated 16 times at 3rd index so we just take the first one
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
            
        #compute all of their logscale probabilities            
        data=torch.zeros(sample.shape,device=self.device)
        data[1:]=sample[:-1]

        #[L//p,B,p] -> [L//p,B,Nh]
        encoded=self.pe(self.tokenize(data))

        #add positional encoding and make the cache
        out,cache=self.transformer.make_cache(encoded)
        probs=torch.zeros([B,L],device=self.device)
        #expand cache to group L//D flipped states
        cache=cache.unsqueeze(2)

        #the cache has to be repeated along the correct axis
        cache=cache.repeat(1,1,L//D,1,1).transpose(2,3).reshape(cache.shape[0],L//self.p,B*L//D,cache.shape[-1])

        pred0 = self.lin(out)
        #shape will be [L//p,B,2^p]
        real=genpatch2onehot(sample,self.p)
        #[L//p,B,2^p] -> [B,L//p]
        total0 = torch.sum(real*pred0,dim=-1).transpose(1,0)

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
            # put sequence before batch so you can use it with your transformer
            tgt=self.pe(self.tokenize(fsample))
            #grab your transformer output
            out,_=self.transformer.next_with_cache(tgt,cache[:,:N//self.p],N//self.p)

            # grab output for the new part
            output = self.lin(out[N//self.p:].transpose(1,0))
            # reshape output separating batch from spin flip grouping
            pred = output.view([B,L//D,(L-N)//self.p,1<<self.p])
            real = genpatch2onehot(real[:,:,N//self.p:],self.p)
            total = torch.sum(real*pred,dim=-1)
            #sum across the sequence for probabilities

            logp=torch.sum(torch.log(total),dim=-1)
            logp+=torch.sum(torch.log(total0[:,:N//self.p]),dim=-1).unsqueeze(-1)
            probs[:,N:(k+1)*L//D]=logp

        return probs

OptionManager.register("ptf",PTF.DEFAULTS)
    
    
