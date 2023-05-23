from tvmc.hamiltonians.hamiltonian import *
from tvmc.models.LPTF import *
from tvmc.hamiltonians.rydberg import Rydberg

def new_rnn_with_optim(rnntype,op,beta1=0.9,beta2=0.999):
    rnn = torch.jit.script(PRNN(op.L,**PRNN.DEFAULTS))
    optimizer = torch.optim.Adam(
    rnn.parameters(), 
    lr=op.lr, 
    betas=(beta1,beta2)
    )
    return rnn,optimizer

def momentum_update(m, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(target_param.data*m + param.data*(1.0-m))

# Setting Constants

import os
def mkdir(dir_):
    try:
        os.mkdir(dir_)
    except:return -1
    return 0

def setup_dir(op_dict):
    """Makes directory for output and saves the run settings there
    Inputs: 
        op_dict (dict) - Dictionary of Options objects
    Outputs:
        Output directory mydir 
    """
    op=op_dict["TRAIN"]
    
    if op.dir=="<NONE>":
        return
    
    hname = op_dict["HAMILTONIAN"].name if "HAMILTONIAN" in op_dict else "NA"
    
    mydir= op.dir+"/%s/%d-B=%d-K=%d%s"%(hname,op.L,op.B,op.K,op.sub_directory)

    os.makedirs(mydir,exist_ok = True)
    biggest=-1
    for paths,folders,files in os.walk(mydir):
        for f in folders:
            try:biggest=max(biggest,int(f))
            except:pass
            
    mydir+="/"+str(biggest+1)
    mkdir(mydir)
    
    print("Output folder path established")
    return mydir

class TrainOpt(Options):
    """
    Training Arguments:
    
        L          (int)     -- Total lattice size (8x8 would be L=64).
        
        Q          (int)     -- Number of minibatches per batch.
        
        K          (int)     -- size of each minibatch.
        
        B          (int)     -- Total batch size (should be Q*K).
        
        NLOOPS     (int)     -- Number of loops within the off_diag_labels function. Higher values save ram and
                                generally makes the code run faster (up to 2x). Note, you can only set this
                                as high as your effective sequence length. (Take L and divide by your patch size).
        
        steps      (int)     -- Number of training steps.
        
        dir        (str)     -- Output directory, set to <NONE> for no output.
        
        lr         (float)   -- Learning rate.
        
        seed       (int)     -- Random seed for the run.
                
        sgrad      (bool)    -- Whether or not to sample with gradients, otherwise create gradients in extra network run.
                                (Uses less ram when but slightly slower)
                                
        true_grad  (bool)    -- Set to false to approximate the gradients, more efficient but approximate.
                                
        sub_directory (str)  -- String to add to the end of the output directory (inside a subfolder).
        
    """
    def get_defaults(self):
        return dict(L=16,Q=1,K=256,B=256,NLOOPS=1,steps=50000,dir="out",lr=5e-4,seed=None,sgrad=False,true_grad=False,sub_directory="")

    
OptionManager.register("train",TrainOpt())
    
import sys
def reg_train(op,net_optim=None,printf=False,mydir=None):
  try:
    
    if "RYDBERG" in op:
        h = Rydberg(**op["RYDBERG"].__dict__)
    else:        
        h_opt=Rydberg.DEFAULTS.copy()
        h_opt.Lx=h_opt.Ly=int(op["TRAIN"].L**0.5)
        h = Rydberg(**h_opt.__dict__)
    
    if mydir==None:
        mydir = setup_dir(op)
    
    op=op["TRAIN"]
    
    if op.true_grad:assert op.Q==1
    
    if type(net_optim)==type(None):
        net,optimizer=new_rnn_with_optim("GRU",op)
    else:
        net,optimizer=net_optim

    debug=[]
    losses=[]
    true_energies=[]

    #samples
    samplebatch = torch.zeros([op.B,op.L,1],device=device)
    #sum of off diagonal labels for each sample (scaled)
    sump_batch=torch.zeros([op.B],device=device)
    #scaling factors for the off-diagonal sums
    sqrtp_batch=torch.zeros([op.B],device=device)

    def fill_batch():
        with torch.no_grad():
            for i in range(op.Q):
                sample,logp = net.sample(op.K,op.L)
                #get the off diagonal info
                sump,sqrtp = net.off_diag_labels_summed(sample,nloops=op.NLOOPS)
                samplebatch[i*op.K:(i+1)*op.K]=sample
                sump_batch[i*op.K:(i+1)*op.K]=sump
                sqrtp_batch[i*op.K:(i+1)*op.K]=sqrtp
        return logp
    i=0
    t=time.time()
    for x in range(op.steps):
        
        #gather samples and probabilities                
        if op.Q!=1:
            fill_batch()
            logp=net.logprobability(samplebatch)
        else:
            if op.sgrad:
                samplebatch,logp = net.sample(op.B,op.L)
            else:
                with torch.no_grad():samplebatch,_= net.sample(op.B,op.L)
                #if you sample without gradients you have to recompute probabilities with gradients
                logp=net.logprobability(samplebatch)
            
            if op.true_grad:
                sump_batch,sqrtp_batch = net.off_diag_labels_summed(samplebatch,nloops=op.NLOOPS)
            else:
                #don't need gradients on the off diagonal when approximating gradients
                 with torch.no_grad(): sump_batch,sqrtp_batch = net.off_diag_labels_summed(samplebatch,nloops=op.NLOOPS)

        #obtain energy
        with torch.no_grad():
            E=h.localenergyALT(samplebatch,logp,sump_batch,sqrtp_batch)
            #energy mean and variance
            Ev,Eo=torch.var_mean(E)
	    
            MAG, ABS_MAG, SQ_MAG, STAG_MAG  = h.magnetizations(samplebatch)
            mag_v, mag = torch.var_mean(MAG)
            abs_mag_v, abs_mag = torch.var_mean(ABS_MAG)
            sq_mag_v, sq_mag = torch.var_mean(SQ_MAG)
            stag_mag_v, stag_mag = torch.var_mean(STAG_MAG)

        ERR  = Eo/(op.L)
        
        if op.true_grad:
            #get the extra loss term
            h_x= h.offDiag *sump_batch* torch.exp(sqrtp_batch-logp/2)
            loss = (logp*E).mean() + h_x.mean()
            
        else:
            loss =(logp*(E-Eo)).mean()  if op.B>1 else (logp*E).mean()

        #Main loss curve to follow
        losses.append(ERR.cpu().item())
        
        #update weights
        net.zero_grad()
        loss.backward()
        optimizer.step()

        debug += [[Eo.item(), Ev.item(), mag.item(), mag_v.item(), abs_mag.item(), abs_mag_v.item(), sq_mag.item(), sq_mag_v.item(), stag_mag.item(), stag_mag_v.item(), time.time()-t]]

        if x%500==0:
            print(int(time.time()-t),end=",%.3f|"%(losses[-1]))
            if x%4000==0:print()
            if printf:sys.stdout.flush()
    print(time.time()-t,x+1)

    DEBUG = np.array(debug)
    
    if op.dir!="<NONE>":
        np.save(mydir+"/DEBUG",DEBUG)
        net.save(mydir+"/T")
        
  except KeyboardInterrupt:
    if op.dir!="<NONE>":
        DEBUG = np.array(debug)
        np.save(mydir+"/DEBUG",DEBUG)
        net.save(mydir+"/T")
  return DEBUG

        
