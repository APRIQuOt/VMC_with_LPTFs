from tvmc.util import Options,OptionManager
from tvmc.models.LPTF import *
from tvmc.models.training import *
from tvmc.hamiltonians.rydberg import Rydberg

import random

def build_model(args):
    """
    Builds a Sampler network using command line arguments
    
    CMD arguments should look like this:
    
    >>> python train.py --<param1> <name11>=<value11> <name12>=<value12> --<param2> <name21>=<value21> <name22>=<value22> . . .
    
    Ex: A Patched Transformer with 2x2 patches, system total size of 8x8, a batch size of K*Q=1024 and 16 loops when calculating
        the off diagonal probabilities to save on memory:
    
    >>> python train.py --train L=64 NLOOPS=16 K=1024 sub_directory=2x2 --ptf patch=2x2
    
    Ex2: A Large Patched Transformer using an RNN subsampler with 3x3 patches on the LPTF and 1D patches of size 3 on the RNN
    
    >>> python train.py --rydberg --train L=576 NLOOPS=64 sub_directory=3x3 --lptf patch=3x3 --rnn L=9 patch=3 Nh=128
    
    """
    

    options_dict = OptionManager.parse_cmd(args)
    is_lptf= ("LPTF" in options_dict)
    all_models=dict(RNN=PRNN,LPTF=LPTF,PTF=PTF)
    
    
    if not "TRAIN" in options_dict:
        options_dict["TRAIN"]=None
        for name in options_dict:
            if name in all_models and (not is_lptf or name=="LPTF"):
                options_dict["TRAIN"] = TrainOpt(L=options_dict[name].L)
    
    
    if options_dict["TRAIN"].seed is None:
        options_dict["TRAIN"].seed = np.random.randint(65536)
    
    torch.manual_seed(options_dict["TRAIN"].seed)
    np.random.seed(options_dict["TRAIN"].seed)
    random.seed(options_dict["TRAIN"].seed)
        
    HAMILTONIAN = None
    for name in options_dict:
        #make sure system size is consistent among all options
        if name == "LPTF" or is_lptf==False:
            options_dict[name].L=options_dict["TRAIN"].L
            if options_dict["TRAIN"].dir == "out" and name in all_models:
                options_dict["TRAIN"].dir=name
        #make sure hamiltonians have correct system size
        if (not name in all_models and name != "TRAIN"):
            options_dict[name].L=options_dict["TRAIN"].L
            HAMILTONIAN = options_dict[name]
            options_dict[name].name=name
            if name=="RYDBERG":
                h=options_dict[name]
                if h.Lx*h.Ly!=h.L:
                    h.Lx=h.Ly=int(h.L**0.5)
        #set model type
        if name in all_models:
            options_dict[name].model_name=all_models[name].__name__
            if not is_lptf or name!="LPTF":
                SMODEL,sub_opt = all_models[name],options_dict[name]
                
    #Special case for no hamiltonian specified
    if HAMILTONIAN is None: 
        HAMILTONIAN=h=Rydberg.DEFAULTS.copy()
        h.L=options_dict["TRAIN"].L
        if h.Lx*h.Ly!=h.L:h.Lx=h.Ly=int(h.L**0.5)
        
    options_dict["HAMILTONIAN"]=HAMILTONIAN
    
    #Make sure batch size makes sense
    train_opt=options_dict["TRAIN"]
    train_opt.B=train_opt.K*train_opt.Q
    
    # Build models
    #for the lptf we need to have a model and submodel
    if is_lptf:
        lptf_opt=options_dict["LPTF"]
        #extra condition on the PTF to make the conditioned sampling work
        if SMODEL==PTF:
            sub_opt.Nh=[sub_opt.Nh,lptf_opt.Nh]
        else:
            sub_opt.Nh = lptf_opt.Nh
        
        subsampler = SMODEL(**sub_opt.__dict__)
        #set lptf options
        #make lptf model and global settings
        model = torch.jit.script(LPTF(subsampler,**lptf_opt.__dict__))
        full_opt = Options(train=train_opt.__dict__,model=lptf_opt.__dict__,
                           submodel=sub_opt.__dict__,hamiltonian=HAMILTONIAN.__dict__)
    else:
        #set model to submodel and create global settings
        full_opt = Options(train=train_opt.__dict__,model=sub_opt.__dict__,hamiltonian=HAMILTONIAN.__dict__)
        model = torch.jit.script(SMODEL(**sub_opt.__dict__))
        
    return model,full_opt,options_dict


