# Code accompanying the paper "Variational Monte Carlo with the Large Patched Transformer"

## Requirements
A suitable [conda](https://conda.io/) environment named `qsr` can be created
and activated with:

```
conda create --name qsr
conda install -n qsr pip
conda activate qsr
pip install -r requirements.txt
```

## Model builder

### TRAINING

This script is used to train new models from scratch. This is an example of a command
to train an $8\times 8$ Rydberg lattice with $V=7$, $\delta=\Omega=1$ with a $2\times 2$ patched transformer:
```
python train.py --train L=64 NLOOPS=16 K=1024 sub_directory=2x2 --ptf patch=2x2 --rydberg V=7 delta=1 Omega=1
```
Training parameters are shown when running:

```
python train.py --help --train
```

These are all possible training arguments:
```    

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
```

### RNN

All optional rnn parameters can be viewed by running 

```
python train.py --help --rnn
```

These are the RNN parameters:


```
    
    RNN Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- RNN hidden size.
    
        patch      (str)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/prod(patch).
                                Example values: 2x2, 2x3, 2, 4
        
        rnntype    (string)  -- Which type of RNN cell to use. Only ELMAN and GRU are valid options at the moment.
    

```

### Patched Transformer (PTF)


All optional ptf parameters can be viewed by running 

```
python train.py --help --ptf
```

These are your PTF parameters:
```
    
    PTF Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
    
        patch      (str)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/prod(patch).
                                Example values: 2x2, 2x3, 2, 4
            
        dropout    (float)   -- The amount of dropout to use in the transformer layers.
        
        num_layers (int)     -- The number of transformer layers to use.
        
        nhead     (int)      -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh.
    
        repeat_pre (bool)    -- Repeat the precondition (input) instead of projecting it out to match the token size.
    

```

### Large-Patched Transformer (LPTF)


All optional LPTF parameters can be viewed by running 

```
python train.py --help --lptf
```
LPTF parameters must be followed by the sub-model (e.g. --rnn) and the corresponding parameters, where the L parameter needs to match the patch parameter of the LPTF (e.g. --lptf path=2x3 --rnn L=2x3).

These are your LPTF parameters:
```
    
    
    LPTF Optional arguments:
    
        L          (int)     -- The total number of atoms in your lattice.
    
        Nh         (int)     -- Transformer token size. Input patches are projected to match the token size.
                                Note: When using an RNN subsampler this Nh MUST match the rnn's Nh.
    
        patch      (int)     -- Number of atoms input/predicted at once (patch size).
                                The Input sequence will have an effective length of L/patch.
            
        dropout    (float)   -- The amount of dropout to use in the transformer layers.
        
        num_layers (int)     -- The number of transformer layers to use.
        
        nhead     (int)     -- The number of heads to use in Multi-headed Self-Attention. This should divide Nh.
        
        subsampler (Sampler) -- The inner model to use for probability factorization. This is set implicitly
                                by including --rnn or --ptf arguments.
    
    

```

## Rydberg Hamiltonian

The following parameters can be chosen for the Rydberg Hamiltonian:

```
Lx                            			4
Ly                            			4
V                             			7.0
Omega                         			1.0
delta                         			1.0

```
