from tvmc.models.ModelBuilder import *

import sys

def helper(args):
    
    help(build_model)
    
    example = "Runtime Example:\n>>>python train.py --rydberg --train L=144"
    while True:
        if "--lptf" in args:
            print(LPTF.INFO)
            print(example+" --lptf patch=3x3 --rnn L=9 patch=3 Nh=128")
            break
        if "--rnn" in args:
            print(PRNN.INFO)
            print(example+" NLOOPS=36 --rnn patch=4")
            break
        if "--ptf" in args:
            print(PTF.INFO)
            print(example+" NLOOPS=24 --ptf patch=2x3")
            break
        if "--train" in args:
            print(TrainOpt.__doc__)
            print(example+" NLOOPS=36 sgrad=False steps=4000 --ptf patch=2x2")
            break
            
        args=["--"+input("What Model do you need help with?\nOptions are rnn, lptf, ptf, and train:\n".lower())]
        


if "--help" in sys.argv:
    print()
    helper(sys.argv)
else:
    print(sys.argv[1:])

    model,full_opt,opt_dict = build_model(sys.argv[1:])
    train_opt=opt_dict["TRAIN"]

    #Initialize optimizer
    beta1=0.9;beta2=0.999
    optimizer = torch.optim.Adam(
	    model.parameters(), 
    	lr=train_opt.lr, 
    	betas=(beta1,beta2)
    )

    print(full_opt)
    mydir=setup_dir(opt_dict)
    orig_stdout = sys.stdout

    full_opt.save(mydir+"\\settings.json")

    f = open(mydir+'\\output.txt', 'w')
    sys.stdout = f
    try:
        reg_train(opt_dict,(model,optimizer),printf=True,mydir=mydir)
    except Exception as e:
        print(e)
        sys.stdout = orig_stdout
        f.close()
        1/0
    sys.stdout = orig_stdout
    f.close()
