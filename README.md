# Lie Symmetry Net (LSN)

The repository contains the source code of our Lie Symmetry Net.

step 1: install pytorch

	conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
	
step 2: install pyDOE	
   
        pip install pyDOE
        
step 3:
        
        python LSN.py
	
        # for the result of PINNs, please run python LSN.py while parameter "lambda_con = 0"; for the result of LPS, please run python LPS.py.
	
        
