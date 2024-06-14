<img src="Drawing27.jpg" width="100%">

# LSN: Preserving Conservation Laws in Modelling Financial Market Dynamics via  Differential Equations


<!-- <img src="aa1.jpeg" width="60%"> -->
<!-- <img src=Figures/Drawing27.pdf width=400 height=300 > -->

<!-- <img src="fig2_mini_r_011_sigma_04.eps" width="60%"> -->



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This paper employs a novel Lie symmetry-based framework to model the intrinsic symmetries within financial market. More details can be found in the paper.

The repository contains the source code of our Lie Symmetry Net.

## Installation

```
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
import math, torch, time, os
import torch.nn as nn
import numpy as np
import argparse
import random
from torch.distributions import Normal
import torch.nn.functional as F
from pyDOE import lhs
```

## Run

For task start run this command from repository root directory:

```
python LSN.py 
```


