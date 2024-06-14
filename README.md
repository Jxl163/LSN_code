<img src="Figures/Drawing27.jpg" width="100%">

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
## Model

```
class Model(nn.Module):
    def __init__(self,
                 in_dims: int,
                 middle_dims: int,
                 out_dims: int,
                 depth: int):
        super(Model, self).__init__()

        self.linearIn = nn.Linear(in_dims, middle_dims)

        self.linear = nn.ModuleList()
        for _ in range(depth):
            linear = nn.Linear(middle_dims, middle_dims)
            self.linear.append(linear)

        self.linearOut = nn.Linear(middle_dims, out_dims)

        self.act = nn.Tanh()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        out = self.act(self.linearIn(x))

        for layer in self.linear:
            out = self.act(layer(out))

        out = self.linearOut(out)

        return out
```
## Run

For task start run this command from repository root directory:

```
python LSN.py 
```


