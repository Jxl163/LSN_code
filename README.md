## This repository contains the code for the paper:
# " LSN: Preserving Conservation Laws in Modelling Financial Market Dynamics via  Differential Equations"


<!-- <img src="aa1.jpeg" width="60%"> -->
<!-- <img src=Figures/Drawing27.pdf width=400 height=300 > -->

<!-- <img src="fig2_mini_r_011_sigma_04.eps" width="60%"> -->


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![](https://img.shields.io/badge/pytorch-1.13.1-red)
![](https://img.shields.io/badge/cuda-11.7-blue)

<!--[![arXiv](https://img.shields.io/badge/arXiv-2406.09189-b31b1b.svg)](https://arxiv.org/abs/2406.09189)-->


This paper introduce LSN, a symmetry-aware approach that addresses a fundamental challenge in AI-driven SDE solvers: ensuring AI models can learn and preserve intrinsic symmetries from data. By incorporating Lie symmetry principles, **LSN achieves a significant reduction in test error—over an order of magnitude—compared to state-of-the-art AI-driven methods**. The framework is not limited to specific equations or methods but provides a universal solution that can be applied across various AI-driven differential equation solvers.

[![Typing SVG](https://readme-typing-svg.demolab.com/?lines=The+repository+contains+the;source+code+of+our+LSN!)](https://git.io/typing-svg)



## Installation

```
Step 1: install pytorch

 conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
 
Step 2: install pyDOE 
   
 pip install pyDOE

```
## Schematic Diagram of Lie Symmetry Network

<img src="Figures/Drawing27.jpg" width="100%">

## Usage

<!--For task start run this command from repository root directory:-->

```
python LSN.py 
```

<!--## Citation

Please consider citing our paper if you find this repo useful in your work.

```
@article{jiang2024lie,
  title={Lie Symmetry Net: Preserving Conservation Laws in Modelling Financial Market Dynamics via Differential Equations},
  author={Jiang, Xuelian and Zhu, Tongtian and Wang, Can and Xu, Yingxiang and He, Fengxiang},
  journal={arXiv preprint arXiv:2406.09189},
  year={2024}
}
```-->

