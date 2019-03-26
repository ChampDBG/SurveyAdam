# A survey for Adam
## Introduction
This repo is an surver of optimizer. 

According to [[1](https://openreview.net/pdf?id=ryQu7f-RZ)], Adam is still not good enough, and purposed AMSGrad. I will re-implement the experiment in [1] first, and then do some other experiments about them.

## Environment
* Python3.6
* TensorFlow 1.5

## Experiment
### Example in [1]
Author offer a simple example to argue that Adam is not good enough. In example, we want to optimize the following loss function.

![](https://i.imgur.com/O1fxGc3.png)

To simplify problem, the x is bounded in [-1, 1]. With calculating expected value of loss function. The optimal solution is x = -1, but Adam failed to find optimal solution in experiment.

#### Result
<img src="./img/Adam_vs_AMSGrad_SimpleExample.png" width="400">

## Reference
[1] [On the Convergence of Adam and Beyond](https://openreview.net/pdf?id=ryQu7f-RZ)

[2] [Re-implementation of AMSGrad in TensorFlow](https://colab.research.google.com/drive/1xXFAuHM2Ae-OmF5M8Cn9ypGCa_HHBgfG)