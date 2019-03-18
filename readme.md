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

![](https://latex.codecogs.com/png.latex?%5Cinline%20%24%24f_%7Bt%7D%28x%29%20%3D%20%5Cleft%5C%7B%20%5Cbegin%7Bmatrix%7D%201010x%2C%20%26%20%5Ctext%7Bwith%20probability%200.01%7D%5C%5C%20-10x%2C%20%26%20%5Ctext%7Botherwise.%7D%5C%5C%20%5Cend%7Bmatrix%7D%20%5Cright.%20%24%24)

## Reference
[1] On the Convergence of Adam and Beyond
[2] Re-implementation of AMSGrad in TensorFlow

