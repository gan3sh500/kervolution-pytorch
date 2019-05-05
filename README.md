## Kervolutional Neural Networks
A Pytorch implementation for the Kervolutional AKA Kernel Convolutional Layer from Kervolutional Neural Networks [[paper](https://arxiv.org/pdf/1904.03955.pdf)].
It is doing something very similar to Network in Network but using kernels to add the non-linearity instead. 

## Dependancies
```
pip install <pytorch-latest.whl url>
```

To use this layer:
```
from layer import KernelConv2d, GaussianKernel, PolynomialKernel
```
