import torch


class LinearKernel(torch.nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()
    
    def forward(self, x_unf, w, b):
        t = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        if b is not None:
            return t + b
        return t
        
        
class PolynomialKernel(LinearKernel):
    def __init__(self, cp=2.0, dp=3, train_cp=True):
        super(PolynomialKernel, self).__init__()
        self.cp = torch.nn.parameter.Parameter(torch.tensor(cp, requires_grad=train_cp))
        self.dp = dp

    def forward(self, x_unf, w, b):
        return (self.cp + super(PolynomialKernel, self).forward(x_unf, w, b))**self.dp


class GaussianKernel(torch.nn.Module):
    def __init__(self, gamma):
        super(GaussianKernel, self).__init__()
        self.gamma = torch.nn.parameter.Parameter(
                            torch.tensor(gamma, requires_grad=True))
    
    def forward(self, x_unf, w, b):
        l = x_unf.transpose(1, 2)[:, :, :, None] - w.view(1, 1, -1, w.size(0))
        l = torch.sum(l**2, 2)
        t = torch.exp(-self.gamma * l)
        if b:
            return t + b
        return t
        
       
class KernelConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_fn=PolynomialKernel,
                 stride=1, padding=0, dilation=1, groups=1, bias=None,
                 padding_mode='zeros'):
        '''
        Follows the same API as torch Conv2d except kernel_fn.
        kernel_fn should be an instance of the above kernels.
        '''
        super(KernelConv2d, self).__init__(in_channels, out_channels, 
                                           kernel_size, stride, padding,
                                           dilation, groups, bias, padding_mode)
        self.kernel_fn = kernel_fn()
   
    def compute_shape(self, x):
        h = (x.shape[2] + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w = (x.shape[3] + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return h, w
    
    def forward(self, x):
        x_unf = torch.nn.functional.unfold(x, self.kernel_size, self.dilation,self.padding, self.stride)
        h, w = self.compute_shape(x)
        return self.kernel_fn(x_unf, self.weight, self.bias).view(x.shape[0], -1, h, w)
