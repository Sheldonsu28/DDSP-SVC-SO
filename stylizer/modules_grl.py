"""Selected dependencies of the SoftVcStylizer frontend workflow."""

import torch
from torch import nn
from torch.autograd import Function

class GradientReversalFunction(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return (dx, None)

class GradientReversal(torch.nn.Module):
    """ Gradient Reversal Layer
            Y. Ganin, V. Lempitsky,
            "Unsupervised Domain Adaptation by Backpropagation",
            in ICML, 2015.
        Forward pass is the identity function
        In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    """

    def __init__(self, lambda_reversal=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
