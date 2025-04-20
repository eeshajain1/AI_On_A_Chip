import torch

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        #Implement your quantization function here
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass the gradients are returned directly without modification.
        # This is the key step of STE
        return grad_output

# To apply the STE
def apply_ste(x):
    return StraightThroughEstimator.apply(x)

# When you want to quantize the weights, call the apply_ste function
# You need to use this function within the forward pass of your model in a custom Conv2d class.
