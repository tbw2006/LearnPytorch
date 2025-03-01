import torch
class CustomRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input.clamp(min = 0)
    @staticmethod
    def backward(ctx, grad_output):
        input, =ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 2] = 0.5
        return grad_input
x = torch.tensor([-1.,1.,3.],requires_grad=True)
y = CustomRelu.apply(x)
print(y)
y.backward(torch.tensor([1.,1.,1.]))
print(x.grad)