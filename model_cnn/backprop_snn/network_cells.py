import torch
import torch.nn as nn


class PseudoSpikeRect(torch.autograd.Function):
    """ Rectangular Pseudo-grad function """

    @staticmethod
    def forward(ctx, input, vth, grad_win, grad_amp):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.grad_win = grad_win
        ctx.grad_amp = grad_amp
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_win = ctx.grad_win
        grad_amp = ctx.grad_amp
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < grad_win
        return grad_amp * grad_input * spike_pseudo_grad.float(), None, None, None


class LinearLIFCell(nn.Module):
    def __init__(self, psp_func, pseudo_grad_ops, param):
        """

        Args:
            psp_func (Torch Function): Pre-synaptic function
            pseudo_grad_ops (Torch Function): Pseudo-grad function
            param (tuple): Cell parameters
        """
        super(LinearLIFCell, self).__init__()
        self.psp_func = psp_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.vdecay, self.vth, self.grad_win, self.grad_amp = param

    def forward(self, input_data, state):
        """
        Forward function

        Args:
            input_data (Tensor): input spike from pre-synaptic neurons
            state (tuple): output spike of last timestep and voltage of last timestep

        Returns:
            output: output spike
            state: updated neuron states

        """
        pre_spike, pre_volt = state
        volt = self.vdecay * pre_volt * (1. - pre_spike) + self.psp_func(input_data)
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win, self.grad_amp)
        return output, (output, volt)
