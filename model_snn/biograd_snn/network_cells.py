import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from quantization import (
    print_stats, quantize_layer_weights_bias, quantize_layer_weights,
    truncated_division, stochastic_rounding, scale_weight
)


class OnlineHiddenCell:
    """ Online Fully-Connected Spiking Neuron Cell for Hidden Layers """

    def __init__(self, forward_func, feedback_func, neuron_param, input_dim, output_dim, float_mode=False):
        """
        Args:
            forward_func (Torch Function): Pre-synaptic function for forward connection
            feedback_func (Torch Function): Feedback function for feedback connection
            neuron_param (dict): LIF neuron parameters
            input_dim (int): input dimension
            output_dim (int): output dimension
            float_mode (bool): Whether to use float mode
        """
        self.forward_func = forward_func
        self.feedback_func = feedback_func
        self.param_dict = neuron_param
        self.vdecay = neuron_param['Vdecay']
        self.vth = torch.tensor(neuron_param['Vth'])
        self.grad_win = torch.tensor(neuron_param['Grad_win'])
        self.grad_amp = neuron_param['Grad_amp']
        self.lr = neuron_param['lr']
        self.weight_decay = neuron_param['Weight_decay']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.float_mode = float_mode
        self.rounding_function = torch.floor

    def train_reset_state(self, batch_size, device):
        """
        At start of training, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (torch.device): device

        Returns:
            tuple: forward_state, feedback_state
        """
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        trace_pre = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # pre-spike trace
        trace_dw = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # gradient trace for weight
        trace_bias = torch.zeros([batch_size, self.output_dim], device=device)  # bias-spike trace
        trace_db = torch.zeros([batch_size, self.output_dim], device=device)  # gradient trace for bias
        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db)

        feedback_state = torch.zeros([batch_size, self.output_dim], device=device)  # error dendrite volt

        return forward_state, feedback_state

    def quantize(self, num_bits, stats, plot, max_val=0, num_bits_feedback=0):
        """
        Quantize forward and feedback weights

        Args:
            num_bits (int): Number of bits for quantization
            stats (bool): Whether to print stats
            plot (bool): Whether to plot stats
            max_val (float): Maximum value for quantization
            num_bits_feedback (int): Number of bits for feedback quantization
        """
        if stats:
            print("Forward Function")
        self.forward_func.weight.data, step_fw, exp_scale_fw = quantize_layer_weights(
            self.forward_func.weight.data, num_bits, stats=stats, plot=plot, max_val=max_val
        )
        if stats:
            print("Feedback Function")
        num_bits_fb = num_bits if num_bits_feedback == 0 else num_bits_feedback
        self.feedback_func.weight.data, step_fb, exp_scale_fb = quantize_layer_weights(
            self.feedback_func.weight.data, num_bits_fb, stats=stats, plot=plot, max_val=max_val
        )
        self.w_fw_exp = exp_scale_fw
        self.w_fb_exp = exp_scale_fb

    def train_forward_step(self, spike_input, forward_state, stats=False):
        """
        One step forward connection simulation for the neuron training

        Args:
            spike_input (torch.Tensor): Spike input from pre-synaptic input
            forward_state (tuple): Forward neuron states

        Returns:
            tuple: spike_output, updated forward_state, max_volt, activation
        """
        volt, spike, trace_pre, trace_dw, trace_bias, trace_db = forward_state

        if stats:
            print_stats(volt.cpu().detach().numpy(), "Volt pre")

        forward_act = self.forward_func_quant(spike_input)
        if not self.float_mode:
            forward_act, _ = scale_weight(forward_act, self.act_bits)
        volt = self.vdecay * volt * (1. - spike) + forward_act
        if not self.float_mode:
            volt = self.rounding_function(volt)

        spike_output = volt.gt(self.vth).float()
        max_volt = torch.max(torch.abs(volt))

        volt_pseudo_grad = (abs(volt - self.vth) < self.grad_win).float() * self.grad_amp
        trace_pre = self.vdecay * trace_pre + spike_input.view(-1, 1, self.input_dim)
        if not self.float_mode:
            trace_pre = self.rounding_function(trace_pre)
        trace_dw = trace_dw + trace_pre * volt_pseudo_grad.view(-1, self.output_dim, 1)

        activation = 100 * torch.sum(spike_output).item() / spike_output.numel()

        if stats:
            print("Activation %:", activation)

        return spike_output, (volt, spike_output, trace_pre, trace_dw, trace_bias, trace_db), max_volt, activation

    def train_feedback_step(self, pos_spike_input, neg_spike_input, feedback_state, stats=False):
        """
        One step feedback connection simulation for the neuron training

        Args:
            pos_spike_input (Tensor): spike input from downstream positive error neuron
            neg_spike_input (Tensor): spike input from downstream negative error neuron
            feedback_state (tuple): feedback neuron states

        Returns:
            feedback_state: updated feedback neuron states
        """
        if self.loss_mode == 'direct':
            error_dendrite_volt = feedback_state + self.feedback_func_quant(pos_spike_input)
        else:
            error_dendrite_volt = feedback_state + (self.feedback_func(pos_spike_input) - self.feedback_func(neg_spike_input))

        return error_dendrite_volt

    def train_update_parameter_sgd(self, update_state, stats=False):
        """
        Update parameter using vanilla SGD

        Args:
            update_state (tuple): neuron states used for update parameter

        Returns:
            error: estimated error for hidden neurons by direct feedback connection
        """
        error_dendrite_volt, error_steps, trace_dw, trace_db = update_state
        if self.float_mode:
            error = error_dendrite_volt / error_steps
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw
            mean_dw = mean_dw.sum(0) / mean_dw.size(0)
            delta = self.lr * mean_dw
        else:
            error = error_dendrite_volt // error_steps
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw
            mean_dw = mean_dw.sum(0)
            mean_dw = mean_dw * self.lr
            mean_dw = self.rounding_function(mean_dw)
            if self.gradient_clip > 0:
                mean_dw = torch.clamp(mean_dw, -self.gradient_clip, self.gradient_clip)
            delta = mean_dw

        wd_fw = self.weight_decay * self.forward_func.weight.data
        if not self.float_mode:
            wd_fw = self.rounding_function(wd_fw)

        delta_p = 100 * delta / self.forward_func.weight.data
        delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)

        self.forward_func.weight.data = self.forward_func.weight.data - delta - wd_fw

        return error, (delta_p, delta)

    def test_reset_state(self, batch_size, device):
        """
        At start of testing, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (torch.device): device

        Returns:
            tuple: forward_state
        """
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        forward_state = (volt, spike)

        return forward_state

    def test_forward_step(self, spike_input, forward_state, stats=False):
        """
        One step forward connection simulation for the neuron (test only)

        Args:
            spike_input (torch.Tensor): spike input from pre-synaptic input
            forward_state (tuple): forward neuron states

        Returns:
            tuple: spike_output, updated forward_state, activation
        """
        volt, spike = forward_state

        if stats:
            print_stats(volt.cpu().detach().numpy(), "Volt pre")

        forward_act = self.forward_func_quant(spike_input)
        if not self.float_mode:
            forward_act, _ = scale_weight(forward_act, self.act_bits)

        volt = self.vdecay * volt * (1. - spike) + forward_act
        if not self.float_mode:
            volt = self.rounding_function(volt)
        spike_output = volt.gt(self.vth).float()
        activation = 100 * torch.sum(spike_output).item() / spike_output.numel()

        return spike_output, (volt, spike_output), activation

    def sleep_forward_step(self, spike_input_pos, spike_input_neg, forward_state):
        """
        One step forward connection simulation for sleep phase of the neuron

        Args:
            spike_input_pos (torch.Tensor): positive Poisson spike input
            spike_input_neg (torch.Tensor): negative Poisson spike input
            forward_state (tuple): forward neuron states

        Returns:
            tuple: spike_output, updated forward_state
        """
        volt, spike = forward_state

        volt = self.vdecay * volt * (1. - spike) + self.forward_func(spike_input_pos) - self.forward_func(spike_input_neg)
        spike_output = volt.gt(self.vth).float()

        return spike_output, (volt, spike_output)


class OnlineOutputCell(OnlineHiddenCell):
    """ Online Fully-Connected Spiking Neuron Cell for Output Layer (including error interneurons) """

    def __init__(self, forward_func, neuron_param, input_dim, output_dim, float_mode=False):
        """
        Args:
            forward_func (Torch Function): Pre-synaptic function for forward
            neuron_param (dict): LIF neuron and feedback parameters
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        self.forward_func = forward_func
        self.feedback_func = nn.Identity()
        self.vdecay = neuron_param['Vdecay']
        self.vth = torch.tensor(neuron_param['Vth'])
        self.grad_win = torch.tensor(neuron_param['Grad_win'])
        self.grad_amp = neuron_param['Grad_amp']
        self.feedback_th = neuron_param['Fb_th']
        self.lr = neuron_param['lr']
        self.weight_decay = neuron_param['Weight_decay']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.float_mode = float_mode
        self.rounding_function = torch.floor

    def train_reset_state(self, batch_size, device):
        """
        At start of training, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (torch.device): device

        Returns:
            tuple: forward_state, feedback_state
        """
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        trace_pre = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # pre-spike trace
        trace_dw = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # gradient trace for weight
        trace_bias = torch.zeros([batch_size, self.output_dim], device=device)  # bias-spike trace
        trace_db = torch.zeros([batch_size, self.output_dim], device=device)  # gradient trace for bias
        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db)

        error_pos_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error pos neuron volt
        error_neg_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error neg neuron volt
        error_pos_spike = torch.zeros([batch_size, self.output_dim], device=device)  # error pos neuron spike
        error_neg_spike = torch.zeros([batch_size, self.output_dim], device=device)  # error neg neuron spike
        error_dendrite_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error dendrite volt
        feedback_state = (error_pos_volt, error_neg_volt, error_pos_spike, error_neg_spike, error_dendrite_volt)

        return forward_state, feedback_state

    def quantize_output_cell(self, num_bits, stats, plot, max_val=0):
        """
        Quantize forward weights

        Args:
            num_bits (int): Number of bits for quantization
            stats (bool): Whether to print stats
            plot (bool): Whether to plot stats
            max_val (float): Maximum value for quantization
        """
        if stats:
            print("Forward Function")
        self.forward_func.weight.data, step_fw, exp_scale_fw = quantize_layer_weights(
            self.forward_func.weight.data, num_bits, stats=stats, plot=plot, max_val=max_val
        )
        self.w_fw_exp = exp_scale_fw

    def train_feedback_step(self, pos_input, neg_input, feedback_state, stats=False):
        """
        One step feedback simulation for the neuron

        Args:
            pos_input (torch.Tensor): current input from error computation
            neg_input (torch.Tensor): current input from error computation
            feedback_state (tuple): feedback neuron states

        Returns:
            tuple: pos_spike_output, neg_spike_output, updated feedback_state, activations
        """
        error_pos_volt, error_neg_volt, error_pos_spike, error_neg_spike, error_dendrite_volt = feedback_state

        error_neuron_psp = pos_input - neg_input

        error_pos_volt = error_pos_volt - error_pos_spike + error_neuron_psp
        pos_spike_output = error_pos_volt.gt(self.feedback_th).float()

        error_neg_volt = error_neg_volt - error_neg_spike - error_neuron_psp
        neg_spike_output = error_neg_volt.gt(self.feedback_th).float()

        error_dendrite_volt = error_dendrite_volt + (pos_spike_output - neg_spike_output)

        pos_activation = 100 * torch.sum(pos_spike_output).item() / pos_spike_output.numel()
        neg_activation = 100 * torch.sum(neg_spike_output).item() / neg_spike_output.numel()
        activations = torch.tensor([pos_activation, neg_activation])

        return pos_spike_output, neg_spike_output, (
            error_pos_volt, error_neg_volt, pos_spike_output, neg_spike_output, error_dendrite_volt), activations