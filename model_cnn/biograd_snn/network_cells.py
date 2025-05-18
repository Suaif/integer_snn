import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization import print_stats, quantize_layer_weights_bias, quantize_layer_weights, truncated_division, stochastic_rounding, scale_weight


class OnlineHiddenCell:
    """ Online Fully-Connected Spiking Neuron Cell for Hidden Layers """

    def __init__(self, forward_func, feedback_func, neuron_param, input_dim, output_dim, float_mode=False):
        """

        Args:
            forward_func (Torch Function): Pre-synaptic function for forward connection
            feedback_func (Torch Function): Feedback function for feedback connection
            neuron_param (tuple): LIF neuron parameters
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        self.forward_func = forward_func
        self.feedback_func = feedback_func
        self.vdecay, self.vth, self.grad_win, self.grad_amp = neuron_param['Vdecay'], neuron_param['Vth'], neuron_param['Grad_win'], neuron_param['Grad_amp']
        self.n_filters, self.kernel_size, self.reduce_dim = neuron_param['n_filters'], neuron_param['kernel_size'], neuron_param['reduce_dim']
        self.lr, self.weight_decay = neuron_param['lr'], neuron_param['Weight_decay']
        self.vth, self.grad_win = torch.tensor(self.vth), torch.tensor(self.grad_win)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim_n = self.n_filters * output_dim[0] * output_dim[1]
        self.float_mode = float_mode
        self.hidden_layer = True
        self.rounding_function = torch.floor
        # self.rounding_function = stochastic_rounding

    def train_reset_state(self, batch_size, device):
        """
        At start of training, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (device): device

        Returns:
            forward_state: forward neuron states
            feedback_state: feedback neuron state

        """
        # Forward neuron states
        volt = torch.zeros([batch_size, self.n_filters, self.output_dim[0], self.output_dim[1]], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.n_filters, self.output_dim[0], self.output_dim[1]], device=device)  # soma spike
        trace_pre = torch.zeros([batch_size, self.channels_in, self.input_dim[0], self.input_dim[1]], device=device)  # pre-spike trace
        trace_dw = torch.zeros([batch_size, self.n_filters, self.output_dim[0], self.output_dim[1], self.channels_in, self.kernel_size, self.kernel_size], device=device)  # gradient trace for weight
        trace_bias = torch.zeros([batch_size, self.output_dim_n], device=device)  # bias-spike trace (spike all step)
        trace_db = torch.zeros([batch_size, self.output_dim_n], device=device)  # gradient trace for bias
        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db)

        # Feedback neuron states
        feedback_state = torch.zeros([batch_size, self.output_dim_n], device=device)  # error dendrite volt

        return forward_state, feedback_state

    def quantize(self, num_bits, stats, plot, max_val=0, num_bits_feedback=0):

        # Quantize forward and feedback weights
        if stats:
            print("Forward Function")
        self.forward_func.weight.data, step_fw, exp_scale_fw = quantize_layer_weights(
            self.forward_func.weight.data, num_bits, stats=stats, plot=plot, max_val=max_val)
        if stats:
            print("Feedback Function")
        num_bits_fb = num_bits if num_bits_feedback == 0 else num_bits_feedback
        self.feedback_func.weight.data, step_fb, exp_scale_fb = quantize_layer_weights(
            self.feedback_func.weight.data, num_bits_fb, stats=stats, plot=plot, max_val=max_val)
        self.w_fw_exp = exp_scale_fw
        self.w_fb_exp = exp_scale_fb
        
    def train_forward_step(self, spike_input, forward_state, stats=False):
        """
        One step forward connection simulation for the neuron training

        Args:
            spike_input (Tensor): spike input from pre-synaptic input
            forward_state (tuple): forward neuron states

        Returns:
            spike_output: spike output to downstream layer
            forward_state: updated forward neuron states

        """
        volt, spike, trace_pre, trace_dw, trace_bias, trace_db = forward_state

        # Update neuron soma (LIF neuron)
        if stats:
            print_stats(volt.cpu().detach().numpy(), "Volt pre        ")

        forward_act = self.forward_func_quant(spike_input)
        if not self.float_mode:
            forward_act, exp = scale_weight(forward_act, self.act_bits)
        volt = self.vdecay * volt * (1. - spike) + forward_act
        if not self.float_mode:
            volt = self.rounding_function(volt)

        if self.writer_spike:
            self.tf_writer.add_histogram(f"activations_volts/{self.tf_name}_Volt", volt, self.spike_time)
            self.tf_writer.add_histogram(f"{self.tf_name}/Forward_Act", forward_act, self.spike_time)
            
        if stats:
            print_stats(forward_act.cpu().detach().numpy(), "Forward Act     ")
            print_stats(volt.cpu().detach().numpy(),        "Volt post       ")
            
        spike_output = volt.gt(self.vth).float()
        max_volt = torch.max(torch.abs(volt))

        # Update neuron traces
        volt_pseudo_grad = (abs(volt - self.vth) < self.grad_win).float() * self.grad_amp

        if self.hidden_layer:
            trace_pre = self.vdecay * trace_pre + spike_input
            # trace_dw = trace_dw + trace_pre * average_kernel_inputs(volt_pseudo_grad, self.kernel_size).sum(1)
            input_patches = F.unfold(trace_pre, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            input_patches = input_patches.permute(0, 2, 1).contiguous() # (batch, 14*14, channels_in * kernel * kernel)
            input_patches = input_patches.view(-1, self.output_dim[0], self.output_dim[1], self.channels_in, self.kernel_size, self.kernel_size)
            input_patches = input_patches.unsqueeze(1).expand(-1, self.n_filters, -1, -1, -1, -1, -1)
            if not self.float_mode:
                trace_pre = self.rounding_function(trace_pre)
            trace_dw = trace_dw + input_patches * volt_pseudo_grad[..., None, None, None]

        else:
            trace_pre = self.vdecay * trace_pre + spike_input.view(-1, 1, self.input_dim)
            if not self.float_mode:
                trace_pre = self.rounding_function(trace_pre)
            trace_dw = trace_dw + trace_pre * volt_pseudo_grad.view(-1, self.output_dim, 1)
       
        activation = 100 * torch.sum(spike_output).item() / (volt.numel())

        if self.writer_spike:
            self.tf_writer.add_histogram(f"{self.tf_name}/input", spike_input, self.spike_time)
            self.tf_writer.add_scalar(f"activations_volts/{self.tf_name}_Activation", activation, self.spike_time)
            self.tf_writer.add_histogram(f"{self.tf_name}/Trace_Pre", trace_pre, self.spike_time)
            self.tf_writer.add_histogram(f"{self.tf_name}/Trace_Dw", trace_dw, self.spike_time)
        
        if stats:
            print("Activation %: ", activation)
            print_stats(spike_output.cpu().detach().numpy(),     "Spike           ")
            print_stats(volt_pseudo_grad.cpu().detach().numpy(), "Volt Pseudo Grad")
            print_stats(trace_pre.cpu().detach().numpy(),        "Trace Pre       ")
            print_stats(trace_dw.cpu().detach().numpy(),         "Trace Dw        ")
            # print(spike_input.shape, volt.shape, self.forward_func.weight.data.shape)
            # print(trace_pre.shape, trace_dw.shape, input_patches.shape)

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
        # Update error dendrite
        if self.loss_mode == 'direct':
            error_dendrite_volt = self.feedback_func_quant(pos_spike_input)
        else:
            error_dendrite_volt = feedback_state + (self.feedback_func(pos_spike_input) - self.feedback_func(neg_spike_input))

        # if not self.float_mode:
        #     error_dendrite_volt = self.rounding_function(error_dendrite_volt * 2**self.forward_exp[4])

        if stats:
            print_stats(error_dendrite_volt.cpu().detach().numpy(), "Error Dendrite Volt")
            print(pos_spike_input.shape, error_dendrite_volt.shape, self.feedback_func.weight.data.shape)
        if self.writer_batch:
            self.tf_writer.add_histogram(f"Training_batch/Error_Dendrite_Volt", error_dendrite_volt, self.n_batch)

        # error_dendrite_volt = torch.clamp(error_dendrite_volt, -2**(self.n_bits-1), 2**(self.n_bits-1) - 1)
        return error_dendrite_volt

    def train_update_parameter_sgd(self, update_state, stats=False):
        """
        Update parameter using vanilla SGD

        Args:
            update_state (tuple): neuron states used for update parameter
            lr (float): learning rate

        Returns:
            error: estimated error for hidden neurons by direct feedback connection

        """
        error_dendrite_volt, error_steps, trace_dw, trace_db = update_state
        if self.float_mode:
            error = error_dendrite_volt / error_steps
            mean_dw = error.view(-1, self.n_filters, self.output_dim[0], self.output_dim[1], 1, 1, 1) * trace_dw # (batch, n_filters, 14, 14, channels_in, kernel, kernel)
            mean_dw = mean_dw.mean((0, 2, 3)) # (n_filters, channels_in, kernel, kernel)
            delta = self.lr * mean_dw
        else:
            error = error_dendrite_volt // error_steps
            # error = truncated_division(error_dendrite_volt, error_steps)
            mean_dw = error.view(-1, self.n_filters, self.output_dim[0], self.output_dim[1], 1, 1, 1) * trace_dw # (batch, n_filters, 14, 14, kernel, kernel)
            mean_dw = mean_dw.sum((0, 2, 3)) # (n_filters, channels_in, kernel, kernel)
            mean_dw = mean_dw * self.lr
            delta = self.rounding_function(mean_dw)
            if self.gradient_clip > 0:
                mean_dw = torch.clamp(mean_dw, -self.gradient_clip, self.gradient_clip)
            delta = mean_dw
        
        # Weight decay term
        if not self.float_mode:
            weight_decay_term = self.forward_func.weight.data * self.weight_decay
            weight_decay_term = self.rounding_function(weight_decay_term)
        else:
            weight_decay_term = self.weight_decay * self.forward_func.weight.data * self.forward_func.weight.data
            weight_decay_term = weight_decay_term
                
        delta_p = 100 * delta / self.forward_func.weight.data
        delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)
        
        if stats:
            print_stats(error.cpu().detach().numpy(), "Error  ")
            print_stats(delta.cpu().detach().numpy(), "Delta  ")
            print_stats(delta_p.cpu().detach().numpy(), "Delta %")
            print_stats(weight_decay_term.cpu().detach().numpy(), "WD Term")
            # print(error.shape, mean_dw.shape)
                          
        if self.writer_batch:
            self.tf_writer.add_histogram(f"Training_batch/Delta_Hidden", delta, self.n_batch)
        
        self.forward_func.weight.data = self.forward_func.weight.data - delta - weight_decay_term

        return error, (delta_p, delta)

    def test_reset_state(self, batch_size, device):
        """
        At start of testing, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (device): device

        Returns:
            forward_state: forward neuron states

        """
        # Forward neuron states
        # volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        # spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        if self.hidden_layer:
            volt = torch.zeros([batch_size, self.n_filters, self.output_dim[0], self.output_dim[1]], device=device)  # soma voltage
            spike = torch.zeros([batch_size, self.n_filters, self.output_dim[0], self.output_dim[1]], device=device)  # soma spike
        else:
            volt = torch.zeros([batch_size, self.output_dim], device=device)
            spike = torch.zeros([batch_size, self.output_dim], device=device)
        forward_state = (volt, spike)

        return forward_state

    def test_forward_step(self, spike_input, forward_state, stats=False):
        """
        One step forward connection simulation for the neuron (test only)

        Args:
            spike_input (Tensor): spike input from pre-synaptic input
            forward_state (tuple): forward neuron states

        Returns:
            spike_output: spike output to downstream layer
            forward_state: updated forward neuron states

        """
        volt, spike = forward_state

        # Update LIF neuron
        if stats:
            print_stats(volt.cpu().detach().numpy(), "Volt pre   ")

        forward_act = self.forward_func_quant(spike_input)
        if not self.float_mode:
            forward_act, exp = scale_weight(forward_act, self.act_bits)
        
        volt = self.vdecay * volt * (1. - spike) + forward_act
        if not self.float_mode:
            volt = self.rounding_function(volt)
        spike_output = volt.gt(self.vth).float()
        activation = 100 * torch.sum(spike_output).item() / spike_output.numel()

        if stats:
            print_stats(forward_act.cpu().detach().numpy(), "Forward Act")
            print_stats(volt.cpu().detach().numpy(),        "Volt post  ")
            print_stats(spike_output.cpu().detach().numpy(), "Spike      ")
            print("Activation %: ", activation)

        return spike_output, (volt, spike_output), activation

    def sleep_forward_step(self, spike_input_pos, spike_input_neg, forward_state):
        """
        One step forward connection simulation for sleep phase of the neuron

        Args:
            spike_input_pos (Tensor): positive Poisson spike input
            spike_input_neg (Tensor): negative Poisson spike input
            forward_state (tuple): forward neuron states

        Returns:
            spike_output: spike output to downstream layer
            forward_state: updated forward neuron states

        """
        volt, spike = forward_state

        # Update LIF neuron
        volt = self.vdecay * volt * (1. - spike) + self.forward_func(spike_input_pos) - self.forward_func(spike_input_neg)
        spike_output = volt.gt(self.vth).float()

        return spike_output, (volt, spike_output)


class OnlineOutputCell(OnlineHiddenCell):
    """ Online Fully-Connected Spiking Neuron Cell for Output Layer (including error interneurons) """

    def __init__(self, forward_func, neuron_param, input_dim, output_dim, float_mode=False):
        """

        Args:
            forward_func (Torch Function): Pre-synaptic function for forward
            neuron_param (tuple): LIF neuron and feedback parameters
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        self.forward_func = forward_func
        self.feedback_func = nn.Identity()
        self.vdecay, self.vth, self.grad_win, self.grad_amp, self.feedback_th = neuron_param['Vdecay'], neuron_param['Vth'], neuron_param['Grad_win'], neuron_param['Grad_amp'], neuron_param['Fb_th']
        self.lr, self.weight_decay = neuron_param['lr'], neuron_param['Weight_decay']
        self.vth, self.grad_win = torch.tensor(self.vth), torch.tensor(self.grad_win)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.float_mode = float_mode
        self.hidden_layer = False
        self.rounding_function = torch.floor
        # self.rounding_function = stochastic_rounding

    def train_reset_state(self, batch_size, device):
        """
        At start of training, reset all states within the neuron

        Args:
            batch_size (int): batch size
            device (device): device

        Returns:
            forward_state: forward neuron states
            feedback_state: feedback neuron states

        """
        # Forward neuron states
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        trace_pre = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # pre-spike trace
        trace_dw = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # gradient trace for weight
        trace_bias = torch.zeros([batch_size, self.output_dim], device=device)  # bias-spike trace (spike all step)
        trace_db = torch.zeros([batch_size, self.output_dim], device=device)  # gradient trace for bias
        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db)

        # Feedback neuron states
        error_pos_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error pos neuron volt
        error_neg_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error neg neuron volt
        error_pos_spike = torch.zeros([batch_size, self.output_dim], device=device)  # error pos neuron spike
        error_neg_spike = torch.zeros([batch_size, self.output_dim], device=device)  # error neg neuron spike
        error_dendrite_volt = torch.zeros([batch_size, self.output_dim], device=device)  # error dendrite volt
        feedback_state = (error_pos_volt, error_neg_volt, error_pos_spike, error_neg_spike, error_dendrite_volt)

        return forward_state, feedback_state

    def quantize_output_cell(self, num_bits, stats, plot, max_val=0):

        # Quantize forward weights
        if stats:
            print("Forward Function")
        self.forward_func.weight.data, step_fw, exp_scale_fw = quantize_layer_weights(
            self.forward_func.weight.data, num_bits, stats=stats, plot=plot, max_val=max_val)
        self.w_fw_exp = exp_scale_fw

    def train_feedback_step(self, pos_input, neg_input, feedback_state, stats=False):
        """
        One step feedback simulation for the neuron

        Args:
            pos_input (Tensor): current input from error computation
            neg_input (Tensor): current input from error computation
            feedback_state (tuple): feedback neuron states

        Returns:
            pos_spike_output: spike output to upstream positive error neuron
            neg_spike_output: spike output to upstream negative error neuron
            feedback_state: updated feedback neuron states

        """
        error_pos_volt, error_neg_volt, error_pos_spike, error_neg_spike, error_dendrite_volt = feedback_state

        # Update error neurons (IF neurons with soft reset)
        error_neuron_psp = pos_input - neg_input
        if stats:
            print_stats(error_neuron_psp.cpu().detach().numpy(), "Error Neuron PSP   ")
        # error_neuron_psp = torch.clamp(error_neuron_psp, -2**(self.n_bits-1), 2**(self.n_bits-1) - 1)

        error_pos_volt = error_pos_volt - error_pos_spike + error_neuron_psp
        pos_spike_output = error_pos_volt.gt(self.feedback_th).float()

        error_neg_volt = error_neg_volt - error_neg_spike - error_neuron_psp
        neg_spike_output = error_neg_volt.gt(self.feedback_th).float()

        # Update error dendrite
        error_dendrite_volt = error_dendrite_volt + (pos_spike_output - neg_spike_output)
        
        pos_activation = 100 * torch.sum(pos_spike_output).item() / (error_pos_volt.numel())
        neg_activation = 100 * torch.sum(neg_spike_output).item() / (error_neg_volt.numel())
        activations = torch.tensor([pos_activation, neg_activation])

        if stats:
            print_stats(error_pos_volt.cpu().detach().numpy(),      "Error Pos Volt     ")
            print_stats(error_neg_volt.cpu().detach().numpy(),      "Error Neg Volt     ")
            print_stats(error_dendrite_volt.cpu().detach().numpy(), "Error Dendrite Volt")
            print("Pos Activation %: ", pos_activation)
            print("Neg Activation %: ", neg_activation)

        return pos_spike_output, neg_spike_output, (
        error_pos_volt, error_neg_volt, pos_spike_output, neg_spike_output, error_dendrite_volt), activations
    
    def train_update_parameter_sgd(self, update_state, stats=False):
        """
        Update parameter using vanilla SGD

        Args:
            update_state (tuple): neuron states used for update parameter
            lr (float): learning rate

        Returns:
            error: estimated error for hidden neurons by direct feedback connection

        """
        error_dendrite_volt, error_steps, trace_dw, trace_db = update_state
        if self.float_mode:
            error = error_dendrite_volt / error_steps
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw # (batch, 10, n_filters * 14 * 14)
            mean_dw = mean_dw.sum(0) # (10, n_filters * 14 * 14)
            delta = self.lr * mean_dw
        else:
            error = error_dendrite_volt // error_steps
            # error = truncated_division(error_dendrite_volt, error_steps)
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw # (batch, 10, n_filters * 14 * 14)
            mean_dw = mean_dw.sum(0) # (10, n_filters * 14 * 14)
            mean_dw = mean_dw * self.lr
            delta = self.rounding_function(mean_dw)
        
        # Weight decay term
        if not self.float_mode:
            weight_decay_term = self.forward_func.weight.data * self.weight_decay
            weight_decay_term = self.rounding_function(weight_decay_term)
        else:
            weight_decay_term = self.weight_decay * self.forward_func.weight.data * self.forward_func.weight.data
            weight_decay_term = weight_decay_term
                
        delta_p = 100 * delta / self.forward_func.weight.data
        delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)
        
        if stats:
            print_stats(error.cpu().detach().numpy(), "Error  ")
            print_stats(delta.cpu().detach().numpy(), "Delta  ")
            print_stats(delta_p.cpu().detach().numpy(), "Delta %")
            print_stats(weight_decay_term.cpu().detach().numpy(), "WD Term")
            # print(error.shape, mean_dw.shape)
                          
        self.forward_func.weight.data = self.forward_func.weight.data - delta - weight_decay_term
        return error, (delta_p, delta)