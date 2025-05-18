import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from quantization import print_stats, quantize_layer_weights_bias, quantize_layer_weights, truncated_division, stochastic_rounding, scale_weight
import copy

class OnlineHiddenCell:
    """ Online Fully-Connected Spiking Neuron Cell for Hidden Layers """

    def __init__(self, forward_func, recurrent_func, feedback_func, neuron_param, input_dim, output_dim, float_mode=False):
        """a

        Args:
            forward_func (Torch Function): Pre-synaptic function for forward connection
            feedback_func (Torch Function): Feedback function for feedback connection
            neuron_param (tuple): LIF neuron parameters
            input_dim (int): input dimension
            output_dim (int): output dimension
        """
        self.forward_func = forward_func
        self.recurrent_func = recurrent_func
        self.feedback_func = feedback_func
        self.param_dict = neuron_param
        self.vdecay, self.vth, self.grad_win, self.grad_amp = neuron_param['Vdecay'], neuron_param['Vth'], neuron_param['Grad_win'], neuron_param['Grad_amp']
        self.lr, self.weight_decay, self.lr_rec, self.vdecay_rec = neuron_param['lr'], neuron_param['Weight_decay'], neuron_param['lr_rec'], neuron_param['Vdecay_rec']
        self.vth, self.grad_win = torch.tensor(self.vth), torch.tensor(self.grad_win)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.float_mode = float_mode
        self.hidden = True
        self.fw_factor, self.rec_factor = torch.tensor(neuron_param['fw_factor']), torch.tensor(neuron_param['rec_factor'])
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
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
        trace_pre = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # pre-spike trace
        trace_dw = torch.zeros([batch_size, self.output_dim, self.input_dim], device=device)  # gradient trace for weight
        trace_bias = torch.zeros([batch_size, self.output_dim], device=device)  # bias-sspike trace (spike all step)
        trace_db = torch.zeros([batch_size, self.output_dim], device=device)  # gradient trace for bias

        # Recurrent neuron states
        trace_pre_rec = torch.zeros([batch_size, self.output_dim], device=device)
        trace_dw_rec = torch.zeros([batch_size, self.output_dim], device=device)
        trace_bias_rec = torch.zeros([batch_size, self.output_dim], device=device)
        trace_db_rec = torch.zeros([batch_size, self.output_dim], device=device)

        # Feedback neuron states
        feedback_state = torch.zeros([batch_size, self.output_dim], device=device)  # error dendrite volt

        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db, trace_pre_rec, trace_dw_rec, trace_bias_rec, trace_db_rec)
        
        return forward_state, feedback_state

    def quantize(self, num_bits, stats, plot, max_val=0, num_bits_feedback=0):

        # Quantize forward and feedback weights
        if stats:
            print("Forward Function")
        self.forward_func.weight.data, step_fw, exp_scale_fw = quantize_layer_weights(
            self.forward_func.weight.data, num_bits, stats=stats, plot=plot, max_val=max_val)
        if stats:
            print("Recurrent Function")
        self.recurrent_func.weight.data, step_fw, exp_scale_fw = quantize_layer_weights(
            self.recurrent_func.weight.data, num_bits, stats=stats, plot=plot, max_val=max_val)
        if stats:
            print("Feedback Function")
        num_bits_fb = num_bits if num_bits_feedback == 0 else num_bits_feedback
        self.feedback_func.weight.data, step_fb, exp_scale_fb = quantize_layer_weights(
            self.feedback_func.weight.data, num_bits_fb, stats=stats, plot=plot, max_val=max_val)
        
        self.w_fw_exp = exp_scale_fw
        self.w_fb_exp = exp_scale_fb

    def quantize_bias(self, num_bits, stats, plot, max_val=0, num_bits_feedback=0):

        # Quantize forward and feedback weights
        if stats:
            print("Forward Function")
        self.forward_func.weight.data, self.forward_func.bias.data, step_fw, exp_scale_fw = quantize_layer_weights_bias(
            self.forward_func.weight.data, self.forward_func.bias.data, num_bits, stats=stats, plot=plot, max_val=max_val)
        if stats:
            print("Recurrent Function")
        self.recurrent_func.weight.data, self.recurrent_func.bias.data, step_fw, exp_scale_fw = quantize_layer_weights_bias(
            self.recurrent_func.weight.data, self.recurrent_func.bias.data, num_bits, stats=stats, plot=plot, max_val=max_val)
        if stats:
            print("Feedback Function")
        num_bits_fb = num_bits if num_bits_feedback == 0 else num_bits_feedback
        self.feedback_func.weight.data, step_fb, exp_scale_fb = quantize_layer_weights(
            self.feedback_func.weight.data, num_bits_fb, stats=stats, plot=plot, max_val=max_val)
        
        self.w_fw_exp = exp_scale_fw
        self.w_fb_exp = exp_scale_fb

    def update_volt(self, volt, spike_input, spike, stats=False):
        """
        Updates the voltage of the neuron
        """

        if stats:
            print_stats(volt.cpu().detach().numpy(),                "Volt pre        ")
        
        forward_act = self.forward_func_quant(spike_input) * self.fw_factor
        if self.hidden:
            pre_quant = torch.floor(volt * self.rec_factor) if not self.float_mode else volt
            recurrent_act = self.recurrent_func_quant(volt)

            if not self.float_mode:
                forward_act, exp = scale_weight(forward_act, self.act_bits)
                recurrent_act, exp_rec = scale_weight(recurrent_act, self.act_bits)
                # forward_act = self.rounding_function(forward_act)
                # recurrent_act = self.rounding_function(recurrent_act)
                # forward_act = torch.clamp(forward_act, -2**(self.n_bits-1), 2**(self.n_bits-1) - 1)
                # recurrent_act = torch.clamp(recurrent_act, -2**16, 2**16)
        else:
            recurrent_act = torch.tensor(0.)
        
        volt_pre = volt.clone()

        volt = self.vdecay * volt * (1. - spike) + forward_act + recurrent_act
        if not self.float_mode:
            volt = self.rounding_function(volt)
            # if not self.hidden:
            #     volt = torch.clamp(volt, -500, 2**(self.n_bits-1) - 1)
            
        if self.writer_spike:
            self.tf_writer.add_histogram(f"activations_volts/{self.tf_name}_Volt", volt, self.spike_time)
            self.tf_writer.add_histogram(f"{self.tf_name}/Forward_Act", forward_act, self.spike_time)
            if self.hidden:
                self.tf_writer.add_histogram(f"{self.tf_name}/Recurrent_Act", recurrent_act, self.spike_time)
            
        if stats:
            print_stats(forward_act.cpu().detach().numpy(),         "Forward Act     ")
            if self.hidden:
                print_stats(recurrent_act.cpu().detach().numpy(),   "Recurrent Act   ")
            print_stats(volt.cpu().detach().numpy(),                "Volt post       ")
        
        return volt_pre, volt
    
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
        volt, spike, trace_pre, trace_dw, trace_bias, trace_db, trace_pre_rec, trace_dw_rec, trace_bias_rec, trace_db_rec = forward_state

        # Update neuron soma (LIF neuron)
        volt_pre, volt = self.update_volt(volt, spike_input, spike, stats=stats)
            
        spike_output = volt.gt(self.vth).float()
        max_volt = torch.max(torch.abs(volt))

        # Update hidden traces
        volt_pseudo_grad = (abs(volt - self.vth) < self.grad_win).float() * self.grad_amp

        if self.hidden:
            trace_pre = self.vdecay * trace_pre + spike_input.view(-1, 1, self.input_dim)
        else:
            trace_pre = self.vdecay * trace_pre + spike_input.view(-1, 1, self.input_dim)
        if not self.float_mode:
            trace_pre = self.rounding_function(trace_pre)
        trace_dw = trace_dw + trace_pre * volt_pseudo_grad.view(-1, self.output_dim, 1)
        # trace_pre = trace_pre * (1 - spike_output - volt * volt_pseudo_grad).view(-1, self.output_dim, 1)
        # trace_pre = trace_pre * 2**-16

        if self.bias:
            trace_bias = self.vdecay * trace_bias + 1.
            trace_db = trace_db + trace_bias * volt_pseudo_grad
            # trace_bias = trace_bias * (1 - spike_output - volt * volt_pseudo_grad)

        # Update recurrent traces
        if self.hidden:
            volt_pre, _ = scale_weight(volt.clone(), 12)
            trace_pre_rec = self.vdecay_rec * trace_pre_rec + volt_pre
            if not self.float_mode:
                # trace_pre_rec, exp = scale_weight(trace_pre_rec, self.n_bits)
                trace_pre_rec = self.rounding_function(trace_pre_rec)
                trace_pre_rec = torch.clamp(trace_pre_rec, -2**(self.n_bits-1), 2**(self.n_bits-1) - 1)
            
            trace_dw_rec = trace_dw_rec + trace_pre_rec * volt_pseudo_grad
            # if not self.float_mode:
            #     trace_dw_rec, exp = scale_weight(trace_dw_rec, self.n_bits)

            # trace_pre_rec = trace_pre_rec * (1 - spike_output - volt * volt_pseudo_grad)
            # trace_pre_rec = trace_pre_rec * 2**-16
            if self.bias:
                trace_bias_rec = self.vdecay_rec * trace_bias_rec + 1.
                trace_db_rec = trace_db_rec + trace_bias_rec * volt_pseudo_grad

        activation = 100 * torch.sum(spike_output).item() / spike_output.numel()
        
        if self.writer_spike:
            self.tf_writer.add_histogram(f"{self.tf_name}/input", spike_input, self.spike_time)
            self.tf_writer.add_scalar(f"activations_volts/{self.tf_name}_Activation", activation, self.spike_time)
            self.tf_writer.add_histogram(f"{self.tf_name}/Trace_Pre", trace_pre, self.spike_time)
            self.tf_writer.add_histogram(f"{self.tf_name}/Trace_Dw", trace_dw, self.spike_time)
            if self.hidden:
                self.tf_writer.add_histogram(f"{self.tf_name}/Trace_Pre_Rec", trace_pre_rec, self.spike_time)
                self.tf_writer.add_histogram(f"{self.tf_name}/Trace_Dw_Rec", trace_dw_rec, self.spike_time)

        if stats:
            print("Activation %:   ", activation)
            print_stats(volt_pseudo_grad.cpu().detach().numpy(), "Volt Pseudo Grad")
            print_stats(trace_pre.cpu().detach().numpy(),        "Trace Pre       ")
            print_stats(trace_dw.cpu().detach().numpy(),         "Trace Dw        ")
            if self.bias:
                print_stats(trace_bias.cpu().detach().numpy(),       "Trace Bias      ")
                print_stats(trace_db.cpu().detach().numpy(),         "Trace Db        ")
            if self.hidden:
                print_stats(trace_pre_rec.cpu().detach().numpy(), "Trace Pre - Rec")
                print_stats(trace_dw_rec.cpu().detach().numpy(),  "Trace Dw - Rec ")
                if self.bias:
                    print_stats(trace_bias_rec.cpu().detach().numpy(), "Trace Bias - Rec")
                    print_stats(trace_db_rec.cpu().detach().numpy(),  "Trace Db - Rec ")
            # print(spike_input.shape, volt.shape, spike_output.shape, trace_pre.shape, trace_dw.shape)

        return spike_output, (volt, spike_output, trace_pre, trace_dw, trace_bias, trace_db, trace_pre_rec, trace_dw_rec, trace_bias_rec, trace_db_rec), max_volt, activation
        
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
            error_dendrite_volt = feedback_state + self.feedback_func_quant(pos_spike_input)
        else:
            error_dendrite_volt = feedback_state + (self.feedback_func(pos_spike_input) - self.feedback_func(neg_spike_input))

        # if not self.float_mode:
            # error_dendrite_volt, exp = scale_weight(error_dendrite_volt, self.n_bits)
            # error_dendrite_volt = torch.clamp(error_dendrite_volt, -2**(self.n_bits-1), 2**(self.n_bits-1) - 1)
        
        if stats:
            print_stats(error_dendrite_volt.cpu().detach().numpy(), "Error Dendrite Volt")
        if self.writer_batch:
            self.tf_writer.add_histogram(f"Training_batch/Error_Dendrite_Volt", error_dendrite_volt, self.n_batch)

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
        error_dendrite_volt, error_steps, trace_dw, trace_db, trace_dw_rec, trace_db_rec = update_state
        if self.float_mode:
            error = error_dendrite_volt / error_steps
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw
            mean_dw = mean_dw.sum(0) / mean_dw.size(0)
            delta = self.lr * mean_dw
            
            mean_dw_rec = error * error
            mean_dw_rec = mean_dw_rec.sum(0) / mean_dw_rec.size(0)
            delta_rec = self.lr_rec * mean_dw_rec

            if self.bias:
                mean_db = error.view(-1, self.output_dim) * trace_db
                mean_db = mean_db.sum(0) / mean_db.size(0)
                delta_bias = self.lr * mean_db
                
                mean_db_rec = error * trace_db_rec
                mean_db_rec = mean_db_rec.sum(0) / mean_db_rec.size(0)
                delta_bias_rec = self.lr_rec * mean_db_rec

        else:
            error = error_dendrite_volt // error_steps
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw
            mean_dw = mean_dw.sum(0)
            # delta = scale_weight(mean_dw, 16)
            mean_dw = mean_dw * self.lr
            mean_dw = self.rounding_function(mean_dw)
            # mean_dw = mean_dw.sum(0)
            if self.gradient_clip > 0:
                mean_dw = torch.clamp(mean_dw, -self.gradient_clip, self.gradient_clip)
            delta = mean_dw

            mean_dw_rec = error * error
            mean_dw_rec = mean_dw_rec.sum(0)
            # delta_rec = scale_weight(mean_dw_rec, 16)
            mean_dw_rec = mean_dw_rec * self.lr_rec
            mean_dw_rec = self.rounding_function(mean_dw_rec)
            # mean_dw_rec = mean_dw_rec.sum(0)
            if self.gradient_clip > 0:
                mean_dw_rec = torch.clamp(mean_dw_rec, -self.gradient_clip, self.gradient_clip)
            delta_rec = mean_dw_rec

            if self.bias: 
                mean_db = error.view(-1, self.output_dim) * trace_db
                mean_db = mean_db * self.lr
                mean_db = self.rounding_function(mean_db)
                delta_bias = mean_db.sum(0)

                mean_db_rec = error * trace_db_rec
                mean_db_rec = mean_db_rec * self.lr_rec
                mean_db_rec = self.rounding_function(mean_db_rec)
                delta_bias_rec = mean_db_rec.sum(0)
        
        # Weight decay term
        wd_fw = self.weight_decay * self.forward_func.weight.data
        wd_rec = self.weight_decay * self.recurrent_func.weight.data
        if self.bias:
            wd_fw_bias = self.weight_decay * self.forward_func.bias.data
            wd_rec_bias = self.weight_decay * self.recurrent_func.bias.data
        if not self.float_mode:
            wd_fw = self.rounding_function(wd_fw)
            wd_rec = self.rounding_function(wd_rec)
            if self.bias:
                wd_fw_bias = self.rounding_function(wd_fw_bias)
                wd_rec_bias = self.rounding_function(wd_rec_bias)

        delta_p = 100 * delta / self.forward_func.weight.data
        delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)

        delta_rec_p = 100 * delta_rec / self.recurrent_func.weight.data
        delta_rec_p = torch.nan_to_num(delta_rec_p, nan=0.0, posinf=0.0, neginf=0.0)

        if self.bias:
            delta_bias_p = 100 * delta_bias / self.forward_func.bias.data
            delta_bias_p = torch.nan_to_num(delta_bias_p, nan=0.0, posinf=0.0, neginf=0.0)

            delta_bias_rec_p = 100 * delta_bias_rec / self.recurrent_func.bias.data
            delta_bias_rec_p = torch.nan_to_num(delta_bias_rec_p, nan=0.0, posinf=0.0, neginf=0.0)
        
        if stats:
            print_stats(error.cpu().detach().numpy(),       "Error  ")
            print_stats(delta.cpu().detach().numpy(),       "Delta  ")
            print_stats(delta_p.cpu().detach().numpy(),     "Delta %")
            print_stats(delta_rec.cpu().detach().numpy(),   "Delta Rec  ")
            print_stats(delta_rec_p.cpu().detach().numpy(), "Delta Rec %")
            if self.bias:
                print_stats(delta_bias.cpu().detach().numpy(),      "Delta Bias      ")
                print_stats(delta_bias_p.cpu().detach().numpy(),    "Delta Bias %    ")
                print_stats(delta_bias_rec.cpu().detach().numpy(),  "Delta Bias Rec  ")
                print_stats(delta_bias_rec_p.cpu().detach().numpy(),"Delta Bias Rec %")

            print_stats(wd_fw.cpu().detach().numpy(),       "WD Term FW ")
            print_stats(wd_rec.cpu().detach().numpy(),      "WD Term Rec")
        
        if self.writer_batch:
            self.tf_writer.add_histogram(f"Training_batch/Delta_Hidden", delta, self.n_batch)
            self.tf_writer.add_histogram(f"Training_batch/Delta_Rec", delta_rec, self.n_batch)

        self.forward_func.weight.data = self.forward_func.weight.data - delta - wd_fw
        self.recurrent_func.weight.data = self.recurrent_func.weight.data - delta_rec - wd_rec
        if self.bias:
            self.forward_func.bias.data = self.forward_func.bias.data - delta_bias
            self.recurrent_func.bias.data = self.recurrent_func.bias.data - delta_bias_rec

        return error, (delta_p, delta, delta_rec_p, delta_rec)

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
        volt = torch.zeros([batch_size, self.output_dim], device=device)  # soma voltage
        spike = torch.zeros([batch_size, self.output_dim], device=device)  # soma spike
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
        
        _, volt = self.update_volt(volt, spike_input, spike, stats=stats)

        spike_output = volt.gt(self.vth).float()
        activation = 100 * torch.sum(spike_output).item() / spike_output.numel()

        if stats:
            print_stats(spike_output.cpu().detach().numpy(),  "Spike        ")
            print("Activation %: ", activation)
            # print(spike_input.shape, volt.shape, spike_output.shape)

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
        self.vdecay, self.vth, self.grad_win, self.grad_amp = neuron_param['Vdecay'], neuron_param['Vth'], neuron_param['Grad_win'], neuron_param['Grad_amp']
        self.lr, self.weight_decay, self.feedback_th = neuron_param['lr'], neuron_param['Weight_decay'], neuron_param['Fb_th']
        self.vth, self.grad_win = torch.tensor(self.vth), torch.tensor(self.grad_win)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.float_mode = float_mode
        self.hidden = False
        self.fw_factor, self.rec_factor = 1., 0.
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

        forward_state = (volt, spike, trace_pre, trace_dw, trace_bias, trace_db, 0, 0, 0, 0)

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
        if self.bias:
            self.forward_func.weight.data, self.forward_func.bias.data, step_fw, exp_scale_fw = quantize_layer_weights_bias(self.forward_func.weight.data, 
                                                                                               self.forward_func.bias.data, 
                                                                                               num_bits, stats=stats, plot=plot, max_val=max_val)
        else:
            self.forward_func.weight.data, step_fw, exp_scale_fw = quantize_layer_weights(self.forward_func.weight.data, 
                                                                                               num_bits, stats=stats, plot=plot, max_val=max_val)
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
        
        pos_activation = 100 * torch.sum(pos_spike_output).item() / pos_spike_output.numel()
        neg_activation = 100 * torch.sum(neg_spike_output).item() / neg_spike_output.numel()
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
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw
            mean_dw = mean_dw.sum(0) / mean_dw.size(0)
            delta =  self.lr * mean_dw

            if self.bias:
                mean_db = error.view(-1, self.output_dim) * trace_db
                mean_db = mean_db.sum(0) / mean_db.size(0)
                delta_bias = self.lr * mean_db
        else:
            error = error_dendrite_volt // error_steps
            mean_dw = error.view(-1, self.output_dim, 1) * trace_dw
            mean_dw = mean_dw.sum(0)
            # delta = scale_weight(mean_dw, 16)
            mean_dw = mean_dw * self.lr
            mean_dw = self.rounding_function(mean_dw)
            # mean_dw = mean_dw.sum(0)
            if self.gradient_clip > 0:
                mean_dw = torch.clamp(mean_dw, -self.gradient_clip, self.gradient_clip)
            delta = mean_dw

            if self.bias:
                mean_db = error.view(-1, self.output_dim) * trace_db
                mean_db = mean_db * self.lr
                mean_db = self.rounding_function(mean_db)
                delta_bias = mean_db.sum(0)
        
        # Weight decay term
        weight_decay_term = self.weight_decay * self.forward_func.weight.data
        if self.bias:
            weight_decay_term_bias = self.weight_decay * self.forward_func.bias.data
        if not self.float_mode:
            weight_decay_term = self.rounding_function(weight_decay_term)
            if self.bias:
                weight_decay_term_bias = self.rounding_function(weight_decay_term_bias)
        
        delta_p = 100 * delta / self.forward_func.weight.data
        delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)
        if self.bias:
            delta_bias_p = 100 * delta_bias / self.forward_func.bias.data
            delta_bias_p = torch.nan_to_num(delta_bias_p, nan=0.0, posinf=0.0, neginf=0.0)
        
        if stats:
            print_stats(error.cpu().detach().numpy(), "Error  ")
            print_stats(delta.cpu().detach().numpy(), "Delta  ")
            print_stats(delta_p.cpu().detach().numpy(), "Delta %")
            print_stats(weight_decay_term.cpu().detach().numpy(), "WD Term")
            if self.bias:
                print_stats(delta_bias.cpu().detach().numpy(), "Delta Bias  ")
                print_stats(delta_bias_p.cpu().detach().numpy(), "Delta Bias %")
                print_stats(weight_decay_term_bias.cpu().detach().numpy(), "WD Term Bias")
        
        if self.writer_batch:
            self.tf_writer.add_histogram(f"Training_batch/Delta_Output", delta, self.n_batch)
        
        self.forward_func.weight.data = self.forward_func.weight.data - delta - weight_decay_term
        if self.bias:
            self.forward_func.bias.data = self.forward_func.bias.data - delta_bias - weight_decay_term_bias

        # plt.hist(delta.cpu().detach().numpy().flatten(), bins=100)
        # plt.title("Delta hist")
        # plt.savefig("dynamic_debug/delta_hist.png")
        # plt.close()

        return error, (delta_p, delta)
