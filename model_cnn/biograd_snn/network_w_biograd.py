import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import copy
import math
from biograd_snn.network_cells import OnlineHiddenCell, OnlineOutputCell
from quantization import print_stats
from biograd_snn.online_error_functions import cross_entropy_loss_error_function
from quantization import print_stats, truncated_division, scale_weight, scale_weight_bias


class BioGradNetworkWithSleep:
    """ Online Learning Network with Sleep Weight Mirror Feedback Learning """

    def __init__(self, input_dim, output_dim, param_dict, error_func, device, 
                 float_mode=False, aligned_weights=True, loss_mode='feedback', terminal_stats=True,
                 num_bits=16, weight_init_bits=16, low_precision_bits=8, activation_bits=16,
                 bias=False, gradient_clip=0):
        """

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim_list (list): list of hidden layer dimension
            param_dict (dict): neuron parameter dictionary
            error_func (function): error function
            device (device): device
        """
        self.terminal_stats = terminal_stats
        self.hidden_cells = []
        self.n_layers = len(param_dict['hidden_layer']) + 1
        # Init Hidden Layers
        for idx, parameters in enumerate(param_dict['hidden_layer']):
            if idx == 0:
                forward_input_dim = input_dim # (28, 28)
                channels_in = 1
            else:
                forward_input_dim = forward_output_dim # (14, 14), (7, 7) ...
                channels_in = n_filters

            n_filters = parameters['n_filters']
            kernel_size = parameters['kernel_size']

            if parameters['reduce_dim']: # Reduce input size by half
                stride = 2
                padding = (kernel_size - 1) // 2
                forward_output_dim = int(forward_input_dim[0]/2), int(forward_input_dim[1]/2)
            else:
                stride = 1
                padding = 0
                forward_output_dim = forward_input_dim[0], forward_input_dim[1]
            
            if idx == self.n_layers - 2: # Last hidden layer
                feedback_func = nn.Linear(output_dim, n_filters * forward_output_dim[0] * forward_output_dim[1], bias=False).to(device)
            else:
                feedback_func = nn.ConvTranspose2d(n_filters, channels_in, kernel_size=kernel_size, stride=2, padding=1, output_padding=1, bias=False).to(device)

            self.hidden_cells.append(
                OnlineHiddenCell(
                    nn.Conv2d(channels_in, n_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False).to(device),
                    feedback_func,
                    parameters, forward_input_dim, forward_output_dim, float_mode=float_mode))
            self.hidden_cells[idx].channels_in = channels_in
            self.hidden_cells[idx].stride = stride
            self.hidden_cells[idx].padding = padding
            self.hidden_cells[idx].bias = bias
            self.hidden_cells[idx].name =  f"Hidden Layer {idx}"
            self.hidden_cells[idx].gradient_clip = gradient_clip
            self.hidden_cells[idx].act_bits = activation_bits
        
        # Init Output Layer
        input_dim_output = n_filters * forward_output_dim[0] * forward_output_dim[1]
        self.output_cell = OnlineOutputCell(
            nn.Linear(input_dim_output, output_dim, bias=False).to(device),
            param_dict['out_layer'], input_dim_output, output_dim, float_mode=float_mode)
        self.output_cell.bias = bias
        self.output_cell.name = "Output Layer"
        self.output_cell.gradient_clip = gradient_clip
        self.output_cell.act_bits = activation_bits

        # Init Feedback Connections
        if aligned_weights == True:
            if self.terminal_stats:
                print("Feedback Weights Aligned")
            feedback_weight = copy.deepcopy(self.output_cell.forward_func.weight.data)
            for idx in reversed(range(len(self.hidden_cells))):
                self.hidden_cells[idx].loss_mode = loss_mode
                self.hidden_cells[idx].feedback_func.weight.data = copy.deepcopy(feedback_weight.t())
                if idx > 0:
                    feedback_weight = torch.matmul(feedback_weight, self.hidden_cells[idx].forward_func.weight.data)
        else:
            if self.terminal_stats:
                print("Feedback Weights Transposed")
            for idx, cell in enumerate(self.hidden_cells):
                cell.loss_mode = loss_mode
                if idx == self.n_layers - 2: # Last hidden layer
                    cell.feedback_func.weight.data = copy.deepcopy(self.output_cell.forward_func.weight.data.t())
                else:
                    cell.feedback_func.weight.data = copy.deepcopy(self.hidden_cells[idx+1].forward_func.weight.data)

        self.float_mode = float_mode
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.error_func = error_func
        self.device = device
        self.cos_func = nn.CosineSimilarity(dim=0)
        self.loss_mode = loss_mode
        self.aligned_weights = aligned_weights
        self.n_bits = num_bits
        self.weight_init_bits = weight_init_bits
        self.low_precision_bits = low_precision_bits
        self.activation_bits = activation_bits
        self.gradient_clip = gradient_clip

        return

    def quantize_network(self, num_bits=0, weight_init_bits=0, stats=False, plot=False):
        """
        Quantize network weight and bias

        Args:
            num_bits (int): number of bits for quantization

        """
        if num_bits == 0:
            num_bits = self.n_bits
        if weight_init_bits == 0:
            weight_init_bits = self.weight_init_bits
            
        forward_exponents = [[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]]

        max_val = 0
        for i, cell in enumerate(self.hidden_cells):
            max_val = max(max_val, torch.max(torch.abs(cell.forward_func.weight.data)))
            max_val = max(max_val, torch.max(torch.abs(cell.feedback_func.weight.data)))
        max_val = max(max_val, torch.max(torch.abs(self.output_cell.forward_func.weight.data)))

        num_bits_feedback = weight_init_bits if not self.aligned_weights else weight_init_bits - 2
        for i, cell in enumerate(self.hidden_cells):
            if stats:
                print("\n Hidden Layer ", i)
            cell.quantize(weight_init_bits, stats=stats, plot=plot, max_val=max_val, num_bits_feedback=num_bits_feedback)
            cell.n_bits = num_bits
            cell.forward_exp = forward_exponents[i]
        if stats:
            print("\n Output layer")
        self.output_cell.quantize_output_cell(weight_init_bits, stats=stats, plot=plot, max_val=max_val)
        self.output_cell.n_bits = num_bits
        self.output_cell.forward_exp = forward_exponents[-1]

        if self.terminal_stats:
            print("Network Quantized to ", num_bits, " bits")
            print("Weights Initialized to ", weight_init_bits, " bits")
            print("Feedback Weights Initialized to ", num_bits_feedback, " bits")
            # w_exp_fw = self.hidden_cells[0].w_fw_exp, self.hidden_cells[1].w_fw_exp, self.output_cell.w_fw_exp
            # w_exp_fb = self.hidden_cells[0].w_fb_exp, self.hidden_cells[1].w_fb_exp
            # print("Weight Exponent FW: ", w_exp_fw)
            # print("Weight Exponent FB: ", w_exp_fb)
            print("Forward Exponents: ", forward_exponents)

    def prepare_inference_weights(self, stats=False):
        """
        Creates a quantized copy of the forward and feedback functions for inference

        """
        
        self.inference_exp = self.low_precision_bits - self.n_bits # TO CHECK: nbits or weight_init_bits?
        if stats:
            print("Exponent for inference weights: ", self.inference_exp)

        for cell in self.hidden_cells:
            cell.forward_func_quant = copy.deepcopy(cell.forward_func)
            cell.feedback_func_quant = copy.deepcopy(cell.feedback_func)

            if not self.float_mode:
            #     cell.forward_func_quant.weight.data = torch.floor(cell.forward_func_quant.weight.data * (2 ** self.inference_exp))
            #     cell.feedback_func_quant.weight.data = torch.floor(cell.feedback_func_quant.weight.data * (2 ** self.inference_exp))
                cell.forward_func_quant.weight.data = scale_weight(cell.forward_func_quant.weight.data, self.low_precision_bits)[0]
                cell.feedback_func_quant.weight.data = scale_weight(cell.feedback_func_quant.weight.data, self.low_precision_bits)[0]
                if self.bias:
                    cell.forward_func_quant.bias.data = torch.floor(cell.forward_func_quant.bias.data * (2 ** self.inference_exp))
        
        self.output_cell.forward_func_quant = copy.deepcopy(self.output_cell.forward_func)
        if not self.float_mode:
            # self.output_cell.forward_func_quant.weight.data = torch.floor(self.output_cell.forward_func_quant.weight.data * (2 ** self.inference_exp))
            self.output_cell.forward_func_quant.weight.data = scale_weight(self.output_cell.forward_func_quant.weight.data, self.low_precision_bits)[0]
            if self.bias:
                self.output_cell.forward_func_quant.bias.data = torch.floor(self.output_cell.forward_func_quant.bias.data * (2 ** self.inference_exp))

    def update_inference_weights(self):

        if self.float_mode:
            for cell in self.hidden_cells:
                cell.forward_func_quant.weight.data = cell.forward_func.weight.data
                cell.feedback_func_quant.weight.data = cell.feedback_func.weight.data
                if self.bias:
                    cell.forward_func_quant.bias.data = cell.forward_func.bias.data
            self.output_cell.forward_func_quant.weight.data = self.output_cell.forward_func.weight.data
            if self.bias:
                self.output_cell.forward_func_quant.bias.data = self.output_cell.forward_func.bias.data
            return
        
        for cell in self.hidden_cells:
            
            # cell.forward_func_quant.weight.data = torch.floor(cell.forward_func.weight.data * (2 ** self.inference_exp))
            # cell.feedback_func_quant.weight.data = torch.floor(cell.feedback_func.weight.data * (2 ** self.inference_exp))
            cell.forward_func_quant.weight.data = scale_weight(cell.forward_func.weight.data, self.low_precision_bits)[0]
            cell.feedback_func_quant.weight.data = scale_weight(cell.feedback_func.weight.data, self.low_precision_bits)[0]

            # Dynamic adaptation of the threshold and gradient window
            # w, exponent = scale_weight(cell.forward_func.weight.data, self.n_bits) # The shadow weights are quantized to n_bits
            # cell.forward_func_quant.weight.data = torch.floor(w * (2 ** self.inference_exp)) # The inference weights are quantized to low_precision_bits
            # if exponent > 0:
            #     cell.vth = torch.floor(cell.vth / (2 ** exponent))
            #     cell.grad_win = torch.floor(cell.grad_win / (2 ** exponent))

        # self.output_cell.forward_func_quant.weight.data =  torch.floor(self.output_cell.forward_func.weight.data * (2 ** self.inference_exp))
        self.output_cell.forward_func_quant.weight.data = scale_weight(self.output_cell.forward_func.weight.data, self.low_precision_bits)[0]
        
        # Dynamic adaptation of the threshold and gradient window
        # w_out, exponent_out = scale_weight(self.output_cell.forward_func.weight.data, self.n_bits)
        # self.output_cell.forward_func_quant.weight.data = torch.floor(w_out * (2 ** self.inference_exp))
        # if exponent_out > 0:
        #     self.output_cell.vth = torch.floor(self.output_cell.vth / (2 ** exponent_out))
        #     self.output_cell.grad_win = torch.floor(self.output_cell.grad_win / (2 ** exponent_out))
    
    def set_initial_lr(self):
        """
        Saves the initial learning rate in initiial_lr to be used for scaling the learning rate during inference
        """

        for cell in self.hidden_cells:
            cell.initial_lr = cell.lr
        
        self.output_cell.initial_lr = self.output_cell.lr

    def update_lr(self, lr_coef):
        """
        Update learning rate of the network

        Args:
            lr_coef (float): learning rate coefficient

        """
        for cell in self.hidden_cells:
            cell.lr = cell.initial_lr * lr_coef

        self.output_cell.lr = self.output_cell.initial_lr * lr_coef

    def train_online(self, spike_data, label_one_hot, soft_error_step, loss_precision=32, stats=False, float_mode=False):
        """
        Train SNN with online learning

        Args:
            spike_data (Tensor): spike data input (batch_size, input_dim, spike_ts)
            label_one_hot (Tensor): one hot vector of label
            soft_error_step (int): soft start step for error feedback

        Returns:
            predict_label: predict labels
            hidden_forward_states: list of hidden forward states
            hidden_feedback_states: list of hidden feedback states
            out_forward_state: output forward state
            out_feedback_state: output feedback state
            feedback_step: number of steps for feedback simulation

        """
        batch_size = spike_data.shape[0]
        spike_ts = spike_data.shape[-1]
        # if len(spike_data.shape) > 3:
        #     spike_data = spike_data.view(batch_size, self.input_dim, spike_ts)
        spike_data = spike_data.view(batch_size, self.input_dim[0], self.input_dim[1], spike_ts)

        # Init Hidden Layer Cell States
        hidden_forward_states, hidden_feedback_states = [], []
        for cell in self.hidden_cells:
            forward_state, feedback_state = cell.train_reset_state(batch_size, self.device)
            hidden_forward_states.append(forward_state)
            hidden_feedback_states.append(feedback_state)

        # Init Output Layer Cell State
        out_forward_state, out_feedback_state = self.output_cell.train_reset_state(batch_size, self.device)

        # Start online simulation of the network
        output = torch.zeros([batch_size, self.output_cell.output_dim], device=self.device)
        feedback_step = 0
        max_volts = torch.zeros(self.n_layers)
        activations = torch.zeros(self.n_layers)
        activations_fb = torch.zeros(2)
        for tt in range(spike_ts):
            if stats:
                show_stats = True if tt in [0, 10, spike_ts-1] else False
            else:
                show_stats = False
            input_spike = spike_data[:, :, :, tt].unsqueeze(1)
            volts = []
            if show_stats:
                print_stats(input_spike.cpu().detach().numpy(), "\n*** Input Spike " + str(tt))
            for idx, cell in enumerate(self.hidden_cells, 0):
                if show_stats:
                    print("\n Hidden Layer ", idx)
                cell.spike_time = tt
                input_spike, hidden_forward_states[idx], max_volt, activation = cell.train_forward_step(input_spike,
                                                                                  hidden_forward_states[idx], show_stats)
                max_volts[idx] = max(max_volts[idx], max_volt)
                activations[idx] += activation
            if show_stats:
                print("\n Output Layer")
            # Flatten input for fully connected output layer
            input_spike = input_spike.view(batch_size, -1)
            self.output_cell.spike_time = tt
            out_spike, out_forward_state, max_error_volt, activation = self.output_cell.train_forward_step(input_spike, out_forward_state, show_stats)
            max_volts[-1] = max(max_volts[-1], max_error_volt)
            activations[-1] += activation
            volts.append(out_forward_state[0])
            output = output + out_spike
            if show_stats:
                print_stats(output.cpu().detach().numpy(), "\n** Output")

            if self.loss_mode in ['direct', 'final']:
                continue

            # Start feedback simulation after a soft start
            if tt >= soft_error_step:
                cross_entropy_loss = cross_entropy_loss_error_function(output/tt, label_one_hot/loss_precision)
                error = self.error_func.loss(output, label_one_hot, loss_precision, float_mode=float_mode)
                error_pos = copy.deepcopy(error)
                error_pos[error_pos < 0] = 0
                error_neg = -copy.deepcopy(error)
                error_neg[error_neg < 0] = 0
                if show_stats:
                    print_stats(error.cpu().detach().numpy(), "** Error ")
                    print_stats(cross_entropy_loss.cpu().detach().numpy(), "** Cross ")
                    print("\n Feedback Simulation")
                    print("Output Layer")
                pos_spike, neg_spike, out_feedback_state, act = self.output_cell.train_feedback_step(
                    error_pos, error_neg, out_feedback_state, show_stats)
                for idx in reversed(range(len(self.hidden_cells))):
                    if show_stats:
                        print("\n Hidden Layer ", idx)
                    hidden_feedback_states[idx] = self.hidden_cells[idx].train_feedback_step(
                        pos_spike, neg_spike, hidden_feedback_states[idx], show_stats)
                feedback_step += 1
                activations_fb += act

        if self.loss_mode in ['direct', 'final']:
            error = self.error_func.loss(output, label_one_hot, loss_precision, float_mode=float_mode)
            cross_entropy_loss = cross_entropy_loss_error_function(output/spike_ts, label_one_hot/loss_precision)
            error_pos = copy.deepcopy(error)
            error_pos[error_pos < 0] = 0
            error_neg = -copy.deepcopy(error)
            error_neg[error_neg < 0] = 0

            if stats:
                print()
                print_stats(error.cpu().detach().numpy(), "** Error ")
                print_stats(cross_entropy_loss.cpu().detach().numpy(), "** Cross ")
        
        if self.loss_mode == 'final':
            if show_stats:
                    print("\n Feedback Simulation")
                    print("Output Layer")
            pos_spike, neg_spike, out_feedback_state, act = self.output_cell.train_feedback_step(
                        error_pos, error_neg, out_feedback_state, show_stats)
            for idx in reversed(range(len(self.hidden_cells))):
                if show_stats:
                    print("\n Hidden Layer ", idx)
                hidden_feedback_states[idx] = self.hidden_cells[idx].train_feedback_step(
                    pos_spike, neg_spike, hidden_feedback_states[idx], show_stats)
            feedback_step = 1
            activations_fb += act
        
        if self.loss_mode == 'direct':
            if show_stats:
                    print("\n Feedback Simulation")
            self.output_cell.loss = error
            for idx in reversed(range(len(self.hidden_cells))):
                if show_stats:
                    print("\n Hidden Layer ", idx)
                hidden_feedback_states[idx] = self.hidden_cells[idx].train_feedback_step(
                    error, None, hidden_feedback_states[idx], show_stats)
                error = hidden_feedback_states[idx]
                if idx > 0: # Error is backpropagated to the previous hidden layer
                    error = error.view(-1, self.hidden_cells[idx].n_filters, 
                                       self.hidden_cells[idx].output_dim[0], self.hidden_cells[idx].output_dim[1])

        # Predict label
        predict_label = torch.argmax(output, 1)
        activations = activations / spike_ts
        activations_fb = activations_fb / feedback_step

        return predict_label, hidden_forward_states, hidden_feedback_states, out_forward_state, out_feedback_state, feedback_step, max_volts, activations, activations_fb

    def train_update_parameter(self, hidden_forward_states, hidden_feedback_states,
                               out_forward_state, out_feedback_state, feedback_step, stats=False):
        """
        Update parameter of the SNN

        Args:
            hidden_forward_states (list): list of hidden forward states
            hidden_feedback_states (list): list of hidden feedback states
            out_forward_state (tuple): output forward state
            out_feedback_state (tuple): output feedback state
            feedback_step (int): number of steps for feedback simulation
            lr (float): learning rate

        """
        if stats:
            print("\n Update Parameters")
        deltas_dict = {}
        # Update Hidden Layer weight and bias
        for idx, cell in enumerate(self.hidden_cells, 0):
            if stats:
                print("\n Hidden Layer ", idx)
            trace_dw, trace_db = hidden_forward_states[idx][3], hidden_forward_states[idx][5]
            error_volt = hidden_feedback_states[idx]
            feedback_step = 1 if self.loss_mode == 'direct' else feedback_step

            _, deltas = cell.train_update_parameter_sgd((error_volt, feedback_step, trace_dw, trace_db), stats)
            deltas_dict[cell.name] = deltas[0]
            deltas_dict[cell.name + " Abs"] = deltas[1]

        # Update Output Layer weight and bias
        trace_pre, trace_dw, trace_db = out_forward_state[2], out_forward_state[3], out_forward_state[5]
        # trace_dw = trace_dw * trace_pre
        if self.loss_mode == 'direct':
            error_volt = self.output_cell.loss
            feedback_step = 1
        else:
            error_volt = out_feedback_state[4]
        if stats:
            print("\n Output Layer")
        _, deltas = self.output_cell.train_update_parameter_sgd((error_volt, feedback_step, trace_dw, trace_db), stats)
        deltas_dict[self.output_cell.name] = deltas[0]
        deltas_dict[self.output_cell.name + " Abs"] = deltas[1]

        # Clamp the weights to n_bits
        for cell in self.hidden_cells:
            cell.forward_func.weight.data = torch.clamp(cell.forward_func.weight.data, -2**(self.n_bits-1), 2**(self.n_bits-1))
            if self.bias:
                cell.forward_func.bias.data = torch.clamp(cell.forward_func.bias.data, -2**(self.n_bits-1), 2**(self.n_bits-1))
        self.output_cell.forward_func.weight.data = torch.clamp(self.output_cell.forward_func.weight.data, -2**(self.n_bits-1), 2**(self.n_bits-1))
        if self.bias:
            self.output_cell.forward_func.bias.data = torch.clamp(self.output_cell.forward_func.bias.data, -2**(self.n_bits-1), 2**(self.n_bits-1))
            
        if not self.aligned_weights:
            for idx, cell in enumerate(self.hidden_cells):
                if idx == self.n_layers - 2: # Last hidden layer
                    cell.feedback_func.weight.data = self.output_cell.forward_func.weight.data.t()
                else:
                    cell.feedback_func.weight.data = self.hidden_cells[idx+1].forward_func.weight.data
            
        return deltas_dict

    def sleep_feedback_update(self, batch_size, spike_ts, oja_power, lr):
        """
        Sleep phase for feedback weight update using spike-based weight mirror

        Args:
            batch_size (int): batch size
            spike_ts (int): spike timesteps
            oja_power (float): oja power factor for oja decay
            lr (float): learning rate

        """
        noise_pos = torch.rand(1)[0]
        noise_neg = noise_pos

        for idx in reversed(range(len(self.hidden_cells))):
            # Generate Poisson Positive and Negative input spikes for this hidden layer
            hidden_output_dim = self.hidden_cells[idx].output_dim
            poisson_spike_pos = Bernoulli(torch.full_like(torch.zeros(batch_size, hidden_output_dim, spike_ts,
                                                                      device=self.device), noise_pos)).sample()
            poisson_spike_neg = Bernoulli(torch.full_like(torch.zeros(batch_size, hidden_output_dim, spike_ts,
                                                                      device=self.device), noise_neg)).sample()

            # Init Hidden Layer Cell States
            hidden_forward_states = []
            for ii in range(idx+1, len(self.hidden_cells)):
                forward_state = self.hidden_cells[ii].test_reset_state(batch_size, self.device)
                hidden_forward_states.append(forward_state)

            # Init Output Layer Cell State
            out_forward_state = self.output_cell.test_reset_state(batch_size, self.device)

            # Init Hidden Layer Spike Trace and Output Spike Trace
            hidden_spike_trace = torch.zeros(batch_size, hidden_output_dim, device=self.device)
            output_spike_trace = torch.zeros(batch_size, self.output_dim, device=self.device)

            # Start Sleeping for this Hidden Layer
            for tt in range(spike_ts):
                input_spike_pos = poisson_spike_pos[:, :, tt]
                input_spike_neg = poisson_spike_neg[:, :, tt]
                hidden_spike_trace = hidden_spike_trace + input_spike_pos - input_spike_neg
                if len(hidden_forward_states) == 0:
                    spike_output, out_forward_state = self.output_cell.sleep_forward_step(input_spike_pos,
                                                                                          input_spike_neg,
                                                                                          out_forward_state)
                else:
                    input_spike, hidden_forward_states[0] = self.hidden_cells[idx+1].sleep_forward_step(input_spike_pos,
                                                                                                        input_spike_neg,
                                                                                                        hidden_forward_states[0])
                    for ii in range(1, len(hidden_forward_states)):
                        input_spike, hidden_forward_states[ii] = self.hidden_cells[idx+ii+1].test_forward_step(input_spike,
                                                                                                               hidden_forward_states[ii])
                    spike_output, out_forward_state, _ = self.output_cell.test_forward_step(input_spike, out_forward_state)
                output_spike_trace = output_spike_trace + spike_output

            # Compute Correlation for feedback weight update
            corr_batch_sum = torch.matmul(hidden_spike_trace.t(), output_spike_trace)

            # Compute Decay base on Oja's Rule
            oja_decay = torch.mul(torch.mean(torch.pow(output_spike_trace, oja_power), axis=0),
                                  self.hidden_cells[idx].feedback_func.weight.data)

            # Update Feedback Weights for this Hidden Layer
            self.hidden_cells[idx].feedback_func.weight.data += lr * (corr_batch_sum - oja_decay)

    def test(self, spike_data, stats=False):
        """
        Test SNN

        Args:
            spike_data (Tensor): spike data input (batch_size, input_dim, spike_ts)

        Returns:
            predict_label: predict labels

        """
        batch_size = spike_data.shape[0]
        spike_ts = spike_data.shape[-1]
        # if len(spike_data.shape) > 3:
        #     spike_data = spike_data.view(batch_size, self.input_dim, spike_ts)
        spike_data = spike_data.view(batch_size, self.input_dim[0], self.input_dim[1], spike_ts)

        # Init Hidden Layer Cell States
        hidden_forward_states = []
        for cell in self.hidden_cells:
            forward_state = cell.test_reset_state(batch_size, self.device)
            hidden_forward_states.append(forward_state)

        # Init Output Layer Cell State
        out_forward_state = self.output_cell.test_reset_state(batch_size, self.device)

        # Start online simulation of the network
        output = torch.zeros([batch_size, self.output_cell.output_dim], device=self.device)
        activations = torch.zeros(self.n_layers)
        for tt in range(spike_ts):
            input_spike = spike_data[:, :, :, tt].unsqueeze(1)
            if stats:
                print_stats(input_spike.cpu().detach().numpy(), "\n*** Input Spike " + str(tt))
            for idx, cell in enumerate(self.hidden_cells, 0):
                if stats:
                    print("\n Hidden Layer ", idx)
                input_spike, hidden_forward_states[idx], activation = cell.test_forward_step(input_spike,
                                                                                 hidden_forward_states[idx], stats)
                activations[idx] += activation
            if stats:
                print("\n Output Layer")
            input_spike = input_spike.view(batch_size, -1) # Flatten input for fully connected layer
            out_spike, out_forward_state, activation = self.output_cell.test_forward_step(input_spike, out_forward_state, stats)
            activations[-1] += activation
            output = output + out_spike

        # Predict label
        predict_label = torch.argmax(output, 1)
        activations = activations / spike_ts

        return predict_label, activations

    def compute_feedback_angle_ratio(self):
        """
        Compute angle and magnitude ratio between feedback connection and forward connection for each layer

        Returns:
            angle_list: list of angle (from lower hidden layer to higher hidden layer)
            ratio_list: list of magnitude ratio (from lower hidden layer to higher hidden layer)

        """
        angle_list, ratio_list = [], []
        forward_weight = copy.deepcopy(self.output_cell.forward_func.weight.data)
        for idx in reversed(range(len(self.hidden_cells))):
            feedback_weight = copy.deepcopy(self.hidden_cells[idx].feedback_func.weight.data)
            angle, ratio = self.compute_angle_ratio_between_weight_matrix(feedback_weight,
                                                                          copy.deepcopy(forward_weight.t()))
            angle_list.append(angle)
            ratio_list.append(ratio)
            if idx > 0:
                forward_weight = torch.matmul(forward_weight, self.hidden_cells[idx].forward_func.weight.data)

        return angle_list, ratio_list

    def compute_angle_ratio_between_weight_matrix(self, weight1, weight2):
        """
        Compute angle and magnitude ratio between two weight matrix

        Args:
            weight1 (Tensor): weight matrix 1
            weight2 (Tensor): weight matrix 2

        Returns:
            angle: angle between two weight matrix
            ratio: magnitude ratio between two weight matrix

        """
        flatten_weight1 = torch.flatten(weight1)
        flatten_weight2 = torch.flatten(weight2)
        weight1_norm = torch.norm(flatten_weight1)
        weight2_norm = torch.norm(flatten_weight2)
        ratio = (weight1_norm / weight2_norm).to('cpu').item()

        weight_cos = self.cos_func(flatten_weight1, flatten_weight2)
        angle = (180. / math.pi) * torch.acos(weight_cos).to('cpu').item()

        return angle, ratio
