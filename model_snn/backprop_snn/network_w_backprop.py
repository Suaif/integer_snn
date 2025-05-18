import torch
import torch.nn as nn
from backprop_snn.network_cells import PseudoSpikeRect, LinearLIFCell


class BackpropNetworkWithSTBP(nn.Module):
    """ Backprop trained SNN with STBP """

    def __init__(self, input_dim, output_dim, hidden_dim_list, param_dict):
        """

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim_list (list): list of hidden layer dimension
            param_dict (dict): neuron parameter dictionary
        """
        super(BackpropNetworkWithSTBP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        pseudo_grad_ops = PseudoSpikeRect.apply
        hidden_cell_list = []
        for idx, hh in enumerate(hidden_dim_list, 0):
            forward_output_dim = hh
            if idx == 0:
                forward_input_dim = input_dim
            else:
                forward_input_dim = hidden_dim_list[idx - 1]
            hidden_cell_list.append(LinearLIFCell(nn.Linear(forward_input_dim, forward_output_dim),
                                                  pseudo_grad_ops, param_dict['hidden_layer'][idx]))
        self.hidden_cells = nn.ModuleList(hidden_cell_list)
        self.output_cell = LinearLIFCell(nn.Linear(hidden_dim_list[-1], output_dim),
                                         pseudo_grad_ops, param_dict['out_layer'])

    def forward(self, spike_data, init_states_dict, batch_size, spike_ts):
        """
        Forward function

        Args:
            spike_data (Tensor): spike data input (batch_size, input_dim, spike_ts)
            init_states_dict (dict): init states for each layer
            batch_size (int): batch size
            spike_ts (int): spike timesteps

        Returns:
            output: number of spikes of output layer

        """
        hidden_states = init_states_dict['hidden_layer']
        out_state = init_states_dict['out_layer']
        spike_data_flatten = spike_data.view(batch_size, self.input_dim, spike_ts)
        output_list = []
        for tt in range(spike_ts):
            input_spike = spike_data_flatten[:, :, tt]
            for idx, hh in enumerate(self.hidden_cells, 0):
                input_spike, hidden_states[idx] = hh(input_spike, hidden_states[idx])
            out_spike, out_state = self.output_cell(input_spike, out_state)
            output_list.append(out_spike)
        output = torch.stack(output_list).sum(dim=0)
        return output


class WrapBackpropNetworkWithSTBP(nn.Module):
    """ Wrap of Backprop trained SNN with STBP """

    def __init__(self, input_dim, output_dim, hidden_dim_list, param_dict, device):
        """

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim_list (list): list of hidden layer dimension
            param_dict (dict): neuron parameter dictionary
            device (device): device
        """
        super(WrapBackpropNetworkWithSTBP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim_list = hidden_dim_list
        self.device = device
        self.snn = BackpropNetworkWithSTBP(input_dim, output_dim, hidden_dim_list, param_dict)

    def forward(self, spike_data):
        """
        Forward function

        Args:
            spike_data (Tensor): spike data input

        Returns:
            output: number of spikes of output layer

        """
        batch_size = spike_data.shape[0]
        spike_ts = spike_data.shape[-1]
        init_states_dict = {}
        # Hidden layer
        hidden_states = []
        for hh in self.hidden_dim_list:
            hidden_volt = torch.zeros(batch_size, hh, device=self.device)
            hidden_spike = torch.zeros(batch_size, hh, device=self.device)
            hidden_states.append((hidden_spike, hidden_volt))
        init_states_dict['hidden_layer'] = hidden_states
        # Output layer
        out_volt = torch.zeros(batch_size, self.output_dim, device=self.device)
        out_spike = torch.zeros(batch_size, self.output_dim, device=self.device)
        init_states_dict['out_layer'] = (out_spike, out_volt)
        # SNN
        output = self.snn(spike_data, init_states_dict, batch_size, spike_ts)
        return output
