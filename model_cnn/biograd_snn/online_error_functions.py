import torch.nn as nn
import torch
from quantization import truncated_division

def cross_entropy_loss_error_function(network_output, label_one_hot):
    """
    Compute output error signal (Here use CrossEntropyLoss)

    Args:
        network_output (Tensor): output of the SNN
        label_one_hot (Tensor): one hot vector of the label

    Returns:
        error: error vector

    """
    output_softmax = nn.functional.softmax(network_output, dim=-1)
    error = output_softmax - label_one_hot

    return error

def pseudo_softmax_torch(x):
    exp_x = torch.pow(2, x)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

def int_softmax(network_output, label_one_hot, norm=10, loss_precision=32, float_mode=False):
    """ Compute the RSS loss function """

    max_prediction = torch.max(torch.abs(network_output), dim=1, keepdim=True)[0]
    max_prediction[max_prediction == 0] = 1
    if not float_mode:
        output_softmax = (network_output * loss_precision) // norm
        # output_softmax2 = nn.functional.softmax(network_output, dim=-1) * loss_precision
        # output_softmax = pseudo_softmax_torch(network_output) * loss_precision
    else:
        # output_softmax = network_output / max_prediction
        output_softmax = nn.functional.softmax(network_output, dim=-1)

    error = output_softmax - label_one_hot
    
    return error

def int_softmax_positive(network_output, label_one_hot, norm=10, loss_precision=32, float_mode=False):

    max_prediction = torch.max(torch.abs(network_output), dim=1, keepdim=True)[0]
    max_prediction[max_prediction == 0] = 1
    if not float_mode:
        output_softmax = (network_output * loss_precision) // norm
    else:
        output_softmax = nn.functional.softmax(network_output, dim=-1)

    error = output_softmax - label_one_hot

    if not float_mode:
        error[error == 0] += 1
    
    return error

def create_int_softmax_random(min_random_default, max_random_default):
        
    def int_softmax_random(network_output, label_one_hot, norm=10, min_random=min_random_default, max_random=max_random_default, loss_precision=32, float_mode=False):

        max_prediction = torch.max(torch.abs(network_output), dim=1, keepdim=True)[0]
        max_prediction[max_prediction == 0] = 1
        if not float_mode:
            output_softmax = (network_output * loss_precision) // norm
        else:
            output_softmax = nn.functional.softmax(network_output, dim=-1)

        error = output_softmax - label_one_hot
        if not float_mode:
            random_tensor = torch.randint(min_random, max_random, error.shape, dtype=error.dtype, device=error.device)
            error += random_tensor

        return error

    return int_softmax_random

class LossFunction():
    def __init__(self, loss_function='normal', parameters=None):

        if loss_function == 'normal':
            self.loss = int_softmax
            self.name = 'normal'
        elif loss_function == 'positive':
            self.loss = int_softmax_positive
            self.name = 'positive'
        elif loss_function == 'random':
            self.loss = create_int_softmax_random(parameters[0], parameters[1])
            self.name = f'random_{parameters[0]}_{parameters[1]}'
        else:
            raise ValueError("Loss function not recognized")
