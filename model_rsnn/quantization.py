import numpy as np
import torch
import copy
import math
import matplotlib.pyplot as plt

def print_stats(x, name='', print_info=True):
    info = "{}: mean {:.5f}, std {:.5f}, min {:.5f}, max {:.5f}, ".format(name, np.mean(x), np.std(x), np.min(x), np.max(x))
    info += "5% {:.5f}, 95% {:.5f}".format(np.percentile(x, 5), np.percentile(x, 95))
    if print_info:
        print(info)
    else:
        return info

def quantize_image(image, n_bits):
    image = image * (2**n_bits - 1)
    image = torch.round(image)
    return image

def quantize_layer_weights(weights, num_bits, stats=False, plot=False, max_val=0):
    if max_val == 0:
        max_val = torch.max(torch.abs(weights))
    step = (2 * max_val) / (2**num_bits - 2)
    quantized_weight = torch.round(weights / step).to(torch.int)
    max_quantized = 2**(num_bits-1) - 1
    exp_scale = round(math.log2(max_val / max_quantized))
    dequantized_weight = quantized_weight.to(torch.float) * 2**exp_scale

    if stats:
        print("Step: ", step.item(), step.item()**-1)
        print("Max val: ", max_val.item(), " | Max quantized: ", max_quantized)
        print("Exp scale: ", exp_scale)
        print_stats(weights.cpu().detach().numpy(), "Weight data")
        print_stats(quantized_weight.cpu().detach().numpy(), "Quantized weight")
        print_stats(dequantized_weight.cpu().detach().numpy(), "Dequantized weight")

    if plot:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(weights.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Original weight")
        plt.subplot(1, 3, 2)
        plt.hist(quantized_weight.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Quantized weight")
        plt.subplot(1, 3, 3)
        plt.hist(dequantized_weight.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Dequantized weight")
        plt.show()
    
    return quantized_weight.type(torch.float), step, exp_scale

def quantize_layer_weights_bias(weights, bias, num_bits, stats = False, plot = False, max_val = 0):

    if max_val == 0:
        max_val_w, max_val_b = torch.max(torch.abs(weights)), torch.max(torch.abs(bias))
        max_val = max(max_val_w, max_val_b)
    step = (2 * max_val) / (2**num_bits - 2)
    quantized_weight = torch.round(weights / step).to(torch.int)
    quantized_bias = torch.round(bias / step).to(torch.int)
    max_quantized = 2**(num_bits-1) - 1
    exp_scale = round(math.log2(max_val / max_quantized))
    dequantized_weight = quantized_weight.to(torch.float) * 2**exp_scale
    dequantized_bias = quantized_bias.to(torch.float) * 2**exp_scale

    if stats:
        print("Step: ", step.item(), step.item()**-1)
        print("Max quantized: ", max_quantized)
        print("Exp scale: ", exp_scale)
        print_stats(weights.cpu().detach().numpy(), "Weight data")
        print_stats(bias.cpu().detach().numpy(), "Bias data")
        print_stats(quantized_weight.cpu().detach().numpy(), "Quantized weight")
        print_stats(quantized_bias.cpu().detach().numpy(), "Quantized bias")
        print_stats(dequantized_weight.cpu().detach().numpy(), "Dequantized weight")
        print_stats(dequantized_bias.cpu().detach().numpy(), "Dequantized bias")

    if plot:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.hist(weights.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Original weight")
        plt.subplot(2, 3, 2)
        plt.hist(quantized_weight.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Quantized weight")
        plt.subplot(2, 3, 3)
        plt.hist(dequantized_weight.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Dequantized weight")
        plt.subplot(2, 3, 4)
        plt.hist(bias.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Original bias")
        plt.subplot(2, 3, 5)
        plt.hist(quantized_bias.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Quantized bias")
        plt.subplot(2, 3, 6)
        plt.hist(dequantized_bias.cpu().detach().numpy().flatten(), bins=100)
        plt.title("Dequantized bias")
        plt.show()
    
    return quantized_weight.type(torch.float), quantized_bias.type(torch.float), step, exp_scale

def truncated_division(numerator: torch.Tensor, denominator: torch.Tensor | int) -> torch.Tensor:
    """
    Perform truncated division, which removes the fractional part of the division result.
    This implementation supports PyTorch tensors and works on both CPU and GPU.

    Parameters
    ----------
    numerator : torch.Tensor
        Numerator of the division
    denominator : torch.Tensor | int
        Denominator of the division

    Returns
    -------
    torch.Tensor
        The result of the truncated division
    """
    # Apply floored division
    result = numerator // denominator

    # Adjust the result for cases where numerator and denominator have different signs
    denominator_sign = torch.sign(denominator) if isinstance(denominator, torch.Tensor) else torch.sign(torch.tensor(denominator))
    needs_adjustment = torch.sign(numerator) != denominator_sign
    has_remainder = numerator % denominator != 0
    adjustment = needs_adjustment & has_remainder
    result += adjustment.int()  # Convert adjustment to the same type as result

    # Return the result of the truncated division
    return result

def stochastic_rounding(x):
    """
    Perform stochastic rounding on the input tensor.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be rounded

    Returns
    -------
    torch.Tensor
        Tensor with stochastic rounding applied
    """
    # Get the fractional part of the tensor
    fractional_part = x - torch.floor(x)
    
    # Generate random values between 0 and 1
    random_values = torch.rand_like(x)
    
    # Perform stochastic rounding
    rounded_tensor = torch.where(random_values < fractional_part, torch.ceil(x), torch.floor(x))
    
    return rounded_tensor

def scale_weight(weight, n_bits):
        """
        If the weight (maximum abs value) is greater than 2 ** (n_bits-1), divide by 2 until it is less than 2 ** bits-1
        Return the first n_bits of the weight and the exponent
        """

        max_val = torch.max(torch.abs(weight))
        # If all weights are zero or already within 2**n_bits, no exponent needed.
        if max_val <= 2**(n_bits-1) or max_val == 0:
            return weight, 0
        
        # Compute how many times we need to shift right
        exp = math.ceil(math.log2(max_val.item())) - (n_bits-1)
        exp = max(exp, 0)  # Don’t let exponent be negative
        
        # Scale down and floor
        scale_factor = 2**exp
        scaled_weight = torch.floor(weight / scale_factor)
        
        return scaled_weight, exp

def scale_weight_bias(weight, bias, n_bits):
        """
        If the weight (maximum abs value) is greater than 2 ** n_bits-1, divide by 2 until it is less than 2 ** bits-1
        Return the first n_bits of the weight and the exponent
        """

        max_val = max(torch.max(torch.abs(weight)), torch.max(torch.abs(bias)))
        # If all weights are zero or already within 2**n_bits, no exponent needed.
        if max_val <= 2**(n_bits-1) or max_val == 0:
            return weight, bias, 0
        
        # Compute how many times we need to shift right
        exp = math.ceil(math.log2(max_val.item())) - (n_bits-1)
        exp = max(exp, 0)  # Don’t let exponent be negative
        
        # Scale down and floor
        scale_factor = 2**exp
        scaled_weight = torch.floor(weight / scale_factor)
        scaled_bias = torch.floor(bias / scale_factor)
        
        return scaled_weight, scaled_bias, exp