import math
import matplotlib.pyplot as plt

# Function to find the nearest power of two with a negative exponent
def nearest_power_of_two(value):
    if value == 0:
        return 0
    # Find the log base 2 and round it
    exponent = int(round(math.log2(value)))
    # Ensure the exponent is negative or zero
    if exponent > 0:
        exponent = -exponent
    # Return the power of two
    return 2 ** exponent

# Custom Scheduler: Power of Two Exponential Decay
class ExponentialDecay:
    def __init__(self, initial_lr=0.5, decay_factor=2, min_lr=2**-40):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.current_lr = initial_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        # Exponential decay adjusted to nearest power of two
        self.current_lr /= self.decay_factor
        self.current_lr = max(self.min_lr, self.current_lr)
        # self.current_lr = max(self.min_lr, nearest_power_of_two(self.current_lr))

        return self.current_lr

# Custom Scheduler: Power of Two Cosine Annealing Warm Restarts
class CosineAnnealingWarmRestarts:
    def __init__(self, initial_lr=0.1, T_0=10, T_mult=2, min_lr=2**-40):
        self.T_0 = T_0
        self.T_i = T_0  # current T_i
        self.T_mult = T_mult
        self.min_lr = min_lr
        self.last_epoch = 0
        self.base_lrs = [initial_lr]
        self.current_lr = initial_lr

    def step(self):
        self.last_epoch += 1
        if self.last_epoch == self.T_i:
            self.T_i = self.T_i * self.T_mult if self.T_mult > 1 else self.T_i + 1
            self.last_epoch = 0
        self.current_lr = self.base_lrs[0] * (1 + math.cos(math.pi * self.last_epoch / self.T_i)) / 2
        self.current_lr = max(self.min_lr, self.current_lr)
        # self.current_lr = max(self.min_lr, nearest_power_of_two(self.current_lr))

        return self.current_lr

# # Custom Scheduler: Power of Two ReduceLROnPlateau
# class ReduceLROnPlateau:
#     def __init__(self, initial_lr=0.1, mode='min', factor=0.5, patience=5, min_lr=2**-16):
#         self.mode = mode
#         self.factor = factor
#         self.patience = patience
#         self.min_lr = min_lr
#         self.best = float('inf') if mode == 'min' else -float('inf')
#         self.num_bad_epochs = 0
#         self.current_lr = initial_lr

#     def step(self, metric):
#         if self.mode == 'min':
#             improved = metric < self.best
#         else:
#             improved = metric > self.best
        
#         if improved:
#             self.best = metric
#             self.num_bad_epochs = 0
#         else:
#             self.num_bad_epochs += 1
        
#         if self.num_bad_epochs > self.patience:
#             self.current_lr *= self.factor
#             self.current_lr = max(self.min_lr, self.current_lr)
#             self.num_bad_epochs = 0
        
#         return self.current_lr
