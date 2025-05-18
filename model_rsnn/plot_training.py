import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training(training_loss, val_loss, test_loss, title, save_path, best_epoch, plot=False):
    fontsize=25
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(training_loss, label='Training Loss')
    if len(val_loss) > 0:
        ax.plot(val_loss, label='Validation Accuracy')
    if len(test_loss) > 0:
        ax.plot(test_loss, label='Test Accuracy')
    ax.axvline(x=best_epoch, color='red', linestyle='--')
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.grid()
    ax.set_title(title + "\nTraining, Validation and Test Loss", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig(save_path + '/training_plot.png')
    if plot:
        plt.show()
    plt.close(fig)

def plot_fb(fb_angle, fb_ration, title, save_path, best_epoch, plot=False):
    fontsize=25
    fig, ax1 = plt.subplots(figsize=(20, 12))
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=fontsize)
    ax1.set_ylabel('Feeback Angle', color=color, fontsize=fontsize)
    ax1.plot(fb_angle, color=color, label='FB Angle')
    ax1.axvline(x=best_epoch, color='red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=fontsize)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Feedback Ratio', color=color, fontsize=fontsize)
    ax2.plot(fb_ration, color=color, label='FB Ratio')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)

    # fig.tight_layout()
    ax1.set_title(title + "\nFeedback Angle and Ratio", fontsize=fontsize)
    fig.savefig(save_path + '/feedback_plot.png')
    if plot:
        plt.show()
    plt.close(fig)

def boxplot_weights(hidden_weights, rec_weights, output_weights, 
                    hidden_weights_quant, rec_weights_quant, output_weights_quant,
                    deltas_list, rec_deltas_list, deltas_list_abs, rec_deltas_list_abs,
                    title, save_path, best_epoch, plot=False):
    
    epochs = np.arange(len(output_weights))

    n_rows = len(hidden_weights) + len(rec_weights) + 1
    fig, axs = plt.subplots(n_rows, 4, figsize=(80, 12*n_rows))
    fig.suptitle(title, fontsize=32)
    fontsize = 25
    
    # Flatten axs if it's a single row/column to ensure consistent indexing
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    
    # Hidden layers
    for i, (weights_quant, rec_weights_quant, weights, rec_weights, 
            deltas, rec_deltas, deltas_abs, rec_deltas_abs) in enumerate(zip(hidden_weights_quant, rec_weights_quant,
                                                                             hidden_weights, rec_weights, 
                                                                             deltas_list, rec_deltas_list,
                                                                             deltas_list_abs, rec_deltas_list_abs)):
        
        plot_data = [weights_quant, rec_weights_quant, weights, rec_weights, deltas, rec_deltas, deltas_abs, rec_deltas_abs]
        plot_titles = [f"Quant Weights - Hidden Layer {i}",
                       f"Quant Recurrent Weights - Hidden Layer {i}",
                       f"Weights - Hidden Layer {i}",
                       f"Recurrent Weights - Hidden Layer {i}", 
                       f"Deltas - Hidden Layer {i}", 
                       f"Recurrent Deltas - Hidden Layer {i}",
                       f"Deltas Abs - Hidden Layer {i}",
                       f"Recurrent Deltas Abs - Hidden Layer {i}"]
        colors = ['blue', 'blue', 'blue', 'blue', 'green', 'green', 'green', 'green']
        index = [[2*i, 0], [2*i+1, 0], [2*i, 1], [2*i+1, 1], [2*i, 2], [2*i+1, 2], [2*i, 3], [2*i+1, 3]]
        ylabels = ["Weight Value", "Weight Value", "Weight Value", "Weight Value", "Delta % Value", "Delta % Value", "Delta Value", "Delta Value"]
        
        for j, (w, title, color, idx, ylabel) in enumerate(zip(plot_data, plot_titles, colors, index, ylabels)):
            min_values, p5_values, q1_values, median_values, q3_values, p95_values, max_values = [], [], [], [], [], [], []
        
            for wts in w:
                min_values.append(torch.min(wts).item())
                p5_values.append(torch.quantile(wts, 0.05).item())
                q1_values.append(torch.quantile(wts, 0.25).item())
                median_values.append(torch.median(wts).item())
                q3_values.append(torch.quantile(wts, 0.75).item())
                p95_values.append(torch.quantile(wts, 0.95).item())
                max_values.append(torch.max(wts).item())

            # Plot distribution
            ax = axs[idx[0], idx[1]]  # Adjusted indexing
            ax.plot(epochs, median_values, color=color, label=' Median')
            ax.fill_between(epochs, q1_values, q3_values, alpha=0.3, color=color, label=' Q1-Q3')
            ax.fill_between(epochs, min_values, max_values, alpha=0.2, color=color, label=' Min-Max')
            
            ax.plot(epochs, p5_values, color=color, label=' 5%', linestyle='--')
            ax.plot(epochs, p95_values, color=color, label=' 95%', linestyle='--')
            
            ax.axvline(x=best_epoch, color='red', linestyle='--')
            ax.set_title(title, fontsize=fontsize)
            ax.set_xlabel("Epoch", fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.yaxis.get_offset_text().set_fontsize(fontsize)
            ax.legend(fontsize=fontsize)
            ax.grid(True)

            # Logaritmic scale for delta plots
            if ylabel == "Delta % Value":
                ax.set_yscale('symlog')

    # Output layer
    output_plot_data = [output_weights_quant, output_weights, deltas_list[-1], deltas_list_abs[-1]]
    output_plot_titles = ["Quant Weights - Output Layer", "Weights - Output Layer", "Deltas - Output Layer", "Deltas Abs - Output Layer"]
    ylabels = ["Weight Value", "Weight Value", "Delta % Value", "Delta Value"]
    colors = ['orange', 'orange', 'green', 'green']
    
    for j, (w, title, ylabel, color) in enumerate(zip(output_plot_data, output_plot_titles, ylabels, colors)):
        min_values, p5_values, q1_values, median_values, q3_values, p95_values, max_values = [], [], [], [], [], [], []
        
        for wts in w:
                min_values.append(torch.min(wts).item())
                p5_values.append(torch.quantile(wts, 0.05).item())
                q1_values.append(torch.quantile(wts, 0.25).item())
                median_values.append(torch.median(wts).item())
                q3_values.append(torch.quantile(wts, 0.75).item())
                p95_values.append(torch.quantile(wts, 0.95).item())
                max_values.append(torch.max(wts).item())

        # Plot distribution
        ax = axs[-1, j]  # Last row, specified column
        ax.plot(epochs, median_values, color=color, label=' Median')
        ax.fill_between(epochs, q1_values, q3_values, alpha=0.3, color=color, label=' Q1-Q3')
        ax.fill_between(epochs, min_values, max_values, alpha=0.2, color=color, label=' Min-Max')
        ax.plot(epochs, p5_values, color=color, label=' 5%', linestyle='--')
        ax.plot(epochs, p95_values, color=color, label=' 95%', linestyle='--')

        ax.axvline(x=best_epoch, color='red', linestyle='--')
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.yaxis.get_offset_text().set_fontsize(fontsize)
        ax.legend(fontsize=fontsize)
        ax.grid(True)

        # Logaritmic scale for delta plots
        if ylabel == "Delta % Value":
            ax.set_yscale('symlog')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path + '/weight_boxplot.png')
    if plot:
        plt.show()
    plt.close()

def activations_plot(activation_list, title, save_path, best_epoch, feedback_act=[], plot=False):
        
    n_plots = len(activation_list) + len(feedback_act)
    fig, axs = plt.subplots(n_plots, 1, figsize=(20, 12*n_plots))
    fig.suptitle(title, fontsize=25)
    fontsize=25

    epochs = np.arange(len(activation_list[0][0]))
    
    for i, act_lists in enumerate(activation_list):

        name = 'Hidden Layer {}'.format(i) if i < len(activation_list) - 1 else 'Output Layer'
        if len(act_lists) == 3:
            train_act, val_act, test_act = act_lists
        elif len(act_lists) == 2:
            train_act, val_act = act_lists
            test_act = []
        elif len(act_lists) == 1:
            train_act = act_lists[0]
            val_act, test_act = [], []
        else:
            raise ValueError("Invalid number of activation lists")

        axs[i].plot(epochs, train_act, '-o', color='blue', label=' Train')
        if len(val_act) > 0:
            axs[i].plot(epochs, val_act, '-o', color='orange', label=' Val')
        if len(test_act) > 0:
            axs[i].plot(epochs, test_act, '-o', color='green', label=' Test')

        axs[i].axvline(x=best_epoch, color='red', linestyle='--')
        axs[i].set_title("Activation % - " + name, fontsize=fontsize)
        axs[i].set_xlabel("Epoch", fontsize=fontsize)
        axs[i].set_ylabel("Activation %", fontsize=fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize)
        axs[i].yaxis.get_offset_text().set_fontsize(fontsize)
        axs[i].legend(fontsize=fontsize)
        axs[i].grid(True)
    
    # Feedback activations
    if feedback_act != []:
        pos_act, neg_act = feedback_act
        axs[n_plots-1].plot(epochs, pos_act, '-o', color='blue', label=' Positive')
        axs[n_plots-1].plot(epochs, neg_act, '-o', color='orange', label=' Negative')
        axs[n_plots-1].axvline(x=best_epoch, color='red', linestyle='--')
        axs[n_plots-1].set_title("Feedback Act % - Output Layer", fontsize=fontsize)
        axs[n_plots-1].set_xlabel("Epoch", fontsize=fontsize)
        axs[n_plots-1].set_ylabel("Activation %", fontsize=fontsize)
        axs[n_plots-1].tick_params(axis='both', which='major', labelsize=fontsize)
        axs[n_plots-1].yaxis.get_offset_text().set_fontsize(fontsize)
        axs[n_plots-1].legend(fontsize=fontsize)
        axs[n_plots-1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path + '/activation_plot.png')
    if plot:
        plt.show()
    plt.close()

def plot_max_volts(max_volts, title, save_path, best_epoch, plot=False):
    
    if isinstance(max_volts, list):
        max_volts = np.array(max_volts).T

    
    n_plots = max_volts.shape[1]
    epochs = np.arange(max_volts.shape[0])
    fontsize=25
    fig, axs = plt.subplots(n_plots, 1, figsize=(20, 12*n_plots))
    fig.suptitle(title, fontsize=25)

    for i in range(n_plots):
        name = 'Hidden Layer {}'.format(i) if i < n_plots - 1 else 'Output Layer'

        axs[i].plot(epochs, max_volts[:, i], label=name)
        axs[i].axvline(x=best_epoch, color='red', linestyle='--')
        axs[i].set_title(f'Max Voltage - {name}', fontsize=fontsize)
        axs[i].set_xlabel('Epoch', fontsize=fontsize)
        axs[i].set_ylabel('Voltage', fontsize=fontsize)
        axs[i].legend(fontsize=fontsize)
        axs[i].grid()
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize)
        axs[i].yaxis.get_offset_text().set_fontsize(fontsize)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path + '/max_voltage_plot.png')
    if plot:
        plt.show()
    plt.close()

def plot_lr(lr_lists, title, save_path, best_epoch, plot=False):
    
    n_plots = len(lr_lists)
    epochs = np.arange(len(lr_lists[0]))
    fontsize=25
    fig, axs = plt.subplots(1, n_plots, figsize=(20*n_plots, 20))
    fig.suptitle(title, fontsize=25)

    lr_names = ['Hidden Layer', 'Recurrent Layer', 'Output Layer']

    for i in range(n_plots):
        if i == len(lr_lists) - 1:
            name = 'Output Layer'
        elif i // 2 == 0:
            name = 'Hidden Layer'
        else:
            name = 'Recurrent Layer'
            
        axs[i].plot(epochs, lr_lists[i], label=lr_names[i])
        axs[i].axvline(x=best_epoch, color='red', linestyle='--')
        axs[i].set_title(f'Learning Rate - {name}', fontsize=fontsize)
        axs[i].set_xlabel('Epoch', fontsize=fontsize)
        axs[i].set_ylabel('Learning Rate', fontsize=fontsize)
        axs[i].legend(fontsize=fontsize)
        axs[i].grid()
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize)
        axs[i].yaxis.get_offset_text().set_fontsize(fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path + '/lr_plot.png')
    if plot:
        plt.show()
    plt.close()
