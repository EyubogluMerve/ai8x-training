#!/usr/bin/env python3

"""
Training+validation log parser
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def log_parser(log_path, plot_loss = True, plot_acc = True):# Regular expressions to extract relevant information

    # Target values to locate
    epoch_pattern = re.compile(r'Epoch: \[(\d+)\]')
    loss_pattern = re.compile(r'Overall Loss (\d+\.\d+)')
    validation_loss_pattern = re.compile(r']    Loss\s*([\d.]+)')
    top1_pattern = re.compile(r'Best \[Top1: \s*([\d.]+)')

    # Open and read the log file
    with open(log_path, 'r') as log_file:
        log_contents = log_file.read()

    # Find corresponding values
    epoch_matches = re.findall(epoch_pattern, log_contents)
    loss_matches = re.findall(loss_pattern, log_contents)
    validation_loss_matches = re.findall(validation_loss_pattern, log_contents)
    top1_matches = re.findall(top1_pattern, log_contents)

    # Convert extracted data to appropriate data types
    epochs = [int(match) for match in epoch_matches]
    losses = [float(match) for match in loss_matches]
    validation_losses = [float(match) for match in validation_loss_matches]
    top1_accuracies = [float(match) for match in top1_matches]
    validation_losses.pop()

    n_epochs = np.linspace(0,epochs[-1],epochs[-1] + 1)
    avg_val_losses = last_loss(validation_losses, n_epochs, 5)
    avg_losses = last_loss(losses, n_epochs, 41)

    # Plot training + val loss vs epoch
    if plot_loss:
        plt.plot(n_epochs, avg_losses, label='Training Loss')
        plt.plot(n_epochs, avg_val_losses, label='Validation Loss',color="r")
        plt.legend(loc="upper right")
        plt.xlabel('Epoch')
        plt.ylabel('Objective Loss')
        plt.title('Training Objective Loss Over Epochs')
        plt.grid(True)
        plt.show()

    # Plot top1 acc  vs epoch
    if plot_acc:
        plt.plot(n_epochs, top1_accuracies, label='Top-1 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.title('Top-1 Accuracy vs Epoch')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    return 0

def last_loss(loss,n_epochs,n_batch):
    n = 200
    last_loss = np.zeros(n)
    cnt = 0

    for i in range(len(loss)):
        if (i+1) % n_batch == 0:
            last_loss[cnt] = loss[i]
            cnt += 1
    return last_loss

    




