import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_actual_vs_predicted(model, test_loader, loss_name):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy() / 60)  # Minutes
            actuals.extend(labels.cpu().numpy() / 60)

    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values (Minutes)')
    plt.ylabel('Predicted Values (Minutes)')
    plt.savefig(f'images/{loss_name}_actual_vs_predicted.png')

def plot_error_distribution(model, test_loader, loss_name):
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            errors.extend((outputs - labels).view(-1).cpu().numpy() / 60)

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='red', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Errors (Minutes)')
    plt.ylabel('Frequency')
    plt.savefig(f'images/{loss_name}_error_distribution.png')

def plot_error_distribution_over_time(model, test_loader, loss_name):
    model.eval()
    errors = []
    timestamps = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_errors = (outputs - labels).view(-1).cpu().numpy() / 60
            errors.extend(batch_errors)
            timestamps.extend([i] * len(batch_errors))

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, errors, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel('Time (Batch Index)')
    plt.ylabel('Errors (Minutes)')
    plt.title('Error Distribution Over Time')
    plt.grid(True)
    plt.savefig(f'images/{loss_name}_error_distribution_over_time.png')
