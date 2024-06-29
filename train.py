from time import time
from matplotlib import pyplot as plt
import torch
import pygad
from neural_network import NeuralNetwork
from prepare_data import read_and_create
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import numpy as np
import scipy.stats as stats
from statsmodels.graphics.regressionplots import plot_leverage_resid2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    file_name = 'SMRT_vectorfingerprints.csv'

    train_loader, val_loader, test_loader, num_inputs = read_and_create(file_name)
    print('Data is ready')

    num_outputs = 1

    # PyGAD parameters
    num_generations = 50
    num_parents_mating = 5
    sol_per_pop = 10
    num_genes = 7
    gene_space = [
        [16, 32, 64, 128, 256, 512, 1024],
        [16, 32, 64, 128, 256, 512, 1024],
        [16, 32, 64, 128, 256, 512, 1024],
        [0.0001, 0.001, 0.01, 0.1], # learning rate
        [0, 1, 2], # Adam, AdamW, SGD
        [1, 5, 10, 20], # step for scheduler
        [1e-6, 1e-5, 1e-4, 1e-3] # weight decay
    ]

    def fitness_function(ga_instance, solution, solution_idx):
        num_hiddens_list = [int(x) for x in solution[:3]]
        learning_rate = solution[3]
        optimizer_name = solution[4]
        step_size = int(solution[5])
        weight_decay = solution[6]

        model = NeuralNetwork(num_inputs, num_outputs, num_hiddens_list).to(device)
        model.apply(init_weights)
        criterion = nn.L1Loss()

        if optimizer_name == 0:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 1:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 2:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

        early_stopping_patience = 10
        epochs_no_improve = 0
        best_vloss = float('inf')

        for epoch in range(100):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch}, Val Loss: {val_loss}")

            if val_loss < best_vloss:
                best_vloss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                break

        return -best_vloss

    def on_generation(ga_instance):
        progress = (ga_instance.generations_completed / ga_instance.num_generations) * 100
        print(f"Generations Completed: {ga_instance.generations_completed}, Progress: {progress:.2f}%")

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           on_generation=on_generation)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Mejor solución (hidden_dims): {solution}, Fitness: {solution_fitness}")

    num_hiddens_list = [int(x) for x in solution[:3]]
    learning_rate = solution[3]
    optimizer_name = solution[4]
    step_size = int(solution[5])
    weight_decay = solution[6]

    model = NeuralNetwork(num_inputs, num_outputs, num_hiddens_list).to(device)
    model.apply(init_weights)

    loss_mae = torch.nn.L1Loss() # Comparable a MAE
    # loss_mse = torch.nn.MSELoss()

    if optimizer_name == 0:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 1:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 2:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    epochs = 100

    loss_name = f'MAE_W={weight_decay}'

    before = time()
    training_model(model, train_loader, val_loader, loss_mae, optimizer, epochs, scheduler, loss_name)
    after = time()

    time_ = (after - before) / 60
    print('Time of inference in min: ', time_)

    evaluate_model(model, test_loader, loss_mae)

    plot_actual_vs_predicted(model, test_loader)
    plot_residuals_histogram(model, test_loader)
    plot_residuals_vs_fitted(model, test_loader)
    plot_qq(model, test_loader)
    plot_scale_location(model, test_loader)
    plot_residuals_vs_leverage(model, test_loader)

    # before = time()
    # training_model(model, train_loader, val_loader, loss_mse, optimizer, epochs, loss_name='MSE_W=2')
    # after = time()

    # time_ = (after - before) / 60
    # print('Time of inference in min: ', time_)

    # evaluate_model(model, test_loader, loss_mse)

def training_model(model, train_loader, val_loader, criterion, optimizer, epochs, scheduler, loss_name):

    train_losses = []
    val_losses = []
    early_stopping_patience = 10
    epochs_no_improve = 0
    best_vloss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_vloss:
            best_vloss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print('Saving the model...')
    name_model = f'{loss_name}_function'
    torch.save(model, f'models/{name_model}.pth')

    plt.figure(figsize=(10, 4))

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()


    plt.tight_layout()
    plt.savefig(f'images/training_validation_loss_{loss_name}_function.png')
    #plt.show()

def evaluate_model(model, test_loader, criterion):
    model.eval()

    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

def plot_actual_vs_predicted(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig('images/actual_vs_predicted.png')

def plot_residuals_histogram(model, test_loader):
    model.eval()
    residuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            residuals.extend((outputs - labels).cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('images/residuals_histogram.png')

def plot_residuals_vs_fitted(model, test_loader):
    model.eval()
    residuals = []
    fitted = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            residuals.extend((outputs - labels).cpu().numpy())
            fitted.extend(outputs.cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.scatter(fitted, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.savefig('images/residuals_vs_fitted.png')

def plot_qq(model, test_loader):
    model.eval()
    residuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            residuals.extend((outputs - labels).cpu().numpy())

    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normality Q-Q Plot')
    plt.savefig('images/qq_plot.png')

def plot_scale_location(model, test_loader):
    model.eval()
    residuals = []
    fitted = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            residuals.extend((outputs - labels).cpu().numpy())
            fitted.extend(outputs.cpu().numpy())

    residuals = np.array(residuals)
    fitted = np.array(fitted)
    sqrt_residuals = np.sqrt(np.abs(residuals))

    plt.figure(figsize=(10, 6))
    plt.scatter(fitted, sqrt_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Scale Location Plot')
    plt.xlabel('Fitted Values')
    plt.ylabel('Sqrt(|Residuals|)')
    plt.savefig('images/scale_location.png')

def plot_residuals_vs_leverage(model, test_loader):
    model.eval()
    residuals = []
    leverage = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            residuals.extend((outputs - labels).cpu().numpy())
            leverage.extend(inputs.cpu().numpy()) # Placeholder, calculate leverage correctly

    plt.figure(figsize=(10, 6))
    plot_leverage_resid2(model, ax=plt.gca())
    plt.title('Residuals vs Leverage')
    plt.savefig('images/residuals_vs_leverage.png')
    plt.show()


if __name__ == '__main__':
    main()
