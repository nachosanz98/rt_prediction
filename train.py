from time import time
from matplotlib import pyplot as plt
import torch
import pygad
from neural_network import NeuralNetwork
from prepare_data import read_and_create
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    file_name = 'SMRT_vectorfingerprints.csv'

    train_loader, val_loader, test_loader, num_inputs = read_and_create(file_name)

    num_outputs = 1
    weight_decay = 1e-4

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

        model = NeuralNetwork(num_inputs, num_outputs, num_hiddens_list)
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

    model = NeuralNetwork(num_inputs, num_outputs, num_hiddens_list)
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
    training_model(model, train_loader, val_loader, loss_mae, optimizer, epochs, scheduler, loss_name='MAE_W=4')
    after = time()

    time_ = (after - before) / 60
    print('Time of inference in min: ', time_)

    evaluate_model(model, test_loader, loss_mae)

    # before = time()
    # training_model(model, train_loader, val_loader, loss_mse, optimizer, epochs, loss_name='MSE_W=2')
    # after = time()

    # time_ = (after - before) / 60
    # print('Time of inference in min: ', time_)

    # evaluate_model(model, test_loader, loss_mse)

def training_model(model, train_loader, val_loader, criterion, optimizer, epochs, scheduler, loss_name):

    train_losses = []
    val_losses = []
    early_stopping_patience = 20
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

if __name__ == '__main__':
    main()