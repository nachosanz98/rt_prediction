from time import time
from matplotlib import pyplot as plt
import torch

from neuronal_network import NeuronalNetwork
from prepare_data import read_and_create
from torch import nn, optim
from datetime import datetime

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    file_name = 'SMRT_vectorfingerprints.csv'

    train_loader, val_loader, test_loader, num_inputs = read_and_create(file_name)

    num_hiddens = 1024
    num_outputs = 1
    lr = 0.01
    weight_decay = 1e-4
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model = NeuronalNetwork(num_inputs, num_outputs, num_hiddens)
    model.apply(init_weights)

    loss_fn = torch.nn.L1Loss() #Comparable a MAE
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 100

    before = time()
    training_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, timestamp)
    after = time()

    time_ = (after - before) / 60
    print('Time of inference in min: ', time_)

    evaluate_model(model, test_loader, loss_fn)

def training_model(model, train_loader, val_loader, criterion, optimizer, epochs, timestamp):

    train_losses = []
    val_losses = []
    early_stopping_patience = 20
    epochs_no_improve = 0
    best_vloss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
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
    name_model = f'{timestamp}_function'
    torch.save(model, f'models/{name_model}.pth')

    plt.figure(figsize=(10, 4))

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()


    plt.tight_layout()
    plt.savefig(f'images/training_validation_loss_{timestamp}_function.png')
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

    print(f'Test Loss: {test_loss:.4f}')
    return test_loss


if __name__ == '__main__':
    main()