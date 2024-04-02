import torch

from neuronal_network import NeuronalNetwork
from prepare_data import read_and_create
from torch import nn, optim

def main():
    file_name = 'SMILES_Big_Data_Set.csv'

    train_loader, test_loader, num_inputs = read_and_create(file_name)

    num_hiddens = 256
    num_outputs = 1
    lr = 0.001
    sigma = 0.001

    model = NeuronalNetwork(num_inputs, num_outputs, num_hiddens, lr, sigma)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        
        test_loss = evaluate_model(model, test_loader, loss_fn)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss}')
    
def evaluate_model(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

if __name__ == '__main__':
    main()