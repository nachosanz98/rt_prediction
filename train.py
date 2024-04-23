import torch

from neuronal_network import NeuronalNetwork
from prepare_data import read_and_create
from torch import nn, optim
from datetime import datetime

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    file_name = 'SMILES_Big_Data_Set_Cleaned.csv'

    train_loader, test_loader, num_inputs = read_and_create(file_name)

    num_hiddens = 256
    num_outputs = 1
    lr = 0.001
    sigma = 0.001
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model = NeuronalNetwork(num_inputs, num_outputs, num_hiddens, lr, sigma)
    model.apply(init_weights)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    epochs = 100
    best_vloss = float('inf')
    for epoch in range(epochs):
        model.train(True)
        running_loss = 0
        total_loss = 0
        count_batches = 0
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_loss += loss.item()
            count_batches += 1
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
        
        avg_loss = total_loss / count_batches
        print('Average training loss for epoch {}: {}'.format(epoch+1, avg_loss))

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}.pt'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()