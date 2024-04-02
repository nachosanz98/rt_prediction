import torch

from neuronal_network import NeuronalNetwork
from prepare_data import read_and_create
from torch import nn, optim
from datetime import datetime

def main():
    file_name = 'SMILES_Big_Data_Set.csv'

    train_loader, test_loader, num_inputs = read_and_create(file_name)

    num_hiddens = 256
    num_outputs = 1
    lr = 0.001
    sigma = 0.001
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model = NeuronalNetwork(num_inputs, num_outputs, num_hiddens, lr, sigma)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 10
    best_vloss = 1_000_000
    for epoch in range(epochs):
        model.train(True)
        running_loss = 0
        last_loss = 0
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
        
        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(running_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()