import numpy as np
import pandas as pd
from sklearn.preprocessing  import StandardScaler
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader

def read_and_create(file_name, chunk_size=1000):
    chunk_list = []
    label_list = []

    for chunk in pd.read_csv(file_name, chunksize=chunk_size):
        feature_columns = [col for col in chunk.columns if col.startswith('V')]
        features = chunk[feature_columns].values
        labels = chunk['rt'].values

        features_tensor = torch.tensor(features).float()
        labels_tensor = torch.tensor(labels).float()

        chunk_list.append(features_tensor)
        label_list.append(labels_tensor)

    features = torch.cat(chunk_list)
    labels = torch.cat(label_list)

    train_loader, val_loader, test_loader, num_inputs = prepare_data(features, labels)

    return train_loader, val_loader, test_loader, num_inputs


def prepare_data(features, labels, train_split=0.7, val_split=0.2):

    num_samples = len(features)
    train_size = int(train_split * num_samples)
    val_size = int(val_split * num_samples)

    index = np.random.permutation(num_samples)
    train_index = index[:train_size]
    val_index = index[train_size:train_size + val_size]
    test_index = index[train_size + val_size:]

    train_features, train_labels = features[train_index], labels[train_index]
    val_features, val_labels = features[val_index], labels[val_index]
    test_features, test_labels = features[test_index], labels[test_index]

    # Normalize
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    train_features = torch.tensor(train_features).float()
    val_features = torch.tensor(val_features).float()
    test_features = torch.tensor(test_features).float()
    train_labels = train_labels.clone().detach().float()
    val_labels = val_labels.clone().detach().float()
    test_labels = test_labels.clone().detach().float()

    train_dataset = TensorDataset(train_features, train_labels.unsqueeze(1))
    val_dataset = TensorDataset(val_features, val_labels.unsqueeze(1))
    test_dataset = TensorDataset(test_features, test_labels.unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_inputs = train_features.shape[1]

    return train_loader, val_loader, test_loader, num_inputs
