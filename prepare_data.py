import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, AtomPairs
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintsFromMols

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

    # data['Molecule'] = data['SMILES'].apply(Chem.MolFromSmiles)

    # data['AtomPair'] = data['Molecule'].apply(lambda mol: AtomPairs.Pairs.GetAtomPairFingerprintAsBitVect(mol))
    # data['Topological'] = data['Molecule'].apply(lambda mol: FingerprintsFromMols(mol))
    # data['MorganFP'] = data['Molecule'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    # data['MACCS'] = data['Molecule'].apply(lambda mol: MACCSkeys.GenMACCSKeys(mol))

    # data['MolWeight'] = data['Molecule'].apply(Descriptors.MolWt)
    # data['NumAtoms'] = data['Molecule'].apply(Descriptors.HeavyAtomCount)
    # data['NumRings'] = data['Molecule'].apply(Descriptors.RingCount)

    features_tensor = torch.cat(chunk_list)
    labels_tensor = torch.cat(label_list)

    train_loader, val_loader, test_loader, num_inputs = prepare_data(features_tensor, labels_tensor)
    return train_loader, val_loader, test_loader, num_inputs


def prepare_data(features, labels, train_split=0.7, val_split=0.2):

    #topo_fps = torch.tensor(data['Topological'].tolist())
    # morgan_fps = torch.tensor(data['MorganFP'].tolist()).float()
    # maccs_fps = torch.tensor(data['MACCS'].tolist()).float()
    # mol_weights = torch.tensor(data['MolWeight'].values).float()
    # num_atoms = torch.tensor(data['NumAtoms'].values).float()
    # num_rings = torch.tensor(data['NumRings'].values).float()

    num_samples = len(features)
    train_size = int(train_split * num_samples)
    val_size = int(val_split * num_samples)
    test_size = num_samples - train_size - val_size

    index = np.random.permutation(num_samples)
    train_index = index[:train_size]
    val_index = index[train_size:train_size + val_size]
    test_index = index[train_size + val_size:]

    # train_morgan_fps, train_maccs_fps, train_mol_weights, train_num_atoms, train_num_rings, train_labels = \
    #     morgan_fps[train_index], maccs_fps[train_index], mol_weights[train_index], \
    #     num_atoms[train_index], num_rings[train_index], labels[train_index]

    # val_morgan_fps, val_maccs_fps, val_mol_weights, val_num_atoms, val_num_rings, val_labels = \
    #     morgan_fps[val_index], maccs_fps[val_index], mol_weights[val_index], \
    #     num_atoms[val_index], num_rings[val_index], labels[val_index]

    # train_features = torch.cat((train_morgan_fps, train_maccs_fps, train_mol_weights.unsqueeze(1),
    #                             train_num_atoms.unsqueeze(1), train_num_rings.unsqueeze(1)), dim=1)
    # val_features = torch.cat((val_morgan_fps, val_maccs_fps, val_mol_weights.unsqueeze(1),
    #                            val_num_atoms.unsqueeze(1), val_num_rings.unsqueeze(1)), dim=1)

    train_features, train_labels = features[train_index], labels[train_index]
    val_features, val_labels = features[val_index], labels[val_index]
    test_features, test_labels = features[test_index], labels[test_index]

    train_dataset = TensorDataset(train_features, train_labels.unsqueeze(1))
    val_dataset = TensorDataset(val_features, val_labels.unsqueeze(1))
    test_dataset = TensorDataset(test_features, test_labels.unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_inputs = train_features.shape[1]

    return train_loader, val_loader, test_loader, num_inputs