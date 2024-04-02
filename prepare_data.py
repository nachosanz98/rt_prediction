import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, AtomPairs
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintsFromMols

def read_and_create(file_name):

    data = pd.read_csv(file_name)

    data['Molecule'] = data['SMILES'].apply(Chem.MolFromSmiles)

    #data['AtomPair'] = data['Molecule'].apply(lambda mol: AtomPairs.Pairs.GetAtomPairFingerprintAsBitVect(mol))
    #data['Topological'] = data['Molecule'].apply(lambda mol: FingerprintsFromMols(mol))
    data['MorganFP'] = data['Molecule'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    data['MACCS'] = data['Molecule'].apply(lambda mol: MACCSkeys.GenMACCSKeys(mol))

    data['MolWeight'] = data['Molecule'].apply(Descriptors.MolWt)
    data['NumAtoms'] = data['Molecule'].apply(Descriptors.HeavyAtomCount)
    data['NumRings'] = data['Molecule'].apply(Descriptors.RingCount)

    train_loader, test_loader, num_inputs = prepare_data(data)
    return train_loader, test_loader, num_inputs


def prepare_data(data, train_split=0.7):

    #topo_fps = torch.tensor(data['Topological'].tolist())
    morgan_fps = torch.tensor(data['MorganFP'].tolist()).float()
    maccs_fps = torch.tensor(data['MACCS'].tolist()).float()
    mol_weights = torch.tensor(data['MolWeight'].values).float()
    num_atoms = torch.tensor(data['NumAtoms'].values).float()
    num_rings = torch.tensor(data['NumRings'].values).float()

    labels = torch.tensor(data['pIC50'].values).float()

    num_samples = len(data)
    train_size = int(train_split * num_samples)

    index = np.random.permutation(num_samples)
    train_index = index[:train_size]
    test_index = index[train_size:]

    train_morgan_fps, train_maccs_fps, train_mol_weights, train_num_atoms, train_num_rings, train_labels = \
        morgan_fps[train_index], maccs_fps[train_index], mol_weights[train_index], \
        num_atoms[train_index], num_rings[train_index], labels[train_index]

    test_morgan_fps, test_maccs_fps, test_mol_weights, test_num_atoms, test_num_rings, test_labels = \
        morgan_fps[test_index], maccs_fps[test_index], mol_weights[test_index], \
        num_atoms[test_index], num_rings[test_index], labels[test_index]

    train_features = torch.cat((train_morgan_fps, train_maccs_fps, train_mol_weights.unsqueeze(1),
                                train_num_atoms.unsqueeze(1), train_num_rings.unsqueeze(1)), dim=1)
    test_features = torch.cat((test_morgan_fps, test_maccs_fps, test_mol_weights.unsqueeze(1),
                               test_num_atoms.unsqueeze(1), test_num_rings.unsqueeze(1)), dim=1)

    train_dataset = TensorDataset(train_features, train_labels.unsqueeze(1))
    test_dataset = TensorDataset(test_features, test_labels.unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_inputs = train_features.shape[1]

    return train_loader, test_loader, num_inputs