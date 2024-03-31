import numpy as np
import pandas as pd
import torch
from torch.utils import data
from d2l import torch as d2l
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Fingerprints, AtomPairs

# Leer el archivo CSV
file_name = 'SMILES_Big_Data_Set.csv'
data = pd.read_csv(file_name)

data['Molecule'] = data['SMILES'].apply(Chem.MolFromSmiles)

# If only for molecules
molecules = torch.tensor(data['Molecule'].tolist())

labels = torch.tensor(data['label'].values)

num_samples = len(data)
train_size = int(0.7 * num_samples)
test_size = num_samples - train_size

index = np.random.permutation(num_samples)
train_index = index[:train_size]
test_index = index[train_size:]

train_molecules = molecules[train_index]
test_molecules = molecules[test_index]
train_labels = labels[train_index]
test_labels = labels[test_index]

# If it is for fingerprints and descriptors

#data['Topological'] = data['Molecule'].apply(lambda mol: Fingerprints.FingerprintMol.FingerprintsFromMols(mol))
#data['AtomPair'] = data['Molecule'].apply(lambda mol: AtomPairs.Pairs.GetAtomPairFingerprintAsBitVect(mol))
data['MorganFP'] = data['Molecule'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
data['MACCS'] = data['Molecule'].apply(lambda mol: MACCSkeys.GenMACCSKeys(mol))

data['MolWeight'] = data['Molecule'].apply(Descriptors.MolWt)
data['NumAtoms'] = data['Molecule'].apply(Descriptors.HeavyAtomCount)
data['NumRings'] = data['Molecule'].apply(Descriptors.RingCount)

morgan_fps = torch.tensor(data['MorganFP'].tolist())
maccs_fps = torch.tensor(data['MACCS'].tolist())
mol_weights = torch.tensor(data['MolWeight'].values)
num_atoms = torch.tensor(data['NumAtoms'].values)
num_rings = torch.tensor(data['NumRings'].values)

labels = torch.tensor(data['label'].values)

num_samples = len(data)
train_size = int(0.7 * num_samples)
test_size = num_samples - train_size

index = np.random.permutation(num_samples)
train_index = index[:train_size]
test_index = index[train_size:]

train_morgan_fps, train_maccs_fps, train_mol_weights, train_num_atoms, train_num_rings, train_labels = \
    morgan_fps[train_index], maccs_fps[train_index], mol_weights[train_index], \
    num_atoms[train_index], num_rings[train_index], labels[train_index]

test_morgan_fps, test_maccs_fps, test_mol_weights, test_num_atoms, test_num_rings, test_labels = \
    morgan_fps[test_index], maccs_fps[test_index], mol_weights[test_index], \
    num_atoms[test_index], num_rings[test_index], labels[test_index]