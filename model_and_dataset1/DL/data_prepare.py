import os
import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm
import esm


model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [int(x == s) for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom,
                  explicit_H=False,
                  use_chirality=True):
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'B',
        'C',
        'N',
        'O',
        'F',
        'Si',
        'P',
        'S',
        'Cl',
        'As',
        'Se',
        'Br',
        'Na',
        'I',
        'K',
        'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [0, 0
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    mol = Chem.MolFromSmiles(smiles)
    all_atom_feature = []
    for atom in mol.GetAtoms():
        all_atom_feature.append(atom_features(atom, explicit_H=explicit_H, use_chirality=use_chirality))
    all_bond_feature = []
    row, col = [], []

    for bond in mol.GetBonds():
        # This is not an error; the same bond needs to store the features twice.
        all_bond_feature.append(bond_features(bond, use_chirality=use_chirality))
        all_bond_feature.append(bond_features(bond, use_chirality=use_chirality))
        # Obtain the atom IDs at both ends of the bond.
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Both the forward and reverse directions need to be stored
        row += [start, end]
        col += [end, start]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    all_atom_feature = torch.tensor(np.array(all_atom_feature), dtype=torch.float)
    all_bond_feature = torch.tensor(np.array(all_bond_feature), dtype=torch.float)

    return all_atom_feature, edge_index, all_bond_feature


if __name__ == "__main__":

    n_fold = 5
    for i in range(1, n_fold + 1):

        with open(f'./dataset/fold_{i}/train_data.txt', 'r') as f:
            data_list_train = f.read().strip().split('\n')  # strip removes leading and trailing spaces

        with open(f'./dataset/fold_{i}/test_data.txt', 'r') as f:
            data_list_test = f.read().strip().split('\n')  # strip removes leading and trailing spaces
        len_train = len(data_list_train)
        len_test = len(data_list_test)

        data_list = data_list_train + data_list_test

        data_geo_list = []

        max_len = 0
        min_len = 99999
        count  = 0
        print(f"Processing data for fold {i}")
        for no, data in enumerate(tqdm((data_list))):
            smiles, sequence, interaction = data.strip().split()

            count += 1

            if max_len < len(sequence):
                max_len = len(sequence)
            if min_len > len(sequence):
                min_len = len(sequence)

            max_length = 4000
            if len(sequence) > max_length:
                sequence = sequence[:max_length]

            all_atom_feature, edge_index, all_bond_feature = smiles_to_graph(smiles)

            proteins = [('', sequence)]

            _, _, proteins = batch_converter(proteins)

            interaction = torch.tensor(int(float(interaction)))

            data = Data(x=all_atom_feature, edge_index=edge_index, edge_attr=all_bond_feature, y=interaction, protein=proteins)

            data_geo_list.append(data)

        dataset_train, dataset_test = data_geo_list[0:len_train], data_geo_list[len_train:]

        dir_input = './dataset/final/'
        os.makedirs(dir_input, exist_ok=True)


        torch.save(dataset_train, dir_input + f'drug-target_train_esm_4000_{i}' + ".pt")
        torch.save(dataset_test, dir_input + f'drug-target_test_esm_4000_{i}' + ".pt")


        print(f'The preprocessing of fold {i} has finished!')
        # break
