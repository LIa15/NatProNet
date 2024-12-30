import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist
import pandas as pd
import os


def get_fps(mol_list):
    fps = []
    for mol in mol_list:
        mol = Chem.MolFromSmiles(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
        fps.append(fp)
    return fps


def compound_clustering(input_file):
    df = pd.read_csv(input_file, header=None)
    ligand_list = (df.iloc[:, 0].values).tolist()
    mol_list = (df.iloc[:, 1].values).tolist()
    print('start nps clustering...')
    fps = get_fps(mol_list)
    C_dist = pdist(fps, 'jaccard')
    C_link = single(C_dist)
    # for thre in [0.3, 0.4, 0.5, 0.6]:
    for thre in [0.3]:
        print(thre)
        C_clusters = fcluster(C_link, thre, 'distance')
        len_list = []
        for i in range(1, max(C_clusters)+1):
            len_list.append(C_clusters.tolist().count(i))
        print('thre', thre, 'total num of compounds', len(ligand_list), 'num of clusters', max(C_clusters), 'max length', max(len_list))
        C_cluster_dict = {ligand_list[i]: C_clusters[i] for i in range(len(ligand_list))}

        # 确保目录存在
        os.makedirs(os.path.dirname('data/nps_feature/drug_cluster_{}.pkl'.format(thre)), exist_ok=True)

        with open('data/nps_feature/drug_cluster_{}.pkl'.format(thre), 'wb') as f:
            pickle.dump(C_cluster_dict, f, protocol=0)


if __name__ == "__main__":
    input_data_path = "./data/nps_file_6.csv"
    compound_clustering(input_data_path)

