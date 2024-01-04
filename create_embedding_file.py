import numpy as np
import multiprocessing as mp
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs


def get_fingerprint(smi, radius, nBits):
    if smi is None:
        return np.zeros(_nBits).reshape((-1, )).tolist()
    else:
        mol = Chem.MolFromSmiles(smi)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape((-1, )).tolist()

def worker(smi):
    return get_fingerprint(smi, radius=2, nBits=1024)

with open('data/zinc_frags_in_stock.txt', 'r') as f:
    data = [l.strip().split()[0] for l in f.readlines()]
print('the number of fragements =', len(data))

with mp.Pool(processes=20) as pool:
    embeddings = pool.map(worker, data)
    
embedding = np.array(embeddings)
output = 'data/zinc_emb_fp_1024.npy'
np.save(f'data/{output}', embeddings)
