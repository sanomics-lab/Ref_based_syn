import os
import numpy as np
from rdkit import Chem
from decomposer import BreakMol
from tqdm import tqdm


def strip_dummy_atoms(mol):
    dummy = Chem.MolFromSmiles('[*]')
    hydrogen = Chem.MolFromSmiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    return Chem.RemoveHs(mols[0])  

def get_morgen_fingerprint(smiles, nBits=4096):
    if smiles is None:
        return np.zeros(nBits).reshape((-1, )).tolist()
    else:
        mol = Chem.MolFromSmiles(smiles)
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_vec, features)
        return features.reshape((-1, )).tolist()

def search_fragements(smiles, building_blocks, bb_emb, nBit=256):
    fp1 = get_morgen_fingerprint(smiles, nBit)
    sim_list = []
    for i in range(len(bb_emb)):
        fp2 = np.array(bb_emb[i])
        tanimoto_smi = np.sum(fp1 * fp2) / (np.sum(fp1) + np.sum(fp2) - np.sum(fp1 * fp2))
        sim_list.append(tanimoto_smi)
    sim_arr = np.array(sim_list)
    rank_idx = sim_arr.argsort()[::-1]
    similar_frags = []
    threshold = 0.5
    for i in range(top_k):
        if sim_arr[rank_idx[i]] >= threshold:
            similar_frags.append(building_blocks[rank_idx[i]])

    return list(set(similar_frags))

ref_smi = ['CC1=CC=C(NC(=O)CCCN2CCN(C/C=C/C3=CC=CC=C3)CC2)C(C)=C1']
query_frags = []
for rs in ref_smi:  
    mol = Chem.MolFromSmiles(rs)
    for i in range(1, 3):
        bm = BreakMol(mol, lower_limit=5, cut_num=2)
        for frags, break_bonds, ori_index in bm.enumerate_break():
            frags = [Chem.MolToSmiles(strip_dummy_atoms(i)) for i in frags]
            query_frags.extend(frags)
    query_frags = list(set(query_frags))
print('query_frags :', len(query_frags))
with open('data/query_frags.txt', 'w') as f:
    for frag in query_frags:
        f.write(frag + '\n') 

all_bbs = []
with open('data/zinc_frags_in_stock.txt', 'r') as f:
    building_blocks = [l.strip().split()[0] for l in f.readlines()]
bb_fps = np.load('data/zinc_emb_fp_1024.npy')
for cf in tqdm(query_frags):
    similar_bbs = search_fragements(cf, building_blocks, bb_fps, nBit=256)
    all_bbs.extend(similar_bbs)
all_bbs = list(set(all_bbs))
print('all_bbs :', len(all_bbs))

with open('data/similar_bbs.txt', 'w') as f:
    for bbs in all_bbs:
        f.write(bbs + '\n')