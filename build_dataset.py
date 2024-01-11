import numpy as np
from rdkit import Chem
from decomposer import BreakMol
from tqdm import tqdm
from utils import *


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
    for i in range(len(rank_idx)):
        if sim_arr[rank_idx[i]] >= threshold:
            similar_frags.append(building_blocks[rank_idx[i]])
    return list(set(similar_frags))

ref_smi = 'CC1=CC=C(NC(=O)CCCN2CCN(C/C=C/C3=CC=CC=C3)CC2)C(C)=C1'
query_frags = []
mol = Chem.MolFromSmiles(ref_smi)
for c in range(1, 3):
    bm = BreakMol(mol, lower_limit=5, cut_num=c)
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
bb_fps = np.load('data/zinc_emb_fp_256.npy')
for cf in tqdm(query_frags):
    similar_bbs = search_fragements(cf, building_blocks, bb_fps, nBit=256)
    all_bbs.extend(similar_bbs)
all_bbs = list(set(all_bbs))
print('all_bbs :', len(all_bbs))
        
process_reac_file('data/rxn_set_uspto.txt', all_bbs)
