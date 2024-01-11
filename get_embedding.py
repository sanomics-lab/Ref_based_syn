import argparse
import numpy as np
import multiprocessing as mp
from utils import get_morgen_fingerprint

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/zinc_frags_in_stock.txt')
parser.add_argument('--output', type=str, default='data/zinc_emb_fp_256.npy')
parser.add_argument('--feature', type=int, default=256)
args = parser.parse_args()
    
def worker(smi):
    return get_morgen_fingerprint(smi, nBits=args.feature)

with open(args.input, 'r') as f:
    data = [l.strip().split()[0] for l in f.readlines()]
print('the number of fragements =', len(data))

with mp.Pool(processes=20) as pool:
    embeddings = pool.map(worker, data)

np.save(args.output, np.array(embeddings))
