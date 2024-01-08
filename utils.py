import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem, DataStructs
from data_utils import Reaction, ReactionSet


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
    
def process_reac_file(rxn_file, building_blocks):
    rxn_templates = []
    with open(rxn_file, 'r') as f:
        for line in f.readlines():
            try:
                print('aaaa :', line.strip())
                rxn = Reaction(line.strip())
                rxn.set_available_reactants(building_blocks)
                rxn_templates.append(rxn)
            except ValueError as e:
                print(e)
                continue
            
    r = ReactionSet(rxn_templates)
    r.save('data/data_for_reaction.json.gz')
