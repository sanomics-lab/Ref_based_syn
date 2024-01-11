import numpy as np
import pandas as pd
import random
import gym
import copy
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Descriptors import ExactMolWt
from data_utils import ReactionSet
from utils import get_reaction_mask, get_available_list, search_with_tanimoto, get_morgen_fingerprint


def get_rewards(args, predictor, smiles):
    if args.predictor == 'vina':
        reward = - np.array(predictor.predict(smiles))
    return reward

def get_mw(mol):
    return round(ExactMolWt(mol), 2)

class SynthesisEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def init(self, args, start_smis, predictor, max_action=5):    
        # init smi
        self.args = args
        self.state_dim = args.state_dim
        self.mol_mw = 500
        self.bb_emb = np.load('data/matched_bbs_emb_256.npy')
        self.building_blocks = [line.strip() for line in open('data/matched_bbs.txt', 'r')]
        self.bb_dict = {self.building_blocks[i]: i for i in range(len(self.building_blocks))}
        
        rxn_set = ReactionSet()
        rxn_set.load('data/data_for_reaction_filtered.json.gz')
        self.rxns = rxn_set.rxns
        self.rxn_class = len(self.rxns)
        print('rxn num', self.rxn_class)
        self.reward_predictor = predictor
        
        self.cand_smiles = start_smis
        self.starting_smi = random.choice(self.cand_smiles)
        self.smi = self.starting_smi 
        self.mol_recent = Chem.MolFromSmiles(self.smi)
        self.rxn_mask = get_reaction_mask(self.smi, self.rxns)
        
        self.max_action = max_action
        self.counter = 0
        self.smiles_list = []
        
    def seed(self,seed):
        np.random.seed(seed=seed)
        random.seed(seed)
    
    def step(self, ob, act, rxn_hot):
        ob_ = {'smi': None, 'ecfp': get_morgen_fingerprint(None, self.state_dim)}
        reward = 0.0
        done = False
        info = []
        self.smi_old = ob['smi']
        
        rxn_id = np.argmax(rxn_hot)    
        select_rxn = self.rxns[rxn_id]
                
        if self.counter >= self.max_action: 
            done = True
            print('counter >= max action {}'.format(self.max_action))
            
        if get_mw(self.mol_recent) > self.mol_mw:
            done = True
            print('molecule weight > max mw')
            
        if not done:
            if select_rxn.num_reactant == 1: 
                flag = select_rxn.is_reactant_first(self.mol_recent)
                if flag:
                    product = select_rxn.run_reaction([self.mol_recent, None])
                    if product:
                        reward = get_rewards(self.args, self.reward_predictor, [product])[0]
                        if reward < 0:
                            print('reward is None')
                            self.mol_recent = None
                            done = True
                        else:
                            ob_.update(self.get_observation(product))
                            self.mol_recent = Chem.MolFromSmiles(product)
                            if get_mw(self.mol_recent) > self.mol_mw:
                                print('product molecule weight > max mw')
                                self.mol_recent = None
                                done = True
                            else:
                                # info[str(self.counter)] = [self.smi_old, str(rxn_id), None, product, str(reward)]
                                info = [self.smi_old, str(rxn_id), None, product, str(reward)]
                                self.counter += 1
                                self.smiles_list.append(product)
                    else:
                        print('product is None')
                        self.mol_recent = None
                        done = True
                else:  
                    print('no match template')
                    self.mol_recent = None
                    done = True
            else: 
                f1 = select_rxn.is_reactant_first(self.mol_recent)
                f2 = select_rxn.is_reactant_second(self.mol_recent)
                if f1 and (not f2):
                    tid = 0
                elif (not f1) and f2:
                    tid = 1
                else:
                    print('no match template')
                    self.mol_recent = None
                    done = True
                    return ob_, reward, done, info
                cand_r2s = self.get_valid_reactants(tid, select_rxn, act)
                if cand_r2s:
                    products = self.forward_reaction(select_rxn, cand_r2s)
                    try:
                        rewards = get_rewards(self.args, self.reward_predictor, products)
                        print('rewards', rewards, len(rewards))
                        reward = rewards[np.argmax(rewards)]
                        if reward == -99.9:
                            print('all rewards were None')
                            self.mol_recent = None
                            done = True
                            return ob_, reward, done, info
                        r2_smi = cand_r2s[np.argmax(rewards)]
                        product = products[np.argmax(rewards)]  
                        ob_.update(self.get_observation(product))
                        self.mol_recent = Chem.MolFromSmiles(product) 
                        if get_mw(self.mol_recent) > self.mol_mw:
                            print('product molecule weight > max mw')
                            self.mol_recent = None
                            done = True
                        else:
                            # info[str(self.counter)] = [self.smi_old, str(rxn_id), r2_smi, product, str(reward)]
                            info = [self.smi_old, str(rxn_id), r2_smi, product, str(reward)]
                            self.counter += 1
                            self.smiles_list.append(product)
                    except Exception as e:
                        print(e)
                        self.mol_recent = None
                        done = True
                else:
                    print('no second reactant')
                    self.mol_recent = None
                    done = True

        return ob_, reward, done, info
    
    def reset(self, smile=None): 
        self.smiles_list = []
        if smile:
            self.smi = smile
        else:
            self.smi = random.choice(self.cand_smiles)
        self.smiles_list.append(self.smi)
        self.counter = 0
        self.mol_recent = Chem.MolFromSmiles(self.smi)
        self.rxn_mask = get_reaction_mask(self.smi, self.rxns)
        a = np.array(self.rxn_mask)
        ob = self.get_observation(smi=self.smi)
        
        return ob
        
    def render(self, mode='human', close=False):
        return
    
    def get_observation(self, smi):
        ob = {}
        
        if smi:
            ecfp = get_morgen_fingerprint(smi, self.state_dim)
        else:
            mol = copy.deepcopy(self.mol)
            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol)
                ecfp = get_morgen_fingerprint(smi, self.state_dim)
            except:
                smi = None
                ecfp = None
        
        ob['smi'] = smi
        ob['ecfp'] = ecfp

        return ob
    
    def get_valid_reactants(self, tid, rxn, query_emb):
        available = get_available_list(tid, rxn) 
        try:
            available = [self.bb_dict[available[i]] for i in range(len(available))]
        except KeyError as e:
            print(e)
            return []
        
        if len(available) == 0:
            print('length of available is 0.')
            return []
        else:
            temp_emb  = self.bb_emb[available]
            ind = search_with_tanimoto(query_emb, temp_emb)
            cand_r2s  = [self.building_blocks[available[idx]] for idx in ind]
            return cand_r2s
    
    def forward_reaction(self, rxn, cand_r2s):
        mol2s = [Chem.MolFromSmiles(cand) for cand in cand_r2s]
        products = []
        for mol2 in mol2s:
            prod = rxn.run_reaction([self.mol_recent, mol2])
            try:
                mprod = Chem.MolFromSmiles(prod)
                products.append(prod)
            except:
                products.append(None)
            
        return products
        
