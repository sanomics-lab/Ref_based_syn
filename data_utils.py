'''
Code from https://github.com/wenhao-gao/SynNet
'''

from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
import gzip, json


class Reaction:
    """
    This class models a chemical reaction based on a SMARTS transformation.

    Args:
        template (str): SMARTS string representing a chemical reaction.
        rxnname (str): The name of the reaction for downstream analysis.
        smiles: (str): A reaction SMILES string that macthes the SMARTS pattern.
        reference (str): Reference information for the reaction.
    """
    def __init__(self, template=None, rxnname=None, smiles=None, reference=None):

        if template is not None:
            # define a few attributes based on the input
            self.smirks    = template
            self.rxnname   = rxnname
            self.smiles    = smiles
            self.reference = reference

            # compute a few additional attributes
            rxn = AllChem.ReactionFromSmarts(self.smirks)
            rdChemReactions.ChemicalReaction.Initialize(rxn)
            self.num_reactant = rxn.GetNumReactantTemplates()
            if self.num_reactant == 0 or self.num_reactant > 2:
                raise ValueError('This reaction is neither uni- nor bi-molecular.')
            self.num_product = rxn.GetNumProductTemplates()
            if self.num_reactant == 1:
                self.reactant_template = list((self.smirks.split('>')[0], ))
            else:
                self.reactant_template = list((self.smirks.split('>')[0].split('.')[0], \
                                                self.smirks.split('>')[0].split('.')[1]))
            self.product_template = self.smirks.split('>')[2]

            del rxn
        else:
            self.smirks = None
        
    def load(self, smirks, num_reactant, num_product, reactant_template,
             product_template, available_reactants, rxnname, smiles, reference):
        """
        This function loads a set of elements and reconstructs a `Reaction` object.
        """
        self.smirks              = smirks
        self.num_reactant        = num_reactant
        self.num_product         = num_product
        self.reactant_template   = list(reactant_template)
        self.product_template    = product_template
        self.available_reactants = list(available_reactants)
        self.rxnname             = rxnname
        self.smiles              = smiles
        self.reference           = reference
    
    def get_mol(self, smi):
        """
        A internal function that returns an `RDKit.Chem.Mol` object.
        """
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError('The input should be either a SMILES string or an '
                            'RDKit.Chem.Mol object.')
    
    def is_reactant(self, mol):
        """
        A function that checks if a molecule is a reactant of the reaction
        defined by the `Reaction` object.
        """
        result = False
        if self.num_reactant == 1:
            template = self.reactant_template[0]
            if self.is_reactant_first(mol):
                result = True
        elif self.num_reactant == 2:
            f1 = self.is_reactant_first(mol)
            f2 = self.is_reactant_second(mol)
            template1, template2 = self.reactant_template
            if f1 and not f2:
                result = True
            elif not f1 and f2:
                result = True
        return result
    
    def is_single(self, template, smi):
        """
        A function that checks if a molecule only once mactch the given template 
        """
        mol = self.get_mol(smi)
        patt = Chem.MolFromSmarts(template)
        matches = mol.GetSubstructMatches(patt, useChirality=True)
        if len(matches) == 1:
            return True
        else:
            return False
        
    def get_reactant_template(self, ind=0):
        """
        A function that returns the SMARTS pattern which represents the specified
        reactant.
        """
        return self.reactant_template[ind]
    
    def is_reactant_first(self, smi):
        """
        A function that checks if a molecule is the first reactant in the reaction
        defined by the `Reaction` object, where the order of the reactants is
        determined by the SMARTS pattern.
        """
        if smi.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(0))):
            return True
        else:
            return False
        
    def is_reactant_second(self, smi):
        """
        A function that checks if a molecule is the second reactant in the reaction
        defined by the `Reaction` object, where the order of the reactants is
        determined by the SMARTS pattern.
        """
        if smi.HasSubstructMatch(Chem.MolFromSmarts(self.get_reactant_template(1))):
            return True
        else:
            return False
        
    def _filter_reactants(self, smi_list):
        """
        Filters reactants which do not match the reaction.
        Args:
            smi_list (list): Contains SMILES to search through for matches.
        Returns:
            tuple: Contains list(s) of SMILES which match either the first
                reactant, or, if applicable, the second reactant.
        """
        if self.num_reactant == 1:  
            smi_w_patt = []
            for smi in tqdm(smi_list):
                mol = Chem.MolFromSmiles(smi)
                if self.is_reactant_first(mol): 
                    smi_w_patt.append(smi)
            return (smi_w_patt, )

        elif self.num_reactant == 2:  
            smi_w_patt1 = []
            smi_w_patt2 = []
            for smi in tqdm(smi_list):
                mol = Chem.MolFromSmiles(smi)
                f1 = self.is_reactant_first(mol)
                f2 = self.is_reactant_second(mol)
                if f1 and not f2: 
                    smi_w_patt1.append(smi)
                elif f2 and not f1: 
                    smi_w_patt2.append(smi)
                else: 
                    continue
            return (smi_w_patt1, smi_w_patt2)
        else:
            raise ValueError('This reaction is neither uni- nor bi-molecular.')
        
    def set_available_reactants(self, building_block_list):
        """
        A function that finds the applicable building blocks from a list of
        purchasable building blocks.

        Args:
            building_block_list (list): The list of purchasable building blocks,
                where building blocks are represented as SMILES strings.
        """
        self.available_reactants = list(self._filter_reactants(building_block_list))
        
        return None
    
    def run_reaction(self, reactants, keep_main=True):
        """
        A function that transform the reactants into the corresponding product.
        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)

        if self.num_reactant == 1:

            if isinstance(reactants, (tuple, list)): 
                if len(reactants) == 1:
                    rmol = self.get_mol(reactants[0])
                elif len(reactants) == 2 and reactants[1] is None:
                    rmol = self.get_mol(reactants[0])
                else:
                    return None
            else:
                raise TypeError('The input of a uni-molecular reaction should '
                                'be a SMILES, an rdkit.Chem.Mol object, or a '
                                'tuple/list of length 1 or 2.')

            if not self.is_reactant(rmol):
                return None

            ps = rxn.RunReactants((rmol, ))

        elif self.num_reactant == 2:
            if isinstance(reactants, (tuple, list)) and len(reactants) == 2:
                r1 = self.get_mol(reactants[0])
                r2 = self.get_mol(reactants[1])
            else:
                raise TypeError('The input of a bi-molecular reaction should '
                                'be a tuple/list of length 2.')

            if self.is_reactant_first(r1) and self.is_reactant_second(r2):
                pass
            elif self.is_reactant_first(r2) and self.is_reactant_second(r1):
                r1, r2 = (r2, r1)
            else:
                return None

            ps = rxn.RunReactants((r1, r2))

        else:
            raise ValueError('This reaction is neither uni- nor bi-molecular.')

        uniqps = []
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                smi = Chem.MolToSmiles(p[0])
                uniqps.append(smi)
            except:
                continue

        uniqps = list(set(uniqps))
        
        if len(uniqps) < 1:
            return None

        del rxn

        if keep_main: 
            import random
            return random.choice(uniqps)
        else: 
            return uniqps


class ReactionSet:
    """
    A class representing a set of reactions, for saving and loading purposes.
    """
    def __init__(self, rxns=None):
        if rxns is None:
            self.rxns = []
        else:
            self.rxns = rxns
            
    def load(self, json_file):
        """
        A function that loads reactions from a JSON-formatted file.
        """
        with gzip.open(json_file, 'r') as f:
            data = json.loads(f.read().decode('utf-8'))

        for r_dict in data['reactions']:
            r = Reaction()
            r.load(**r_dict)
            self.rxns.append(r)
    
    def save(self, json_file):
        """
        A function that saves the reaction set to a JSON-formatted file.
        """
        r_list = {'reactions': [r.__dict__ for r in self.rxns]}
        with gzip.open(json_file, 'w') as f:
            f.write(json.dumps(r_list, indent=4).encode('utf-8'))
    
    def __len__(self):
        return len(self.rxns)
