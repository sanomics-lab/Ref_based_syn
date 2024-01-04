import numpy as np
from rdkit import Chem
from itertools import combinations, product


class BreakMol:
    """
    Break the molecule into fragments.
    """

    def __init__(self, mol, lower_limit, cut_num):
        self.mol = mol
        self.lower_limit = lower_limit
        self.cut_num = cut_num
        self.chain_bonds = self._get_chain_bonds()

    def _get_chain_bonds(self):
        """
        Get the ring and chain bonds respectively.
        """
        ring_bonds = [list(i) for i in self.mol.GetRingInfo().BondRings()]
        all_ring_bonds = sum(ring_bonds, [])
        chain_bonds = [i.GetIdx() for i in self.mol.GetBonds() if i.GetIdx() not in all_ring_bonds]
        return chain_bonds

    def _get_terminal_bonds(self):
        """
        Gets all the bonds at the endpoint of the molecule.
        """
        terminal_bonds = []
        for a in self.mol.GetAtoms():
            neighbors = [i.GetIdx() for i in a.GetNeighbors()]
            if len(neighbors) == 1:
                terminal_bond = self.mol.GetBondBetweenAtoms(a.GetIdx(), neighbors[0])
                terminal_bonds.append(terminal_bond.GetIdx())
        return terminal_bonds

    @staticmethod
    def _have_adjacent_chain_bonds(bonds):
        """
        Whether a bond list contains adjacent bonds.
        """
        if len(bonds) >= 2:
            com_two = list(combinations(bonds, 2))
            com_two = np.array(com_two)
            diff_two = (np.diff(com_two))
            adjacent_num = len(diff_two[diff_two <= 1])
            if adjacent_num >= 1:
                return True
        return False

    def get_break_bonds(self):
        """
        Iterate over all combinations of bonds that can be used for cutting.
        """
        cut_point = self.chain_bonds
        for bonds in combinations(cut_point, self.cut_num):
            skip = False
            cut_chain_bonds = []
            for bond in bonds:
                if type(bond) != list:
                    if bond in self._get_terminal_bonds():
                        skip = True
                        break
                    else:
                        cut_chain_bonds.append(bond)
            if skip:
                continue

            if cut_chain_bonds:
                if self._have_adjacent_chain_bonds(cut_chain_bonds):
                    continue

            yield cut_chain_bonds

    @staticmethod
    def get_frags(mol, break_bonds):
        """
        Get all fragments obtained by the list of cutting bonds.
        The fragment SMILES has dummyLabels.
        """
        res = Chem.FragmentOnBonds(mol, break_bonds, dummyLabels=[(i + 1, i + 1) for i in range(len(break_bonds))])

        res_smi = Chem.MolToSmiles(res)
        frags = [Chem.MolFromSmiles(i, sanitize=False) for i in res_smi.split('.')]
        frags_smi = []
        for frag in frags:
            BreakMol.calibrate_ar(frag)
            frags_smi.append(Chem.MolToSmiles(frag))

        frags = []
        std_frags_smi = []
        for smi in frags_smi:
            frag = Chem.MolFromSmiles(smi)
            if frag is None:
                # force to fix
                frag = Chem.MolFromSmiles(smi, sanitize=False)
                for bond in frag.GetBonds():
                    if str(bond.GetBondType()) == 'AROMATIC':
                        bond.SetBondType(Chem.BondType.SINGLE)
                for atom in frag.GetAtoms():
                    atom.SetIsAromatic(False)
                frag_smi = Chem.MolToSmiles(frag)
                std_frags_smi.append(frag_smi)
                frag = Chem.MolFromSmiles(frag_smi)
                if frag is None:
                    print('Unreadable frag SMILES:', smi)
                    return
            else:
                std_frags_smi.append(smi)
            frags.append(frag)

        # get original atom index
        ori_index_list = []
        try:
            final_frags_smi = '.'.join(frags_smi)
            Chem.GetMolFrags(Chem.MolFromSmiles(final_frags_smi), asMols=True, fragsMolAtomMapping=ori_index_list)
            return frags, ori_index_list
        except Exception:
            return frags, []

    def enumerate_break(self):
        """
        Enumerate all cases of break molecules.
        """
        for break_bonds in self.get_break_bonds():
            frags, ori_index = self.get_frags(self.mol, break_bonds)
            if not frags:
                continue
            if not all([True if frag.GetNumAtoms() >= self.lower_limit else False for frag in frags]):
                continue
            yield frags, break_bonds, ori_index

    @staticmethod
    def calibrate_ar(mol):
        """
        Fix the aromatic model for a mol object.
        """

        def check(objects):
            for obj in objects:
                if not obj.IsInRing():
                    if obj.GetIsAromatic():
                        obj.SetIsAromatic(False)

        # fix bonds and atoms that aren't in the ring
        check(mol.GetAtoms())
        check(mol.GetBonds())

        # get AR ring and AL ring
        aromatic_atoms = [i.GetIdx() for i in mol.GetAromaticAtoms()]
        ar_ring_atom = []
        al_ring = []
        for ring in mol.GetRingInfo().AtomRings():
            if set(ring) & set(aromatic_atoms) == set(ring):
                ar_ring_atom.extend(ring)
            else:
                al_ring.append(ring)

        # fix atoms that are not in the aromatic ring
        for ring in al_ring:
            for i in ring:
                if i not in ar_ring_atom:
                    mol.GetAtomWithIdx(i).SetIsAromatic(False)

        # fix bonds that are not in the aromatic ring, set bond type to SINGLE
        for ring in mol.GetRingInfo().BondRings():
            for i in ring:
                bond = mol.GetBondWithIdx(i)
                if (not bond.GetBeginAtom().GetIsAromatic()) or (not bond.GetEndAtom().GetIsAromatic()):
                    bond.SetIsAromatic(False)
                    if str(bond.GetBondType()) == 'AROMATIC':
                        bond.SetBondType(Chem.BondType.SINGLE)

        # fix aromatic bond in chain
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                if str(bond.GetBondType()) == 'AROMATIC':
                    bond.SetBondType(Chem.BondType.SINGLE)
