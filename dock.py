'''
Code from https://github.com/AITRICS/FREED/tree/main
'''

import os, sys
import subprocess
from multiprocessing import Manager, Process, Queue
from openbabel import pybel


class DockingVina(object):

    def __init__(self, docking_params):
        super(DockingVina, self).__init__()
        self.vina_program = docking_params['vina_program']
        self.receptor_file = docking_params['receptor_file']
        (self.box_center, self.box_size) = docking_params['box_parameter']
        self.temp_dir = docking_params['temp_dir']
        self.exhaustiveness = docking_params['exhaustiveness']
        self.num_sub_proc = docking_params['num_sub_proc']
        self.num_cpu_dock = docking_params['num_cpu_dock']
        self.num_modes = docking_params['num_modes']
        self.timeout_gen3d = docking_params['timeout_gen3d']
        self.timeout_dock = docking_params['timeout_dock']

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def docking(self, receptor_file, ligand_file, ligand_pdbqt_file, docking_pdbqt_file):
        """
            run_docking program using subprocess
            input :
                receptor_file
                ligand_file
                ligand_pdbqt_file
                docking_pdbqt_file
            output :
                affinity list for a input molecule
        """
        ms = list(pybel.readfile("mol", ligand_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' %(self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' %(self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
            
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        """
            put data to queue
            input: queue
                data = [(idx1,smi1), (idx2,smi2), ...]
                num_sub_proc (for end signal)
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        """
            generate subprocess for docking
            input
                q (queue)
                return_dict
                sub_id: subprocess index for temp file
        """
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            receptor_file = self.receptor_file
            ligand_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_file)
            except Exception as e:
                print(e)
                print("gen_3d unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            
            try:
                score_list = self.docking(receptor_file, ligand_file,
                                            ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                print(e)
                print("docking unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            
            if len(score_list)==0:
                score_list.append(99.9)
            
            score = score_list[0]
            return_dict[idx] = score

    def predict(self, smiles_list):
        """
            input SMILES list
            output score list corresponding to the SMILES list
            if docking is fail, docking score is 99.9
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator,
                              args=(q1, data, self.num_sub_proc))
        proc_master.start()

        # create slave process
        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess,
                           args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        score_list = list()
        for key in keys:
            score = return_dict[key]
            score_list += [score]
            
        return score_list
    
    def gen_3d(self, smi, ligand_file):
        """
            generate initial 3d conformation from SMILES
            input :
                SMILES string
                ligand_file (output file)
        """
        run_line = 'obabel -:%s --gen3D --minimize --steps 200 --sd --ff MMFF94 -O %s' % (smi, ligand_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, 
                                         universal_newlines=True)


def get_docking_config_for_vina():
    docking_config = dict()
    docking_config['receptor_file'] = 'data/2y9x.pdbqt' 
    docking_config['vina_program'] = './qvina02'
    box_center = (-31.96273, -2.509867, -93.069374)
    box_size = (23.473999, 16.628, 22.723999)
    docking_config['box_parameter'] = (box_center, box_size)
    docking_config['temp_dir'] = 'tmp'
    docking_config['exhaustiveness'] = 4
    docking_config['num_sub_proc'] = 10
    docking_config['num_cpu_dock'] = 3
    docking_config['num_modes'] = 10 
    docking_config['timeout_gen3d'] = 30
    docking_config['timeout_dock'] = 100 
   
    return docking_config
