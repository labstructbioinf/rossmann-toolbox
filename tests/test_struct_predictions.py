import os
import warnings; warnings.filterwarnings("ignore")
import pytest
import numpy as np
from Bio import SeqIO
from rossmann_toolbox import RossmannToolbox


class TestStructPredictions:

    @staticmethod
    def _clean_cache_file():
        cache_dir = 'test-data/pdb/struct/'
        fns = os.listdir(cache_dir)
        fns.remove('3m6i_A.pdb')
        for fn in fns:
            os.remove(f'{cache_dir}/{fn}')

    @pytest.mark.struct
    def test_core_evaluation_importance(self):
        self._clean_cache_file()
        rtb = RossmannToolbox(use_gpu=False, foldx_loc=os.environ['FOLDX'], dssp_loc=os.environ['DSSP'])
        path_to_structures = 'test-data/pdb/struct/'
        path_to_structures = os.path.abspath(path_to_structures)
        chains_to_use = ['3m6i_A']
        preds = rtb.predict_structure(path_to_structures, chains_to_use, mode='seq', core_detect_mode='dl')
        pr = list(preds[0].values())
        del pr[4:6]
        pr_arr = np.asarray(pr)
        ref_arr = np.load('test-data/ref/struct_full_length_detect_eval.npy')
        assert np.square(pr_arr.flatten() - ref_arr.flatten()).mean(axis=0) < 10e-5
        self._clean_cache_file()
