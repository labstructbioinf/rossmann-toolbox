import pytest
import os
import numpy as np
from Bio import SeqIO
from rossmann_toolbox import RossmannToolbox


class TestHHSearch:
    @pytest.mark.hhsearch
    def test_core_detection_evaluation_hhsearch(self):
        data = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('test-data/fasta/full_length.fas', 'fasta')}
        rtb = RossmannToolbox(use_gpu=False, hhsearch_loc=os.environ['HHSEARCH'])
        preds = rtb.predict(data, mode='seq', core_detect_mode='hhsearch', importance=False)
        assert preds['3m6i_A']['sequence'] == 'VLICGAGPIGLITMLCAKAAGACPLVITDIDE'
        pr_arr = np.asarray(list(preds['3m6i_A'].values())[0:-1])
        ref_arr = np.load('test-data/ref/seq_full_length_detect_eval_hhsearch.npy')
        assert np.square(pr_arr.flatten() - ref_arr.flatten()).mean(axis=0) < 10e-5
