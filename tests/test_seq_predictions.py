import numpy as np
from rossmann_toolbox import RossmannToolbox
from Bio import SeqIO


class TestPredictions:

    def test_core_prediction(self):
        data = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('tests/data/test.fas', 'fasta')}
        rtb = RossmannToolbox(use_gpu=False)
        preds = rtb.predict(data, mode='core')
        pr_arr = np.asarray([list(preds[key].values()) for key in data.keys()])
        ref_arr = np.load('tests/data/test.npy')
        assert np.square(pr_arr.flatten() - ref_arr.flatten()).mean(axis=0) < 10e-5

    def test_full_length_prediction(self):
        data = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('tests/data/test_full.fas', 'fasta')}
        rtb = RossmannToolbox(use_gpu=False)
        preds = rtb.predict(data, mode='seq')
        pr_arr = np.asarray([list(preds[key].values()) for key in data.keys()])
        assert 'CGPGKKVGIVGLGGIGSMGTLISKAMGAETYVISRSSR' in preds['1piw_A']['sequence']
