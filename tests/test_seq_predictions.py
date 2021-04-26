import numpy as np
import pytest
from rossmann_toolbox import RossmannToolbox
from Bio import SeqIO


class TestSeqPredictions:
    @pytest.mark.parametrize('use_gpu',
                             [pytest.param(True, marks=pytest.mark.gpu), pytest.param(False, marks=pytest.mark.cpu)])
    def test_multiple_core_predictions(self, use_gpu):
        data = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('test-data/fasta/core_benchmark.fas', 'fasta')}
        rtb = RossmannToolbox(use_gpu=use_gpu)
        preds = rtb.predict(data, mode='core')
        pr_arr = np.asarray([list(preds[key].values()) for key in data.keys()])
        ref_arr = np.load('test-data/ref/core_benchmark.npy')
        assert np.square(pr_arr.flatten() - ref_arr.flatten()).mean(axis=0) < 10e-5

    @pytest.mark.parametrize('use_gpu',
                             [pytest.param(True, marks=pytest.mark.gpu), pytest.param(False, marks=pytest.mark.cpu)])
    def test_core_detection(self, use_gpu):
        data = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('test-data/fasta/full_length.fas', 'fasta')}
        rtb = RossmannToolbox(use_gpu=use_gpu)
        cores = rtb.seq_detect_cores(data)
        assert len(cores['3m6i_A'][0]) == 1 # One core should be found
        beg, end, prob = cores['3m6i_A'][0][0]
        assert data['3m6i_A'][beg:end] == 'AGVRLGDPVLICGAGPIGLITMLCAKAAGACPLVITDIDEGRL' # Match the sequence of detected core
        pr_arr = cores['3m6i_A'][1]
        ref_arr = np.load('test-data/ref/seq_full_length_core_dl.npy')
        assert np.square(pr_arr.flatten() - ref_arr.flatten()).mean(axis=0) < 10e-5 # Predictions

    @pytest.mark.parametrize('use_gpu',
                             [pytest.param(True, marks=pytest.mark.gpu), pytest.param(False, marks=pytest.mark.cpu)])
    def test_core_detection_evaluation_importance(self, use_gpu):
        data = {str(entry.id): str(entry.seq) for entry in SeqIO.parse('test-data/fasta/full_length.fas', 'fasta')}
        rtb = RossmannToolbox(use_gpu=use_gpu)
        preds, importance = rtb.predict(data, mode='seq', core_detect_mode='dl', importance=True)
        pr_imp_arr = np.asarray(list(importance['3m6i_A'].values()))
        pr_arr = np.asarray(list(preds['3m6i_A'].values())[0:-1])
        ref_imp_arr = np.load('test-data/ref/seq_full_length_detect_eval_importance.npy')
        ref_arr = np.load('test-data/ref/seq_full_length_detect_eval.npy')
        assert np.square(pr_arr.flatten() - ref_arr.flatten()).mean(axis=0) < 10e-5 # Predictions
        assert np.square(pr_imp_arr.flatten() - ref_imp_arr.flatten()).mean(axis=0) < 10e-5 # Importances
