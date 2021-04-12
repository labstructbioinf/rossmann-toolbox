import warnings
import torch
import os
import numpy as np
import pandas as pd

from zipfile import ZipFile
from rossmann_toolbox.utils import sharpen_preds, corr_seq, custom_warning
from rossmann_toolbox.utils import SeqVec
from rossmann_toolbox.utils.encoders import SeqVecMemEncoder
from rossmann_toolbox.utils.generators import SeqChunker
from rossmann_toolbox.models import SeqCoreEvaluator, SeqCoreDetector
from conditional import conditional
from captum.attr import IntegratedGradients

warnings.showwarning = custom_warning


class RossmannToolbox:

    def __init__(self, use_gpu=True):
        """
        Rossmann Toolbox - A framework for predicting and engineering the cofactor specificity of Rossmann-fold proteins
        :param use_gpu: Use GPU to speed up predictions
        """
        self.label_dict = {'FAD': 0, 'NAD': 1, 'NADP': 2, 'SAM': 3}
        self.rev_label_dict = {value: key for key, value in self.label_dict.items()}
        self.n_classes = 4

        # Handle GPU config
        self.use_gpu = False
        self.device = torch.device('cpu')
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.use_gpu = True
            else:
                warnings.warn('No GPU detected, falling back to the CPU version!')

        self._path = os.path.dirname(os.path.abspath(__file__))
        self._seqvec = self._setup_seqvec()
        self._weights_prefix = f'{self._path}/weights/'

    def _process_input(self, data, full_length=False):
        """
        Validate and process the input eventually filtering/correcting problematic sequences
        :param data: Input data - a dictionary with ids and corresponding sequences as keys and values
        :param full_length: Denotes whether full-length sequences are validated (True/False)
        :return: pd.DataFrame for internal processing
        """
        if not isinstance(data, dict):
            raise ValueError('Input data must be a dictionary with ids and sequences as keys and values!')
        valid_seqs = [isinstance(seq, str) for seq in data.values()]
        if not all(valid_seqs):
            raise ValueError('Input data must be a dictionary with ids and sequences as keys and values!')
        if len(data) == 0:
            raise ValueError('Empty dictionary was passed as an input!')
        non_std_seqs = [sequence != corr_seq(sequence) for sequence in data.values()]
        if not full_length:
            too_short_seqs = [len(sequence) < 20 for sequence in data.values()]
            too_long_seqs = [len(sequence) > 65 for sequence in data.values()]
            if all(too_short_seqs):
                raise ValueError('Input sequence(s) are below 20aa length!')

            if all(too_short_seqs):
                raise ValueError('Input sequence(s) are above 65aa length!')

        # Convert input dict to pd.DataFrame for easy handling
        data = pd.DataFrame.from_dict(data, orient='index', columns=['sequence'])

        if any(non_std_seqs):
            data['sequence'] = data['sequence'].apply(lambda x: corr_seq(x))
            warnings.warn('Non-standard residues detected in input data were corrected to X token.', UserWarning)
        if not full_length:
            if any(too_short_seqs):
                data = data[data['sequence'].str.len() >= 20]
                warnings.warn('Filtered out input sequences shorter than 20 residues.', UserWarning)

            if any(too_long_seqs):
                data = data[data['sequence'].str.len() <= 65]
                warnings.warn('Filtered out input sequences longer than 65 residues.', UserWarning)

        return data

    def _setup_seqvec(self):
        """
        Load SeqVec model to either GPU or CPU, if weights are not cached - download them from the mirror.
        :return: instance of SeqVec embedder
        """
        seqvec_dir = f'{self._path}/weights/seqvec'
        seqvec_conf_fn = f'{seqvec_dir}/uniref50_v2/options.json'
        seqvec_weights_fn = f'{seqvec_dir}/uniref50_v2/weights.hdf5'
        if not (os.path.isfile(seqvec_conf_fn) and os.path.isfile(seqvec_weights_fn)):
            print('SeqVec weights are not available, downloading from the remote source (this\'ll happen only once)...')
            torch.hub.download_url_to_file('https://rostlab.org/~deepppi/seqvec.zip',
                                           f'{self._path}/weights/seqvec.zip')
            archive = ZipFile(f'{self._path}/weights/seqvec.zip')
            archive.extract('uniref50_v2/options.json', seqvec_dir)
            archive.extract('uniref50_v2/weights.hdf5', seqvec_dir)
        if self.use_gpu:
            return SeqVec(model_dir=f'{seqvec_dir}/uniref50_v2', cuda_device=0, tokens_per_batch=8000)
        else:
            return SeqVec(model_dir=f'{seqvec_dir}/uniref50_v2', cuda_device=-1, tokens_per_batch=8000)

    def _setup_seq_core_detector(self):
        return SeqCoreDetector().to(self.device)

    def _setup_seq_core_evaluator(self):
        return SeqCoreEvaluator().to(self.device)

    def _setup_struct_core_evaluator(self):
        raise NotImplementedError('TODO')

    def seq_detect_cores(self, data, return_probs = False):
        """
        Detects Rossmann beta-alpha-beta cores in full-length protein sequences.
        :param data: Input data - a dictionary with ids and corresponding sequences as keys and values
        :param return_probs: return additional per-residue probabilities (True/False)
        :return: dictionary with ids and detected core locations as keys and values. If return_probs is True the values
        of the returned dict are (detected cores, probability profile).
        """

        data = self._process_input(data, full_length=True)

        # Prepare embeddings and models
        embeddings = self._seqvec.encode(data, to_file=False)
        seqvec_enc = SeqVecMemEncoder(embeddings, pad_length=500)
        gen = SeqChunker(data, batch_size=64, W_size=500, shuffle=False,
                         data_encoders=[seqvec_enc], data_cols=['sequence'])
        model = self._setup_seq_core_detector()
        model.load_state_dict(torch.load(f'{self._weights_prefix}/coredetector.pt'))
        model = model.eval()
        preds = []

        # Predict core locations
        with torch.no_grad():
            for batch in gen:
                batch = torch.tensor(batch[0].astype(dtype=np.float32), device=self.device)
                batch_preds = model(batch)
                preds.append(batch_preds.cpu().detach().numpy())

        # Post-process predictions
        preds = np.vstack(preds)
        preds_depadded = {key: np.concatenate([preds[ix][ind[0]:ind[1]] for ix, ind in zip(*value)]) for key, value
                          in gen.indices.items()}
        if return_probs:
            return {key: (sharpen_preds(value), value) for key, value in preds_depadded.items()}
        else:
            return {key: sharpen_preds(value) for key, value in preds_depadded.items()}

    def seq_evaluate_cores(self, data, importance = False):
        """
        Predicts the cofactor specitificty of the Rossmann core sequences
        :param data: Input data - a dictionary with ids and corresponding sequences as keys and values
        :param importance: Return additional per-residue importances, i.e. contributions to the final
        specificity predictions
        :return: Dictionary with the sequence ids and per-sequence predictions of the cofactor specificties.
        If importance is True each dictionary value will contain additional per-residue importances.
        """

        data = self._process_input(data)
        # Initialize importances and preds per fold values which latter will be averaged
        preds_ens = []
        attrs_ens = []

        # Encode with SeqVec
        embeddings = self._seqvec.encode(data, to_file=False)

        # Setup generator that'll be evaluated
        seqvec_enc = SeqVecMemEncoder(embeddings, pad_length=65)
        gen = SeqChunker(data, batch_size=64, W_size=65, shuffle=False,
                         data_encoders=[seqvec_enc], data_cols=['sequence'])

        # Predict with each of N predictors, depad predictions and average out for final output
        model = self._setup_seq_core_evaluator()
        model = model.eval()
        for i in range(0, 5):
            model.load_state_dict(torch.load(f'{self._weights_prefix}/{i}.pt'))
            preds = []
            attrs = []
            with conditional(not importance, torch.no_grad()):
                # Raw predictions
                for batch in gen:
                    batch = torch.tensor(batch[0].transpose(0, 2, 1).astype(dtype=np.float32), device=self.device)
                    batch_preds = model(batch)
                    preds.append(batch_preds.cpu().detach().numpy())
                    if importance:
                        ig = IntegratedGradients(model)
                        baseline = torch.full_like(batch, 0, device=self.device)
                        batch_attrs = np.asarray(
                            [ig.attribute(batch, baseline, i).sum(axis=1).clip(min=0).cpu().detach().numpy() for i in
                             range(0, self.n_classes)])
                        attrs.append(batch_attrs.transpose(1, 0, 2))

            preds_ens.append(np.vstack(preds))
            if importance:
                attrs_ens.append(np.vstack(attrs))

        # Average predictions between all predictors
        preds_ens = np.asarray(preds_ens)
        attrs_ens = np.asarray(attrs_ens)
        avgs = preds_ens.mean(axis=0)
        stds = preds_ens.std(axis=0)
        results = {key: {f'{self.rev_label_dict[i]}{suf}': val[i] for i in range(0, self.n_classes) for suf, val in
                         zip(['', '_std'], [avg, std])} for key, avg, std in zip(data.index, avgs, stds)}
        if importance:
            avgs = attrs_ens.mean(axis=0)
            stds = attrs_ens.std(axis=0)
            attrs = {key: {f'{self.rev_label_dict[j]}':
                               ((np.concatenate([avgs[ix, j, ind[0]:ind[1]] for ix, ind in zip(*value)]),
                                 np.concatenate([stds[ix, j, ind[0]:ind[1]] for ix, ind in zip(*value)]))
                               ) for j in range(0, self.n_classes)}
                     for key, value in gen.indices.items()}
            return results, attrs
        return results

    def struct_evaluate_cores(self):
        raise NotImplementedError('TODO')

    def predict(self, data, mode = 'core', importance = False):
        """
        Evaluate cofactor specificity of full-length or Rossmann-core sequences.
        :param data: Input data - a dictionary with ids and corresponding sequences as keys and values
        :param mode: Prediction mode - either 'seq' for full-sequence input or 'core' for Rossmann-core sequences
        :param importance: Return additional per-residue importances, i.e. contributions to the final
        specificity predictions
        :return: Dictionary with the sequence ids and per-sequence predictions of the cofactor specificties.
        If importance is True each dictionary value will contain additional per-residue importances.
        If mode is 'seq' output will additionally contain the 'sequence' keys for each input id indicating the detected
        Rossmann core sequence.
        """
        if mode not in ['seq', 'core']:
            raise ValueError('Prediction mode must be either \'seq\' or \'core\'!')

        # Full length sequence input
        if mode == 'seq':
            detected_cores = self.seq_detect_cores(data)

            # Check for undetected cores
            detected_ids = {key for key, value in detected_cores.items() if len(value) > 0}
            passed_ids = set(data.keys())
            if len(set(detected_ids)) != len(set(passed_ids)):
                missing_ids = set(passed_ids) - set(detected_ids)
                warnings.warn('Rossmann cores were not detected for ids: {}'.format(', '.join(missing_ids)))

            # Check for multiple cores in one sequence
            multiple_hits_ids = {str(key) for key, value in detected_cores.items() if len(value) > 1}
            if len(multiple_hits_ids) > 0:
                warnings.warn(
                    'Found multiple Rossmann cores for ids: \'{}\'. Passing first hit for further predictions'.format(
                        ', '.join(multiple_hits_ids)))
            cores_filtered = {key: data[key][value[0][0]:value[0][1]] for key, value in detected_cores.items() if
                              len(value) > 0}
            data = cores_filtered

        predictions = self.seq_evaluate_cores(data, importance=importance)
        if mode == 'seq':
            for key in predictions.keys():
                predictions[key]['sequence'] = data[key]
        return predictions


    def predict_structure(self):
        # TODO implement structure-based prediction
        raise NotImplementedError('TODO')
